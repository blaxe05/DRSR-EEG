# -*- encoding: utf-8 -*-
"""Frozen component-removal variants for DRSR source routing.

The production DRSR implementation assigns complementary source weights to
the structural and corrective objectives.  This module changes only that
assignment.  All feature extraction, losses, optimizer behavior, warm-up,
prototype updates, and inference code are inherited from the sealed parent.

``full`` delegates to the parent implementation exactly.  The four ablations
are defined as follows:

``uniform``
    Uniform structural and corrective weights.
``shared_score``
    The lower-evidence (structural) distribution is shared by both objectives.
``structural_only``
    Structural weights are retained and corrective weights are uniform.
``corrective_only``
    Corrective weights are retained and structural weights are uniform.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.RWHEDNPseudoConditional import RWHEDNPseudoConditional


ROUTING_MODES = (
    "full",
    "uniform",
    "shared_score",
    "structural_only",
    "corrective_only",
)


class RWHEDNRoutingAblation(RWHEDNPseudoConditional):
    """DRSR with one prespecified source-routing assignment."""

    def __init__(self, routing_mode: str, **kwargs):
        super().__init__(**kwargs)
        mode = str(routing_mode).strip().lower()
        if mode not in ROUTING_MODES:
            raise ValueError(f"unknown routing_mode {mode!r}")
        self.routing_mode = mode
        initial = torch.ones(self.num_sources) / self.num_sources
        self.register_buffer("last_structural_weights", initial.clone())
        self.register_buffer("last_corrective_weights", initial.clone())

    @staticmethod
    def assign_route_weights(
        score: torch.Tensor, temperature: float, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return structural and corrective simplex weights for ``mode``."""
        if score.ndim != 1 or score.numel() == 0:
            raise ValueError("score must be a non-empty one-dimensional tensor")
        tau = float(temperature)
        if tau <= 0:
            raise ValueError("temperature must be positive")
        mode = str(mode).strip().lower()
        structural = F.softmax(-score / tau, dim=0)
        corrective = F.softmax(score / tau, dim=0)
        uniform = torch.ones_like(score) / score.numel()
        if mode == "full":
            return structural, corrective
        if mode == "uniform":
            return uniform, uniform
        if mode == "shared_score":
            return structural, structural
        if mode == "structural_only":
            return structural, uniform
        if mode == "corrective_only":
            return uniform, corrective
        raise ValueError(f"unknown routing_mode {mode!r}")

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # This provides an executable equivalence control for the new class.
        if self.routing_mode == "full":
            return super().forward(srcs, tgt, src_labels, src_clusters, tgt_cluster)
        if not self.soft_sra:
            raise RuntimeError("routing ablations require soft source routing")

        srcs = srcs.permute(1, 0, 2)
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)
        num_sources = srcs.size(0)
        if num_sources != self.num_sources:
            raise RuntimeError("source-count mismatch")

        tgt_feat = self.feature_extractor(tgt)
        src_feats = [self.feature_extractor(srcs[i]) for i in range(num_sources)]
        src_lab_idx = [torch.argmax(src_labels[i], dim=1) for i in range(num_sources)]

        with torch.no_grad():
            src_logits = [self.hard_classifier(src_feats[i]) for i in range(num_sources)]
            cls_scores = torch.stack([
                self.cls_loss(src_logits[i], src_lab_idx[i]) for i in range(num_sources)
            ])
            if self.js_relevance_weight == 0.0:
                relevance_score = cls_scores
            else:
                tgt_mean_probability = F.softmax(
                    self.hard_classifier(tgt_feat), dim=1
                ).mean(dim=0)
                js_scores = torch.stack([
                    self._js_divergence(
                        F.softmax(src_logits[i], dim=1).mean(dim=0),
                        tgt_mean_probability,
                    )
                    for i in range(num_sources)
                ])
                relevance_score = (
                    self._zscore(cls_scores)
                    + self.js_relevance_weight * self._zscore(js_scores)
                )
            structural_w, corrective_w = self.assign_route_weights(
                relevance_score, self.sra_temp, self.routing_mode
            )
            self.src_reliability = (
                self.rel_momentum * self.src_reliability
                + (1.0 - self.rel_momentum) * structural_w
            ).detach()
            self.last_structural_weights.copy_(structural_w)
            self.last_corrective_weights.copy_(corrective_w)

        loss_cls = sum(
            corrective_w[i]
            * self.cls_loss(self.hard_classifier(src_feats[i]), src_lab_idx[i])
            for i in range(num_sources)
        )
        source_only = bool(self.source_only_warmup_active)
        loss_adv = (
            torch.zeros((), device=tgt_feat.device)
            if source_only
            else sum(
                corrective_w[i] * self.hard_advcriterion(src_feats[i], tgt_feat)
                for i in range(num_sources)
            )
        )

        conditional_parts = []
        if self.conditional_alignment_weight > 0.0 and not source_only:
            target_probability = F.softmax(self.hard_classifier(tgt_feat), dim=1)
            for index in range(num_sources):
                conditional_loss, _ = self._soft_pseudo_class_alignment(
                    src_feats[index], tgt_feat, src_lab_idx[index], target_probability
                )
                conditional_parts.append(corrective_w[index] * conditional_loss)
        conditional_alignment_loss = (
            sum(conditional_parts)
            if conditional_parts
            else torch.zeros((), device=tgt_feat.device)
        )
        transfer_loss = (
            loss_adv
            + self.conditional_alignment_weight * conditional_alignment_loss
        )

        src_clu_loss = sum(
            structural_w[i]
            * self.clu_loss(self.easy_network.extractor(src_feats[i]), src_clusters[i])
            for i in range(num_sources)
        )
        tgt_clu_loss = (
            torch.zeros((), device=tgt_feat.device)
            if source_only
            else self.clu_loss(self.easy_network.extractor(tgt_feat), tgt_cluster)
        )

        with torch.no_grad():
            for i in range(num_sources):
                source_embedding = self.easy_network.extractor(src_feats[i].detach())
                self.easy_network.update_source_cluster_centers(
                    source_embedding, src_clusters[i], i
                )
            if not source_only:
                target_embedding = self.easy_network.extractor(tgt_feat.detach())
                self.easy_network.update_target_cluster_centers(
                    target_embedding, tgt_cluster
                )

        if source_only:
            loss_consis = torch.zeros((), device=tgt_feat.device)
        else:
            with torch.no_grad():
                proto_dist = self._easy_soft_dist(tgt.detach())
                proto_dist = proto_dist / (
                    proto_dist.sum(dim=1, keepdim=True) + 1e-8
                )
                tgt_conf, tgt_proto_pred = proto_dist.max(dim=1)
            tgt_logit = self.hard_classifier(tgt_feat)
            tgt_proto_pred = tgt_proto_pred.to(tgt_logit.device)
            if self.conf_consis:
                threshold = torch.quantile(tgt_conf, self.conf_quantile).item()
                mask = (tgt_conf >= threshold).to(tgt_logit.device)
                loss_consis = (
                    F.cross_entropy(tgt_logit[mask], tgt_proto_pred[mask])
                    if mask.any()
                    else torch.zeros((), device=tgt_logit.device)
                )
            else:
                loss_consis = self.consis_loss(tgt_logit, tgt_proto_pred)

        with torch.no_grad():
            self.last_conditional_alignment_loss.copy_(
                conditional_alignment_loss.detach().to(
                    self.last_conditional_alignment_loss.device
                )
            )
        easy_idx = torch.argmax(structural_w)
        hard_idx = torch.argmax(corrective_w)
        return (
            loss_cls,
            transfer_loss,
            loss_consis,
            src_clu_loss,
            tgt_clu_loss,
            easy_idx,
            hard_idx,
        )

