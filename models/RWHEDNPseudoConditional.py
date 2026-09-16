# -*- encoding: utf-8 -*-
"""Stage E2: RW-HEDN-JS plus one isolated pseudo-class alignment term.

Coefficient zero delegates to :class:`RWHEDNJS` exactly.  A positive
coefficient adds only soft pseudo-class conditional feature-mean alignment to
the transfer loss.  Confidence, mass, support, and class-specific reliability
gates intentionally remain absent for later Stage E experiments.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.RWHEDNJS import RWHEDNJS


class RWHEDNPseudoConditional(RWHEDNJS):
    """Add ungated soft pseudo-class conditional alignment to Stage E1."""

    def __init__(
        self,
        conditional_alignment_weight: float = 0.0,
        conditional_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conditional_alignment_weight = float(conditional_alignment_weight)
        self.conditional_eps = float(conditional_eps)
        if self.conditional_alignment_weight < 0.0:
            raise ValueError("conditional_alignment_weight must be non-negative")
        if self.conditional_eps <= 0.0:
            raise ValueError("conditional_eps must be positive")
        self.register_buffer("last_conditional_alignment_loss", torch.zeros(()))
        self.register_buffer("last_target_pseudo_class_mass", torch.zeros(self.num_classes))
        self.register_buffer("last_conditional_valid_class_count", torch.zeros(()))

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # Delegate to standard forward during warmup, when conditional alignment is disabled, or without soft SRA.
        if (
            self.conditional_alignment_weight == 0.0
            or bool(self.source_only_warmup_active)
            or not self.soft_sra
        ):
            with torch.no_grad():
                self.last_conditional_alignment_loss.zero_()
                self.last_target_pseudo_class_mass.zero_()
                self.last_conditional_valid_class_count.zero_()
            return super().forward(srcs, tgt, src_labels, src_clusters, tgt_cluster)

        srcs = srcs.permute(1, 0, 2)
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)
        num_sources = srcs.size(0)

        tgt_feat = self.feature_extractor(tgt)
        src_feats = [self.feature_extractor(srcs[i]) for i in range(num_sources)]
        src_lab_idx = [torch.argmax(src_labels[i], dim=1) for i in range(num_sources)]

        with torch.no_grad():
            src_logits = [self.hard_classifier(src_feats[i]) for i in range(num_sources)]
            cls_scores = torch.stack([
                self.cls_loss(src_logits[i], src_lab_idx[i]) for i in range(num_sources)
            ])
            tgt_probability = F.softmax(self.hard_classifier(tgt_feat), dim=1)
            if self.js_relevance_weight == 0.0:
                # Match RWHEDN.forward for a Stage E1 zero-JS selection.
                relevance_score = cls_scores
            else:
                tgt_mean_probability = tgt_probability.mean(dim=0)
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
            hard_w = F.softmax(relevance_score / self.sra_temp, dim=0)
            easy_w = F.softmax(-relevance_score / self.sra_temp, dim=0)
            self.src_reliability = (
                self.rel_momentum * self.src_reliability
                + (1.0 - self.rel_momentum) * easy_w
            ).detach()

        loss_cls = sum(
            hard_w[i] * self.cls_loss(self.hard_classifier(src_feats[i]), src_lab_idx[i])
            for i in range(num_sources)
        )
        loss_adv = sum(
            hard_w[i] * self.hard_advcriterion(src_feats[i], tgt_feat)
            for i in range(num_sources)
        )

        conditional_parts = []
        valid_class_count = 0
        for index in range(num_sources):
            conditional_loss, valid_classes = self._soft_pseudo_class_alignment(
                src_feats[index], tgt_feat, src_lab_idx[index], tgt_probability
            )
            conditional_parts.append(hard_w[index] * conditional_loss)
            valid_class_count += valid_classes
        conditional_alignment_loss = sum(conditional_parts)
        transfer_loss = (
            loss_adv
            + self.conditional_alignment_weight * conditional_alignment_loss
        )

        src_clu_loss = sum(
            easy_w[i]
            * self.clu_loss(self.easy_network.extractor(src_feats[i]), src_clusters[i])
            for i in range(num_sources)
        )
        tgt_clu_loss = self.clu_loss(self.easy_network.extractor(tgt_feat), tgt_cluster)

        with torch.no_grad():
            for i in range(num_sources):
                feat_i = self.easy_network.extractor(src_feats[i].detach())
                self.easy_network.update_source_cluster_centers(feat_i, src_clusters[i], i)
            tgt_ext = self.easy_network.extractor(tgt_feat.detach())
            self.easy_network.update_target_cluster_centers(tgt_ext, tgt_cluster)

        with torch.no_grad():
            proto_dist = self._easy_soft_dist(tgt.detach())
            proto_dist = proto_dist / (proto_dist.sum(dim=1, keepdim=True) + 1e-8)
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
            self.last_target_pseudo_class_mass.copy_(
                tgt_probability.detach().sum(dim=0).to(
                    self.last_target_pseudo_class_mass.device
                )
            )
            self.last_conditional_valid_class_count.copy_(
                torch.tensor(
                    float(valid_class_count),
                    device=self.last_conditional_valid_class_count.device,
                )
            )

        easy_idx = torch.argmax(easy_w)
        hard_idx = torch.argmax(hard_w)
        return (
            loss_cls,
            transfer_loss,
            loss_consis,
            src_clu_loss,
            tgt_clu_loss,
            easy_idx,
            hard_idx,
        )

    def _soft_pseudo_class_alignment(
        self,
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        source_label: torch.Tensor,
        target_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Align source class means to ungated soft target pseudo-class means."""
        losses = []
        for class_index in range(self.num_classes):
            source_mask = source_label == class_index
            target_weight = target_probability[:, class_index].detach()
            target_mass = target_weight.sum()
            if not source_mask.any() or target_mass <= self.conditional_eps:
                continue
            source_mean = source_feature[source_mask].mean(dim=0)
            target_mean = (
                target_feature * target_weight.unsqueeze(1)
            ).sum(dim=0) / (target_mass + self.conditional_eps)
            losses.append(F.mse_loss(source_mean, target_mean, reduction="mean"))
        if not losses:
            return source_feature.sum() * 0.0, 0
        return torch.stack(losses).mean(), len(losses)
