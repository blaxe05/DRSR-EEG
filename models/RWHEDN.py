# -*- encoding: utf-8 -*-
"""
RW-HEDN: Reliability-Weighted Hard-Easy Dual Network.

An improvement over HEDN (arXiv 2511.06782) with four independently-ablatable levers:

  L1  soft_sra    : soft multi-source reliability weighting. Stock HEDN uses only 1 of 14
                    sources per step (argmin/argmax of source classification loss). RW-HEDN
                    scores ALL sources each step and weights the hard (adversarial) and easy
                    (prototype) branch losses, plus the test-time vote, by source reliability.
  L2  ensemble    : test-time fusion of the hard classifier and the easy prototype voter.
                    Stock HEDN predicts the target with the easy branch only.
  L3  soft_vote /
      conf_consis : similarity-weighted soft prototype voting (instead of hard torch.mode) and
                    a confidence-filtered cross-network consistency loss (stabilises SEED-IV).
  L4  fix_proto   : proper weighted-mean cluster centres (removes the spurious `+eye` term in
                    HEDN.compute_cluster_center) and a single reconciled target momentum.

With every lever off, RW-HEDN reproduces stock HEDN exactly (it defers to the parent methods).
The forward() return signature is identical to HEDN.forward() so HEDNTrainer works unchanged.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HEDN import HEDN, EasyNetwork


class RWHEDN(HEDN):
    def __init__(self,
                 soft_sra: bool = True,
                 ensemble: bool = True,
                 soft_vote: bool = True,
                 conf_consis: bool = True,
                 fix_proto: bool = True,
                 sra_temp: float = 1.0,
                 vote_temp: float = 0.1,
                 rel_momentum: float = 0.9,
                 ensemble_alpha: float = 0.5,
                 conf_quantile: float = 0.5,
                 **hedn_kwargs):
        super(RWHEDN, self).__init__(**hedn_kwargs)
        self.soft_sra = soft_sra
        self.ensemble = ensemble
        self.soft_vote = soft_vote
        self.conf_consis = conf_consis
        self.fix_proto = fix_proto

        self.sra_temp = sra_temp
        self.vote_temp = vote_temp
        self.rel_momentum = rel_momentum
        self.ensemble_alpha = ensemble_alpha
        self.conf_quantile = conf_quantile

        # Transient execution flag used only by the locked V2 warm-up campaign.
        # It is deliberately not a parameter/buffer, so checkpoint compatibility
        # and the warmup=0 execution path remain unchanged.
        self.source_only_warmup_active = False

        # per-source reliability (EMA of "easiness" weights); uniform init
        self.register_buffer("src_reliability", torch.ones(self.num_sources) / self.num_sources)

        if self.fix_proto:
            # give the easy network a corrected cluster-centre routine
            self.easy_network.compute_cluster_center = _fixed_compute_cluster_center.__get__(
                self.easy_network, EasyNetwork)

    # ---- factory for ablation stages -------------------------------------------------
    @staticmethod
    def stage_flags(stage: str):
        """Map an ablation-stage name to the five lever booleans."""
        stage = stage.lower()
        table = {
            "baseline": dict(soft_sra=False, ensemble=False, soft_vote=False, conf_consis=False, fix_proto=False),
            "l1":       dict(soft_sra=True,  ensemble=False, soft_vote=False, conf_consis=False, fix_proto=False),
            "l1l2":     dict(soft_sra=True,  ensemble=True,  soft_vote=False, conf_consis=False, fix_proto=False),
            "l1l2l3":   dict(soft_sra=True,  ensemble=True,  soft_vote=True,  conf_consis=True,  fix_proto=False),
            "full":     dict(soft_sra=True,  ensemble=True,  soft_vote=True,  conf_consis=True,  fix_proto=True),
        }
        if stage not in table:
            raise ValueError(f"unknown stage {stage!r}; choices={list(table)}")
        return table[stage]

    # ---- training forward ------------------------------------------------------------
    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        if not self.soft_sra:
            # behave exactly like stock HEDN (single hard/easy source)
            return super().forward(srcs, tgt, src_labels, src_clusters, tgt_cluster)

        # [B, N, F] -> [N, B, F]
        srcs = srcs.permute(1, 0, 2)
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)
        num_sources = srcs.size(0)

        tgt_feat = self.feature_extractor(tgt)                       # [B, 64]  (grad)
        src_feats = [self.feature_extractor(srcs[i]) for i in range(num_sources)]
        src_lab_idx = [torch.argmax(src_labels[i], dim=1) for i in range(num_sources)]

        # ---- reliability scoring over ALL sources (no grad) ----
        with torch.no_grad():
            cls_scores = torch.stack([
                self.cls_loss(self.hard_classifier(src_feats[i]), src_lab_idx[i])
                for i in range(num_sources)
            ])                                                       # [N] source cls loss
            hard_w = F.softmax(cls_scores / self.sra_temp, dim=0)    # emphasise HARD (high loss)
            easy_w = F.softmax(-cls_scores / self.sra_temp, dim=0)   # emphasise EASY (low loss)
            self.src_reliability = (self.rel_momentum * self.src_reliability
                                    + (1 - self.rel_momentum) * easy_w).detach()

        # ---- hard (adversarial) branch: reliability-weighted over all sources ----
        loss_cls = sum(hard_w[i] * self.cls_loss(self.hard_classifier(src_feats[i]), src_lab_idx[i])
                       for i in range(num_sources))
        source_only = bool(self.source_only_warmup_active)
        loss_adv = (
            torch.zeros((), device=tgt_feat.device)
            if source_only else
            sum(hard_w[i] * self.hard_advcriterion(src_feats[i], tgt_feat)
                for i in range(num_sources))
        )

        # ---- easy (prototype) branch: weighted SupCon on sources + target ----
        src_clu_loss = sum(easy_w[i] * self.clu_loss(self.easy_network.extractor(src_feats[i]),
                                                     src_clusters[i])
                           for i in range(num_sources))
        tgt_clu_loss = (
            torch.zeros((), device=tgt_feat.device)
            if source_only else
            self.clu_loss(self.easy_network.extractor(tgt_feat), tgt_cluster)
        )

        # ---- update ALL source cluster centres (detached) ----
        with torch.no_grad():
            for i in range(num_sources):
                feat_i = self.easy_network.extractor(src_feats[i].detach())
                self.easy_network.update_source_cluster_centers(feat_i, src_clusters[i], i)
            if not source_only:
                tgt_ext = self.easy_network.extractor(tgt_feat.detach())
                self.easy_network.update_target_cluster_centers(tgt_ext, tgt_cluster)

        # ---- cross-network consistency (confidence filtered) ----
        if source_only:
            loss_consis = torch.zeros((), device=tgt_feat.device)
        else:
            with torch.no_grad():
                proto_dist = self._easy_soft_dist(tgt.detach())     # [B, C] on CPU
                proto_dist = proto_dist / (proto_dist.sum(dim=1, keepdim=True) + 1e-8)
                tgt_conf, tgt_proto_pred = proto_dist.max(dim=1)
            tgt_logit = self.hard_classifier(tgt_feat)
            tgt_proto_pred = tgt_proto_pred.to(tgt_logit.device)
            if self.conf_consis:
                thr = torch.quantile(tgt_conf, self.conf_quantile).item()
                mask = (tgt_conf >= thr).to(tgt_logit.device)
                if mask.any():
                    loss_consis = F.cross_entropy(tgt_logit[mask], tgt_proto_pred[mask])
                else:
                    loss_consis = torch.zeros((), device=tgt_logit.device)
            else:
                loss_consis = self.consis_loss(tgt_logit, tgt_proto_pred)

        easy_idx = torch.argmax(easy_w)
        hard_idx = torch.argmax(hard_w)
        return loss_cls, loss_adv, loss_consis, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx

    # ---- test-time prediction --------------------------------------------------------
    def predict(self, data, mode="target"):
        return torch.argmax(self.predict_proba(data, mode=mode), dim=1)

    @torch.no_grad()
    def predict_proba(self, data, mode="target"):
        self.eval()
        if mode == "source":
            feat = self.feature_extractor(data)
            return F.softmax(self.hard_classifier(feat), dim=1).cpu()
        if not (self.soft_vote or self.soft_sra or self.ensemble):
            feat = self.feature_extractor(data)
            return self.easy_network.predict_proba(feat).cpu()
        easy_dist = self._easy_soft_dist(data)
        easy_dist = easy_dist / (easy_dist.sum(dim=1, keepdim=True) + 1e-8)
        if self.ensemble:
            feat = self.feature_extractor(data)
            hard_dist = F.softmax(self.hard_classifier(feat), dim=1).cpu()
            easy_dist = self.ensemble_alpha * easy_dist + (1 - self.ensemble_alpha) * hard_dist
        return easy_dist / (easy_dist.sum(dim=1, keepdim=True) + 1e-8)

    def predict_by_easy(self, data):
        return torch.argmax(self.predict_proba(data, mode="target"), dim=1)

    @torch.no_grad()
    def _easy_soft_dist(self, data):
        """Reliability-weighted, similarity-soft prototype class distribution. Returns [N, C] on CPU.

        Two-hop like HEDN.predict: sample -> target-cluster (soft) -> source-cluster (soft) -> label,
        aggregated over sources by reliability. With soft_vote off, the two hops use argmax (hard),
        recovering HEDN's routing but adding reliability weights and (optionally) an ensemble.
        """
        feat = self.feature_extractor(data)
        return self._easy_soft_dist_from_feature(feat, output_device=torch.device("cpu"))

    @torch.no_grad()
    def _easy_soft_dist_from_feature(self, feature, output_device=None):
        """Prototype class distribution from shared features.

        This keeps prototype voting usable during forward() without re-running f(.), and moves
        prototype banks to the feature device for CPU/GPU-safe inference.
        """
        device = feature.device
        per_source = self._source_specific_soft_dist_from_feature(feature)
        rel = self.src_reliability.detach().to(device)
        rel = rel / (rel.sum() + 1e-8)
        if not self.soft_sra:
            rel = torch.ones_like(rel) / self.num_sources
        dist = (per_source * rel[:, None, None]).sum(dim=0)
        if output_device is not None:
            dist = dist.to(output_device)
        return dist

    @torch.no_grad()
    def source_specific_predict_proba(self, data):
        self.eval()
        feature = self.feature_extractor(data)
        return self._source_specific_soft_dist_from_feature(feature).cpu()

    @torch.no_grad()
    def _source_specific_soft_dist_from_feature(self, feature):
        """Return unweighted prototype-routing distributions as [source, sample, class]."""
        en = self.easy_network
        device = feature.device
        feat = F.normalize(en.extractor(feature).detach(), p=2, dim=1)
        tgt_centers = F.normalize(en.tgt_cluster_centers.to(device), p=2, dim=1)
        sim_tgt = feat @ tgt_centers.T
        if self.soft_vote:
            a = F.softmax(sim_tgt / self.vote_temp, dim=1)
        else:
            a = F.one_hot(torch.argmax(sim_tgt, dim=1), tgt_centers.size(0)).float()
        parts = []
        for i in range(self.num_sources):
            src_centers = F.normalize(en.src_cluster_centers[i].to(device), p=2, dim=1)  # [Ksrc, D]
            sim_src = tgt_centers @ src_centers.T                   # [Ktgt, Ksrc]
            if self.soft_vote:
                b = F.softmax(sim_src / self.vote_temp, dim=1)
            else:
                b = F.one_hot(torch.argmax(sim_src, dim=1), src_centers.size(0)).float()
            b = b.to(device)
            labels = en.src_cluster_labels[i].to(device=device, dtype=torch.long)
            lbl_onehot = F.one_hot(labels, self.num_classes).float() # [Ksrc, C]
            cluster_class = b @ lbl_onehot                          # [Ktgt, C]
            source_dist = a @ cluster_class
            parts.append(source_dist / (source_dist.sum(dim=1, keepdim=True) + 1e-8))
        return torch.stack(parts, dim=0)

    # ---- state incl. reliability buffer ---------------------------------------------
    def get_state(self):
        state = super().get_state()
        state["proto"]["src_reliability"] = self.src_reliability.clone().detach()
        return state

    def load_state(self, state):
        super().load_state(state)
        if "src_reliability" in state.get("proto", {}):
            self.src_reliability.copy_(state["proto"]["src_reliability"].to(self.src_reliability.device))


@torch.no_grad()
def _fixed_compute_cluster_center(self, features, clusters, num_clusters):
    """Proper weighted mean per cluster (L4). Replaces HEDN's `inverse(diag)+eye` quirk."""
    one_hot = F.one_hot(clusters.to(torch.long), num_classes=num_clusters).float()
    counts = one_hot.sum(dim=0) + 1e-6                              # [K]
    sums = torch.matmul(one_hot.T, features.cpu())                 # [K, D]
    centers = sums / counts.unsqueeze(1)
    return centers
