# -*- encoding: utf-8 -*-
"""RW-HEDN with one isolated Stage-E objective term: normalized JS relevance.

The coefficient-zero path delegates directly to :class:`RWHEDN`, preserving
the validated CE-only execution path.  A positive coefficient changes only
the per-source relevance score used by the existing hard/easy weighting.
"""

import torch
import torch.nn.functional as F

from models.RWHEDN import RWHEDN


class RWHEDNJS(RWHEDN):
    """Add normalized Jensen-Shannon source/target relevance to RW-HEDN L1."""

    def __init__(self, js_relevance_weight: float = 0.0, js_eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.js_relevance_weight = float(js_relevance_weight)
        self.js_eps = float(js_eps)
        if self.js_relevance_weight < 0.0:
            raise ValueError("js_relevance_weight must be non-negative")
        if self.js_eps <= 0.0:
            raise ValueError("js_eps must be positive")

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # Delegate to standard forward when JS relevance is disabled or soft SRA is not used.
        if self.js_relevance_weight == 0.0 or not self.soft_sra:
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
            tgt_mean_probability = F.softmax(self.hard_classifier(tgt_feat), dim=1).mean(dim=0)
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
                + (1 - self.rel_momentum) * easy_w
            ).detach()

        loss_cls = sum(
            hard_w[i] * self.cls_loss(self.hard_classifier(src_feats[i]), src_lab_idx[i])
            for i in range(num_sources)
        )
        source_only = bool(self.source_only_warmup_active)
        loss_adv = (
            torch.zeros((), device=tgt_feat.device)
            if source_only
            else sum(
                hard_w[i] * self.hard_advcriterion(src_feats[i], tgt_feat)
                for i in range(num_sources)
            )
        )

        src_clu_loss = sum(
            easy_w[i]
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
                feat_i = self.easy_network.extractor(src_feats[i].detach())
                self.easy_network.update_source_cluster_centers(feat_i, src_clusters[i], i)
            if not source_only:
                tgt_ext = self.easy_network.extractor(tgt_feat.detach())
                self.easy_network.update_target_cluster_centers(tgt_ext, tgt_cluster)

        if source_only:
            loss_consis = torch.zeros((), device=tgt_feat.device)
        else:
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

        easy_idx = torch.argmax(easy_w)
        hard_idx = torch.argmax(hard_w)
        return loss_cls, loss_adv, loss_consis, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx

    @staticmethod
    def _zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (values - values.mean()) / (values.std(unbiased=False) + eps)

    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(self.js_eps)
        q = q.clamp_min(self.js_eps)
        p = p / p.sum()
        q = q / q.sum()
        midpoint = 0.5 * (p + q)
        return 0.5 * (
            torch.sum(p * torch.log(p / midpoint))
            + torch.sum(q * torch.log(q / midpoint))
        )
