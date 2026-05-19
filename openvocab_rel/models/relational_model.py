from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import TrainConfig
from ..fp8_utils import fp8_autocast
from ..geometry import geom_feats_torch


class ProgressiveCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, context, context, need_weights=False)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query


class ProgressiveEdgeConditionedLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.sub_attn = ProgressiveCrossAttentionBlock(dim, num_heads=num_heads)
        self.obj_attn = ProgressiveCrossAttentionBlock(dim, num_heads=num_heads)
        self.gate_proj = nn.Linear(dim, dim * 2)
        self.rel_update = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.rel_norm = nn.LayerNorm(dim)

    def forward(
        self,
        sub_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        rel_feat: torch.Tensor,
        visual_tokens: torch.Tensor,
        geom_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate_proj(rel_feat))
        gate_sub, gate_obj = torch.chunk(gates, chunks=2, dim=-1)

        sub_query = (sub_feat + gate_sub * rel_feat).unsqueeze(0)
        obj_query = (obj_feat + gate_obj * rel_feat).unsqueeze(0)
        context = visual_tokens.unsqueeze(0)

        sub_upd = self.sub_attn(sub_query, context).squeeze(0)
        obj_upd = self.obj_attn(obj_query, context).squeeze(0)
        rel_delta = self.rel_update(torch.cat([sub_upd, obj_upd, rel_feat, geom_feat], dim=-1))
        rel_upd = self.rel_norm(rel_feat + rel_delta)
        gate_strength = 0.5 * (gate_sub.mean(dim=-1) + gate_obj.mean(dim=-1))
        return sub_upd, obj_upd, rel_upd, gate_strength


class DeformableObjectRouter(nn.Module):
    def __init__(self, dim: int, num_points: int = 8, offset_scale: float = 0.35):
        super().__init__()
        self.dim = int(dim)
        self.num_points = int(max(1, num_points))
        self.offset_scale = float(max(0.0, offset_scale))
        self.offset_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_points * 2),
        )
        self.weight_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_points),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        box_norm: torch.Tensor,
        visual_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(query.shape[0]) == 0:
            empty_w = torch.zeros((0, self.num_points), device=query.device, dtype=query.dtype)
            empty_p = torch.zeros((0, self.num_points, 2), device=query.device, dtype=query.dtype)
            return query, empty_w, empty_p

        centers = 0.5 * (box_norm[:, :2] + box_norm[:, 2:4])
        sizes = (box_norm[:, 2:4] - box_norm[:, :2]).clamp_min(1.0e-3)
        offsets = torch.tanh(self.offset_mlp(query)).view(-1, self.num_points, 2)
        points = centers.unsqueeze(1) + offsets * sizes.unsqueeze(1) * self.offset_scale
        points = torch.clamp(points, 0.0, 1.0)

        weights = torch.softmax(self.weight_mlp(query).float(), dim=-1).to(dtype=query.dtype)
        grid = points.mul(2.0).sub(1.0).view(1, int(query.shape[0]), self.num_points, 2)
        samples = F.grid_sample(
            visual_grid.float().unsqueeze(0),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        samples = samples.to(dtype=query.dtype).squeeze(0).permute(1, 2, 0).contiguous()
        routed = torch.sum(samples * weights.unsqueeze(-1), dim=1)
        return self.out_proj(routed), weights, points


class DynamicBilinearLayer(nn.Module):
    def __init__(self, dim: int, low_rank: bool = False, rank: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.dim = int(dim)
        self.low_rank = bool(low_rank)
        self.rank = int(max(4, rank))
        self.residual_scale = float(max(0.0, residual_scale))

        if self.low_rank:
            self.u_proj = nn.Linear(self.dim, self.dim * self.rank)
            self.v_proj = nn.Linear(self.dim, self.dim * self.rank)
        else:
            self.base_weight = nn.Parameter(torch.empty(self.dim, self.dim))
            nn.init.xavier_uniform_(self.base_weight)
            self.row_scale = nn.Linear(self.dim, self.dim)
            self.col_scale = nn.Linear(self.dim, self.dim)

        self.dynamic_bias = nn.Linear(self.dim, self.dim)
        self.rel_update = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.rel_norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        sub_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        rel_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.low_rank:
            n_pairs = int(rel_feat.shape[0])
            u = self.u_proj(rel_feat).view(n_pairs, self.dim, self.rank)
            v = self.v_proj(rel_feat).view(n_pairs, self.dim, self.rank)
            mid = torch.einsum("ndr,nd->nr", v, obj_feat)
            transformed_obj = torch.einsum("ndr,nr->nd", u, mid)
        else:
            row = 1.0 + 0.5 * torch.tanh(self.row_scale(rel_feat))
            col = 1.0 + 0.5 * torch.tanh(self.col_scale(rel_feat))
            obj_scaled = obj_feat * col
            transformed_obj = torch.matmul(obj_scaled, self.base_weight.t()) * row

        transformed_obj = transformed_obj + self.dynamic_bias(rel_feat)
        bilinear_vec = sub_feat * transformed_obj
        bilinear_score = torch.sum(sub_feat * transformed_obj, dim=-1)
        rel_delta = self.rel_update(torch.cat([bilinear_vec, rel_feat, sub_feat - obj_feat], dim=-1))
        rel_upd = self.rel_norm(rel_feat + (self.residual_scale * rel_delta))
        return rel_upd, bilinear_score


class ProgressiveRelationalDecoder(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        clip_vision_dim: int,
        dim: int,
        img_res: int,
        node_layers: int,
        edge_layers: int,
        bilinear_layers: int,
        bilinear_low_rank: bool,
        bilinear_rank: int,
        fp8_enabled: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.img_res = int(img_res)
        self.fp8_enabled = bool(fp8_enabled)
        self.capture_diagnostics = bool(getattr(cfg, "model_diagnostics_enabled", False))
        self.deformable_routing_enabled = bool(getattr(cfg, "deformable_routing_enabled", True))
        self.last_routing_attention: List[torch.Tensor] = []
        self.last_deformable_points: List[torch.Tensor] = []
        self.last_vector_gate: Optional[torch.Tensor] = None

        self.visual_proj = nn.Linear(int(clip_vision_dim), self.dim)
        self.box_query = nn.Sequential(
            nn.Linear(4, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.global_proj = nn.Linear(self.dim, self.dim)
        self.routing_attn = nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        self.deformable_router = DeformableObjectRouter(
            self.dim,
            num_points=int(getattr(cfg, "deformable_num_points", 8)),
            offset_scale=float(getattr(cfg, "deformable_offset_scale", 0.35)),
        )
        self.routing_gate = nn.Linear(self.dim * 2, self.dim)
        self.node_norm = nn.LayerNorm(self.dim)
        self.node_layers = nn.ModuleList(
            [ProgressiveCrossAttentionBlock(self.dim, num_heads=8) for _ in range(max(0, int(node_layers)))]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, 
            nhead=8, 
            dim_feedforward=self.dim * 2, 
            batch_first=True, 
            norm_first=True
        )
        self.context_mixer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.context_alpha = nn.Parameter(torch.tensor([0.05])) 
        
        self.use_geom_bias = getattr(cfg, "use_geom_bias", True)
        self.vector_fusion_gate = bool(getattr(cfg, "vector_fusion_gate", True))
        self.fusion_gate_temperature = float(max(0.2, getattr(cfg, "fusion_gate_temperature", 0.7)))
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        if self.use_geom_bias:
            f_dim = int(getattr(cfg, "geom_fourier_dim", 256))
            self.geom_B = nn.Parameter(torch.randn(8, f_dim) * 10.0, requires_grad=False)
            self.geom_mlp = nn.Sequential(
                nn.Linear(f_dim * 2, f_dim),
                nn.GELU(),
                nn.LayerNorm(f_dim),
                nn.Linear(f_dim, self.dim),
                nn.LayerNorm(self.dim)
            )
            self.geom_alpha = nn.Parameter(torch.tensor([0.1]))
        else:
            self.geom_proj = nn.Linear(12, self.dim)
        
        self.rel_seed = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        
        self.edge_layers = nn.ModuleList(
            [ProgressiveEdgeConditionedLayer(self.dim, num_heads=8) for _ in range(max(0, int(edge_layers)))]
        )
        self.bilinear_layers = nn.ModuleList(
            [
                DynamicBilinearLayer(
                    self.dim,
                    low_rank=bool(bilinear_low_rank),
                    rank=int(bilinear_rank),
                    residual_scale=float(getattr(cfg, "bilinear_residual_scale", 0.2)),
                )
                for _ in range(max(0, int(bilinear_layers)))
            ]
        )
        context_layers = int(max(0, getattr(cfg, "relation_context_layers", 0)))
        context_heads = int(max(1, getattr(cfg, "relation_context_heads", 8)))
        if context_layers > 0:
            rel_context_layer = nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=context_heads,
                dim_feedforward=self.dim * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.relation_context = nn.TransformerEncoder(rel_context_layer, num_layers=context_layers)
        else:
            self.relation_context = None
        self.relation_context_alpha = nn.Parameter(torch.tensor([0.1]))
        self.out_norm = nn.LayerNorm(self.dim)

    def ground_nodes(
        self,
        feat_map: torch.Tensor,
        boxes: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        device = feat_map.device
        batch_size, ch, h, w = feat_map.shape

        visual_tokens = feat_map.permute(0, 2, 3, 1).reshape(batch_size, h * w, ch)
        visual_tokens = visual_tokens.to(device, non_blocking=True)
        visual_tokens = self.visual_proj(visual_tokens)
        visual_grid = visual_tokens.view(batch_size, h, w, self.dim).permute(0, 3, 1, 2).contiguous()

        object_tokens: List[torch.Tensor] = []
        if self.capture_diagnostics:
            self.last_routing_attention = []
            self.last_deformable_points = []

        for bi in range(batch_size):
            box_i = boxes[bi]

            if box_i is not None:
                box_i = box_i.to(device, non_blocking=True)

            if int(box_i.shape[0]) == 0:
                object_tokens.append(
                    torch.zeros((0, self.dim), device=device, dtype=visual_tokens.dtype)
                )
                continue

            box_norm = torch.clamp(
                box_i.to(visual_tokens.device) / float(self.img_res), 0.0, 1.0
            ).to(device=device, dtype=visual_tokens.dtype)

            context = visual_tokens[bi].unsqueeze(0)

            global_context = self.global_proj(context.mean(dim=1, keepdim=True))
            global_bias = global_context.squeeze(0)

            query = self.box_query(box_norm) + global_bias.expand(
                int(box_norm.shape[0]), -1
            )

            query_b = query.unsqueeze(0)

            for layer in self.node_layers:
                query_b = layer(query_b, context)

            if self.deformable_routing_enabled:
                routed_local, sample_weights, sample_points = self.deformable_router(
                    query_b.squeeze(0), box_norm, visual_grid[bi]
                )
                routed = routed_local.unsqueeze(0)
                pooled = routed
                if self.capture_diagnostics:
                    self.last_routing_attention.append(sample_weights.detach().float().unsqueeze(0))
                    self.last_deformable_points.append(sample_points.detach().float())
            else:
                routed, attn_weights = self.routing_attn(
                    query_b, context, context, need_weights=True
                )
                if self.capture_diagnostics:
                    self.last_routing_attention.append(attn_weights.detach().float())

                pooled = torch.bmm(attn_weights, context)

            gate = torch.sigmoid(
                self.routing_gate(torch.cat([query_b, routed], dim=-1))
            )

            fused = query_b + gate * routed + (1.0 - gate) * pooled
            fused = fused + global_bias.unsqueeze(0).expand_as(fused)

            object_tokens.append(self.node_norm(fused.squeeze(0)))

        return visual_tokens, object_tokens

    def forward_pairs(
        self,
        visual_tokens,
        sub_feat,
        obj_feat,
        geom_feat_raw,
    ):
        self.last_gate_reg = None
        self.last_gate_val = None
        if self.capture_diagnostics:
            self.last_vector_gate = None

        if getattr(self, "use_geom_bias", True):
            geom_feat_raw_clamp = geom_feat_raw[..., :8]
            geom_feat_raw_clamp = torch.nan_to_num(geom_feat_raw_clamp, nan=0.0, posinf=1.0, neginf=-1.0)
            geom_feat_raw_clamp = torch.clamp(geom_feat_raw_clamp, -10.0, 10.0)
            
            proj = (2.0 * math.pi * geom_feat_raw_clamp) @ self.geom_B
            fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            geom_feat_proj = self.geom_mlp(fourier)
            
            sub_norm = F.normalize(sub_feat, dim=-1)
            obj_norm = F.normalize(obj_feat, dim=-1)
            sem_feat = F.normalize(sub_norm + obj_norm, dim=-1, eps=1e-6)
            geom_norm = F.normalize(geom_feat_proj, dim=-1, eps=1e-6)
            if self.vector_fusion_gate:
                concat_feats = torch.cat([sub_norm, obj_norm, geom_norm], dim=-1)
                gate_logits = self.fusion_gate(concat_feats)
                gate = torch.sigmoid(gate_logits / self.fusion_gate_temperature)
                fused_feat = gate * sem_feat + (1.0 - gate) * geom_norm
                mean_gate = gate.mean(dim=-1)
                self.last_gate_reg = ((mean_gate - 0.5) ** 2).mean()
                self.last_gate_val = gate.mean().detach()
                if self.capture_diagnostics:
                    self.last_vector_gate = gate.detach().float()
            else:
                fused_feat = sem_feat + self.geom_alpha * geom_feat_proj
                self.last_gate_val = self.geom_alpha.detach().mean()
                self.last_gate_reg = torch.tensor(0.0, device=geom_feat_proj.device)
            
            rel_feat = self.rel_seed(fused_feat)
            edge_gate = torch.zeros((rel_feat.shape[0],), device=rel_feat.device, dtype=rel_feat.dtype)
        else:
            if int(geom_feat_raw.shape[-1]) < 12:
                geom_feat_raw = F.pad(geom_feat_raw, (0, 12 - int(geom_feat_raw.shape[-1])), value=0.0)
            elif int(geom_feat_raw.shape[-1]) > 12:
                geom_feat_raw = geom_feat_raw[..., :12]

            geom_feat_raw = torch.nan_to_num(geom_feat_raw, nan=0.0, posinf=1.0, neginf=-1.0)
            geom_feat_raw = torch.clamp(geom_feat_raw, -10.0, 10.0)
            geom_feat_raw = torch.tanh(geom_feat_raw)
            geom_feat_raw = F.normalize(geom_feat_raw, dim=-1, eps=1e-6)

            geom_feat_proj = self.geom_proj(geom_feat_raw)
            sub_norm = F.normalize(sub_feat, dim=-1)
            obj_norm = F.normalize(obj_feat, dim=-1)
            geom_norm = F.normalize(geom_feat_proj, dim=-1)

            concat_feats = torch.cat([sub_norm, obj_norm, geom_norm], dim=-1)
            gate_logits = self.fusion_gate(concat_feats)
            gate = torch.sigmoid(gate_logits / self.fusion_gate_temperature)

            self.last_gate_reg = ((gate - 0.5) ** 2).mean()
            self.last_gate_val = gate.mean().detach()

            fused_feat = gate * (sub_norm + obj_norm) + (1.0 - gate) * geom_norm
            rel_feat = self.rel_seed(fused_feat)
            edge_gate = torch.zeros((rel_feat.shape[0],), device=rel_feat.device, dtype=rel_feat.dtype)

        for edge_layer in self.edge_layers:
            sub_feat, obj_feat, rel_feat, gate_strength = edge_layer(
                sub_feat, obj_feat, rel_feat, visual_tokens, geom_feat_proj
            )
            edge_gate = edge_gate + gate_strength

        if len(self.edge_layers) > 0:
            edge_gate = edge_gate / float(len(self.edge_layers))

        for bilin_layer in self.bilinear_layers:
            rel_feat, _ = bilin_layer(sub_feat, obj_feat, rel_feat)

        rel_feat = self.out_norm(rel_feat)
        return rel_feat, edge_gate


class RelationalModel(nn.Module):
    def __init__(self, cfg: TrainConfig, clip_vision_dim: int, text_dim: Optional[int] = None):
        super().__init__()
        self.dim = int(cfg.emb_dim)
        self.text_dim = int(text_dim) if text_dim is not None else None
        self.default_prune_k = int(getattr(cfg, "learned_prune_k", 0))
        self.use_checkpoint = bool(getattr(cfg, "gradient_checkpointing", False))
        self.cfg = cfg
        self.predicate_classifier_enabled = bool(getattr(cfg, "predicate_classifier_enabled", True))
        self.predicate_classifier_classes = int(getattr(cfg, "predicate_classifier_classes", 51))
        self.object_language_anchor_enabled = bool(getattr(cfg, "object_language_anchor_enabled", False))
        self.object_language_anchor_source = str(getattr(cfg, "object_language_anchor_source", "auto"))
        self.object_language_anchor_alpha = float(max(0.0, getattr(cfg, "object_language_anchor_alpha", 0.35)))
        self.object_semantic_proj = (
            nn.Sequential(
                nn.LayerNorm(int(text_dim)),
                nn.Linear(int(text_dim), self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
            if text_dim is not None and int(text_dim) > 0
            else None
        )

        self.decoder = ProgressiveRelationalDecoder(
            cfg=cfg,
            clip_vision_dim=int(clip_vision_dim),
            dim=self.dim,
            img_res=int(cfg.clip_input_res),
            node_layers=int(getattr(cfg, "progressive_node_layers", 2)),
            edge_layers=int(getattr(cfg, "progressive_edge_layers", 2)),
            bilinear_layers=int(getattr(cfg, "progressive_bilinear_layers", 2)),
            bilinear_low_rank=bool(getattr(cfg, "bilinear_low_rank", False)),
            bilinear_rank=int(getattr(cfg, "bilinear_rank", 32)),
            fp8_enabled=bool(getattr(cfg, "fp8_enabled", False)),
        )
        self.predicate_classifier = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim, self.predicate_classifier_classes),
        )
        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_impl(
        self,
        feat_map: torch.Tensor,
        obj_boxes: List[torch.Tensor],
        pairs: List[List[Tuple[int, int]]],
        obj_semantic_feats: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        device = feat_map.device

        if obj_boxes is not None:
            obj_boxes = [
                b.to(device, non_blocking=True) if b is not None else b
                for b in obj_boxes
            ]

        visual_tokens, raw_obj_feats = self.decoder.ground_nodes(feat_map, obj_boxes)

        if (
            self.object_language_anchor_enabled
            and self.object_semantic_proj is not None
            and obj_semantic_feats is not None
        ):
            anchored_feats = []
            for bi, feats in enumerate(raw_obj_feats):
                sem = obj_semantic_feats[bi] if bi < len(obj_semantic_feats) else None
                if sem is not None and int(feats.shape[0]) > 0 and int(sem.shape[0]) == int(feats.shape[0]):
                    sem = sem.to(device=device, dtype=feats.dtype, non_blocking=True)
                    sem_feat = self.object_semantic_proj(sem)
                    feats = feats + (self.object_language_anchor_alpha * sem_feat)
                anchored_feats.append(feats)
            raw_obj_feats = anchored_feats

        obj_feats = []
        for feats in raw_obj_feats:
            if feats.shape[0] > 0:
                mixed_feats = self.decoder.context_mixer(feats.unsqueeze(0)).squeeze(0)
                obj_feats.append(feats + self.decoder.context_alpha * mixed_feats)
            else:
                obj_feats.append(feats)

        regs = obj_feats
        rels: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []

        for bi in range(int(feat_map.shape[0])):
            pair_list = pairs[bi]

            if len(pair_list) == 0:
                rels.append(
                    torch.zeros((0, self.dim), device=device, dtype=feat_map.dtype)
                )
                gates.append(
                    torch.zeros((0,), device=device, dtype=feat_map.dtype)
                )
                continue

            s_idx = torch.tensor(
                [int(p[0]) for p in pair_list],
                dtype=torch.long,
                device=device,
            )
            o_idx = torch.tensor(
                [int(p[1]) for p in pair_list],
                dtype=torch.long,
                device=device,
            )

            sub_feat = obj_feats[bi].index_select(0, s_idx)
            obj_feat = obj_feats[bi].index_select(0, o_idx)

            norm_boxes = obj_boxes[bi] / float(self.decoder.img_res)

            geom_feat = geom_feats_torch(
                norm_boxes[s_idx], norm_boxes[o_idx]
            ).to(device=device, dtype=sub_feat.dtype)

            if self.use_checkpoint and self.training:
                rel_i, gate_i = checkpoint(
                    self.decoder.forward_pairs,
                    visual_tokens[bi],
                    sub_feat,
                    obj_feat,
                    geom_feat,
                    use_reentrant=False,
                )
            else:
                rel_i, gate_i = self.decoder.forward_pairs(
                    visual_tokens=visual_tokens[bi],
                    sub_feat=sub_feat,
                    obj_feat=obj_feat,
                    geom_feat_raw=geom_feat,
                )

            if self.decoder.relation_context is not None and int(rel_i.shape[0]) > 1:
                context_rel = self.decoder.relation_context(rel_i.unsqueeze(0)).squeeze(0)
                rel_i = self.decoder.out_norm(rel_i + self.decoder.relation_context_alpha * context_rel)

            rels.append(rel_i)
            gates.append(gate_i)

        return regs, rels, gates

    def forward_from_featmap(
        self,
        feat_map: torch.Tensor,
        obj_boxes: Optional[List[torch.Tensor]] = None,
        pairs: Optional[List[List[Tuple[int, int]]]] = None,
        obj_boxes_224: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        """Legacy compatibility wrapper used by train/evals scripts."""
        
        device = feat_map.device
        boxes = obj_boxes_224 if obj_boxes_224 is not None else obj_boxes
        
        if boxes is None:
            raise TypeError("forward_from_featmap requires obj_boxes or obj_boxes_224.")
        if pairs is None:
            raise TypeError("forward_from_featmap requires pairs.")

        boxes = [b.to(device, non_blocking=True) if b is not None else b for b in boxes]
    
        return_swapped = bool(kwargs.get("return_swapped", True))
        prune_override = kwargs.get("learned_prune_k_override", None)
        if prune_override is None:
            prune_k = int(self.default_prune_k)
        else:
            prune_k = int(prune_override)
        force_keep = kwargs.get("force_keep", None)
        obj_semantic_feats = kwargs.get("obj_semantic_feats", None)

        boxes = obj_boxes_224 if obj_boxes_224 is not None else obj_boxes
        if boxes is None:
            raise TypeError("forward_from_featmap requires obj_boxes or obj_boxes_224.")
        if pairs is None:
            raise TypeError("forward_from_featmap requires pairs.")

        kept_idx = self._compute_kept_indices(
            obj_boxes=boxes,
            pairs=pairs,
            force_keep=force_keep,
            prune_k=prune_k,
            device=feat_map.device,
        )
        pairs_kept: List[List[Tuple[int, int]]] = []
        for bi in range(len(pairs)):
            keep_local = kept_idx[bi].tolist()
            pairs_kept.append([pairs[bi][j] for j in keep_local])

        regs, rel_feats, gates = self._forward_impl(
            feat_map=feat_map,
            obj_boxes=boxes,
            pairs=pairs_kept,
            obj_semantic_feats=obj_semantic_feats,
        )

        if return_swapped:
            pairs_sw: List[List[Tuple[int, int]]] = []
            for pair_list in pairs_kept:
                swapped_list = []
                for p in pair_list:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        p_mut = list(p)
                        p_mut[0], p_mut[1] = int(p[1]), int(p[0])
                        swapped_list.append(tuple(p_mut))
                    else:
                        swapped_list.append((0, 0))
                pairs_sw.append(swapped_list)
            _, rel_swaps, _ = self._forward_impl(
                feat_map=feat_map,
                obj_boxes=boxes,
                pairs=pairs_sw,
                obj_semantic_feats=obj_semantic_feats,
            )
        else:
            rel_swaps = rel_feats

        assigns = [None for _ in range(int(feat_map.shape[0]))]
        return regs, rel_feats, rel_swaps, assigns, gates, kept_idx

    def forward(
        self,
        feat_map: torch.Tensor,
        obj_boxes: List[torch.Tensor],
        pairs: List[List[Tuple[int, int]]],
    ) -> List[torch.Tensor]:
        _, rels, _ = self._forward_impl(feat_map=feat_map, obj_boxes=obj_boxes, pairs=pairs)
        return rels

    def predicate_logits(self, rel_feats: torch.Tensor) -> torch.Tensor:
        return self.predicate_classifier(rel_feats)

    @staticmethod
    def score(rel_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        if rel_feats.numel() == 0 or text_feats.numel() == 0:
            return torch.zeros((rel_feats.shape[0], text_feats.shape[0]), device=rel_feats.device)
        
        xr = F.normalize(rel_feats.float(), dim=-1)
        yt = F.normalize(text_feats.float(), dim=-1)
        return xr @ yt.t()

    def _compute_kept_indices(
        self,
        obj_boxes: List[torch.Tensor],
        pairs: List[List[Tuple[int, int]]],
        force_keep,
        prune_k: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        kept: List[torch.Tensor] = []
        for bi, pair_list in enumerate(pairs):
            n = len(pair_list)
            if n == 0:
                kept.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue

            if prune_k > 0 and n > prune_k:
                s_idx = torch.tensor([int(p[0]) for p in pair_list], dtype=torch.long, device=device)
                o_idx = torch.tensor([int(p[1]) for p in pair_list], dtype=torch.long, device=device)
                boxes = obj_boxes[bi].to(device=device, dtype=torch.float32)
                centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
                w = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
                h = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
                scale = (w[s_idx] + h[s_idx] + w[o_idx] + h[o_idx]).clamp_min(1.0)
                dist = (centers[s_idx] - centers[o_idx]).pow(2).sum(dim=1).sqrt()
                geom_score = dist / scale
                keep = torch.topk(geom_score, k=int(prune_k), largest=False).indices
            else:
                keep = torch.arange(n, dtype=torch.long, device=device)

            if isinstance(force_keep, list) and bi < len(force_keep) and force_keep[bi] is not None:
                fk = force_keep[bi].to(device=device, dtype=torch.long).view(-1)
                if fk.numel() > 0:
                    fk = fk[(fk >= 0) & (fk < n)]
                    keep = torch.unique(torch.cat([keep, fk], dim=0), sorted=True)
            kept.append(keep)
        return kept