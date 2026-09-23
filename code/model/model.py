from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import M2Config


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.bool()
    masked = scores.masked_fill(~mask, -1e4)
    probabilities = torch.softmax(masked, dim=dim) * mask.to(scores.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class LightweightTextEncoder(nn.Module):
    def __init__(self, config: M2Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_text_len + 1, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.text_layers, enable_nested_tensor=False)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        structural_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        input_ids = input_ids.clamp(0, self.token_embedding.num_embeddings - 1)
        token_type_ids = token_type_ids.clamp(0, self.type_embedding.num_embeddings - 1)
        position_features = self.position_embedding(positions)
        features = (
            self.token_embedding(input_ids)
            + self.type_embedding(token_type_ids)
            + position_features
        )
        null = self.null_token.expand(batch, -1, -1)
        encoded = self.encoder(
            torch.cat((features, null), dim=1),
            src_key_padding_mask=~torch.cat(
                (attention_mask.bool(), torch.ones(batch, 1, dtype=torch.bool, device=input_ids.device)),
                dim=1,
            ),
        )[:, :length]
        missing = self.missing_token + position_features
        encoded = torch.where(attention_mask[..., None].bool(), encoded, missing)
        return self.norm(encoded) * structural_mask[..., None].to(encoded.dtype)


class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, config: M2Config) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use text_backend='hf'") from exc
        self.backbone = AutoModel.from_pretrained(config.bert_model_name)
        if config.freeze_bert:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.projection = nn.Linear(self.backbone.config.hidden_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_text_len, config.hidden_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        structural_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state
        encoded = self.projection(outputs)
        positions = torch.arange(encoded.shape[1], device=encoded.device).unsqueeze(0)
        missing = self.missing_token + self.position_embedding(positions)
        encoded = torch.where(attention_mask[..., None].bool(), encoded, missing)
        return self.norm(encoded) * structural_mask[..., None].to(encoded.dtype)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, config: M2Config) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim + 1, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_av_len + 1, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.modality_layers, enable_nested_tensor=False)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        reliability: torch.Tensor,
        structural_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = values.shape
        positions = torch.arange(length, device=values.device).unsqueeze(0)
        position_features = self.position_embedding(positions)
        features = self.input_projection(
            torch.cat((values, reliability[..., None].to(values.dtype)), dim=-1)
        ) + position_features
        null = self.null_token.expand(batch, -1, -1)
        encoded = self.encoder(
            torch.cat((features, null), dim=1),
            src_key_padding_mask=~torch.cat(
                (reliability.bool(), torch.ones(batch, 1, dtype=torch.bool, device=values.device)),
                dim=1,
            ),
        )[:, :length]
        missing = self.missing_token + position_features
        encoded = torch.where(reliability[..., None].bool(), encoded, missing)
        return self.norm(encoded) * structural_mask[..., None].to(encoded.dtype)


class SafeCrossAttention(nn.Module):
    def __init__(self, config: M2Config) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.null_key = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_reliability: torch.Tensor,
        query_structural: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = query.shape[0]
        key_value = torch.cat((key_value, self.null_key.expand(batch, -1, -1)), dim=1)
        key_mask = torch.cat(
            (key_reliability.bool(), torch.ones(batch, 1, dtype=torch.bool, device=query.device)),
            dim=1,
        )
        attended, weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~key_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        output = self.norm(query + self.dropout(attended))
        output = output * query_structural[..., None].to(output.dtype)
        return output, weights[..., :-1]


class ReliabilityGate(nn.Module):
    def __init__(self, config: M2Config) -> None:
        super().__init__()
        gate_hidden = max(16, config.hidden_dim // 2)
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_dim + 2, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(gate_hidden, 1),
                )
                for _ in range(3)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        representations: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        reliability: torch.Tensor,
        missing_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = []
        for modality, representation in enumerate(representations):
            ratio = missing_ratio[:, modality, None].expand(-1, representation.shape[1])
            gate_input = torch.cat(
                (
                    representation,
                    reliability[..., modality, None].to(representation.dtype),
                    ratio[..., None].to(representation.dtype),
                ),
                dim=-1,
            )
            scores.append(self.gates[modality](gate_input))
        logits = torch.cat(scores, dim=-1)
        weights = _masked_softmax(logits, reliability, dim=-1)
        weighted = [weights[..., index, None] * value for index, value in enumerate(representations)]
        return self.norm(self.fusion(torch.cat(weighted, dim=-1))), weights


class M2Model(nn.Module):
    """Single-layer cross-attention plus mask-aware dynamic gating (M2)."""

    def __init__(self, config: M2Config) -> None:
        super().__init__()
        self.config = config
        if config.text_backend == "hf":
            self.text_encoder: nn.Module = HuggingFaceTextEncoder(config)
        else:
            self.text_encoder = LightweightTextEncoder(config)
        self.audio_encoder = ModalityEncoder(config.audio_dim, config)
        self.vision_encoder = ModalityEncoder(config.vision_dim, config)
        self.text_from_audio = SafeCrossAttention(config)
        self.text_from_vision = SafeCrossAttention(config)
        self.gate = ReliabilityGate(config)
        self.temporal_score = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.regressor = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        reliability = batch["reliability_mask"].bool()
        structural = batch["text_structural_mask"].bool()
        text = self.text_encoder(
            batch["input_ids"].long(),
            batch["text_attention_mask"].bool(),
            batch["token_type_ids"].long(),
            structural,
        )
        audio = self.audio_encoder(batch["audio"].float(), reliability[..., 1], structural)
        vision = self.vision_encoder(batch["vision"].float(), reliability[..., 2], structural)
        audio_context, audio_attention = self.text_from_audio(
            text, audio, reliability[..., 1], structural
        )
        vision_context, vision_attention = self.text_from_vision(
            text, vision, reliability[..., 2], structural
        )

        content = batch["content_mask"].bool()
        denominators = content.sum(dim=1, keepdim=True).clamp_min(1)
        observed_content = reliability & content[..., None]
        missing_ratio = 1.0 - observed_content.sum(dim=1).to(text.dtype) / denominators.to(text.dtype)
        fused, modality_weights = self.gate(
            (text, audio_context, vision_context), reliability, missing_ratio
        )
        valid_time = structural & reliability.any(dim=-1)
        fallback = structural & ~valid_time.any(dim=1, keepdim=True)
        pool_mask = valid_time | fallback
        temporal_weights = _masked_softmax(
            self.temporal_score(fused).squeeze(-1), pool_mask, dim=1
        )
        pooled = torch.sum(temporal_weights[..., None] * fused, dim=1)
        class_logits = self.classifier(pooled)
        intensity = 3.0 * torch.tanh(self.regressor(pooled).squeeze(-1))
        return {
            "class_logits": class_logits,
            "intensity": intensity,
            "fusion_feature": pooled,
            "modality_weights": modality_weights,
            "temporal_weights": temporal_weights,
            "audio_attention": audio_attention,
            "vision_attention": vision_attention,
            "missing_ratio": missing_ratio,
        }


def checkpoint_state(model: M2Model) -> tuple[dict[str, torch.Tensor], bool]:
    """Avoid duplicating a frozen public BERT inside the <=50 MB submission."""

    state = model.state_dict()
    omit_backbone = model.config.text_backend == "hf" and model.config.freeze_bert
    if omit_backbone:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("text_encoder.backbone.")
        }
    return state, omit_backbone


def load_checkpoint_state(model: M2Model, checkpoint: dict[str, Any]) -> None:
    omit_backbone = bool(checkpoint.get("omitted_frozen_text_backbone", False))
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=not omit_backbone)
    if omit_backbone:
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("text_encoder.backbone.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Invalid compact checkpoint: missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
