from __future__ import annotations

import copy
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
        return_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
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
        encoded = torch.cat((features, null), dim=1)
        padding_mask = ~torch.cat(
            (attention_mask.bool(), torch.ones(batch, 1, dtype=torch.bool, device=input_ids.device)),
            dim=1,
        )
        if return_hidden_states:
            hidden_states = []
            for layer in self.encoder.layers:
                encoded = layer(encoded, src_key_padding_mask=padding_mask)
                hidden_states.append(encoded[:, :length])
            if self.encoder.norm is not None:
                hidden_states[-1] = self.encoder.norm(encoded)[:, :length]
        else:
            encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)[:, :length]
        missing = self.missing_token + position_features

        def finalize(value: torch.Tensor) -> torch.Tensor:
            value = torch.where(attention_mask[..., None].bool(), value, missing)
            return self.norm(value) * structural_mask[..., None].to(value.dtype)

        if return_hidden_states:
            return [finalize(value) for value in hidden_states]
        return finalize(encoded)


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
        self.fusion_levels = config.fusion_levels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        structural_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),
            token_type_ids=token_type_ids,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        raw_states = (
            list(outputs.hidden_states[1:])[-self.fusion_levels :]
            if return_hidden_states
            else [outputs.last_hidden_state]
        )
        positions = torch.arange(raw_states[-1].shape[1], device=raw_states[-1].device).unsqueeze(0)
        missing = self.missing_token + self.position_embedding(positions)

        def finalize(value: torch.Tensor) -> torch.Tensor:
            value = self.projection(value)
            value = torch.where(attention_mask[..., None].bool(), value, missing)
            return self.norm(value) * structural_mask[..., None].to(value.dtype)

        encoded_states = [finalize(value) for value in raw_states]
        return encoded_states if return_hidden_states else encoded_states[0]


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
        return_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        batch, length, _ = values.shape
        positions = torch.arange(length, device=values.device).unsqueeze(0)
        position_features = self.position_embedding(positions)
        features = self.input_projection(
            torch.cat((values, reliability[..., None].to(values.dtype)), dim=-1)
        ) + position_features
        null = self.null_token.expand(batch, -1, -1)
        encoded = torch.cat((features, null), dim=1)
        padding_mask = ~torch.cat(
            (reliability.bool(), torch.ones(batch, 1, dtype=torch.bool, device=values.device)),
            dim=1,
        )
        if return_hidden_states:
            hidden_states = []
            for layer in self.encoder.layers:
                encoded = layer(encoded, src_key_padding_mask=padding_mask)
                hidden_states.append(encoded[:, :length])
            if self.encoder.norm is not None:
                hidden_states[-1] = self.encoder.norm(encoded)[:, :length]
        else:
            encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)[:, :length]
        missing = self.missing_token + position_features

        def finalize(value: torch.Tensor) -> torch.Tensor:
            value = torch.where(reliability[..., None].bool(), value, missing)
            return self.norm(value) * structural_mask[..., None].to(value.dtype)

        if return_hidden_states:
            return [finalize(value) for value in hidden_states]
        return finalize(encoded)


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


class VertFoundMultiLevelFusion(nn.Module):
    """Temporal adaptation of VertFound's multi-level bidirectional attention.

    Each encoder level first updates audio/vision from text, then lets text read
    the updated modality. Reliability gating is applied at every level before a
    learned top-down-style aggregation across encoder depths.
    """

    def __init__(self, config: M2Config) -> None:
        super().__init__()
        self.num_levels = config.fusion_levels
        self.audio_from_text = nn.ModuleList(
            [SafeCrossAttention(config) for _ in range(self.num_levels)]
        )
        self.vision_from_text = nn.ModuleList(
            [SafeCrossAttention(config) for _ in range(self.num_levels)]
        )
        self.text_from_audio = nn.ModuleList(
            [SafeCrossAttention(config) for _ in range(self.num_levels)]
        )
        self.text_from_vision = nn.ModuleList(
            [SafeCrossAttention(config) for _ in range(self.num_levels)]
        )
        self.gates = nn.ModuleList(
            [ReliabilityGate(config) for _ in range(self.num_levels)]
        )
        self.level_score = nn.Sequential(
            nn.Linear(config.hidden_dim, max(16, config.hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, config.hidden_dim // 2), 1),
        )
        self.level_bias = nn.Parameter(torch.zeros(self.num_levels))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        text_levels: list[torch.Tensor],
        audio_levels: list[torch.Tensor],
        vision_levels: list[torch.Tensor],
        reliability: torch.Tensor,
        structural: torch.Tensor,
        missing_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_levels = text_levels[-self.num_levels :]
        audio_levels = audio_levels[-self.num_levels :]
        vision_levels = vision_levels[-self.num_levels :]
        if not (
            len(text_levels) == len(audio_levels) == len(vision_levels) == self.num_levels
        ):
            raise ValueError("Every encoder must provide fusion_levels hidden states")

        fused_levels = []
        modality_weights = []
        audio_attentions = []
        vision_attentions = []
        for index, (text, audio, vision) in enumerate(
            zip(text_levels, audio_levels, vision_levels)
        ):
            # First direction: text information updates the two non-text streams.
            enhanced_audio, _ = self.audio_from_text[index](
                audio, text, reliability[..., 0], structural
            )
            enhanced_vision, _ = self.vision_from_text[index](
                vision, text, reliability[..., 0], structural
            )
            # Reverse direction: the text timeline reads the updated streams.
            audio_context, audio_attention = self.text_from_audio[index](
                text, enhanced_audio, reliability[..., 1], structural
            )
            vision_context, vision_attention = self.text_from_vision[index](
                text, enhanced_vision, reliability[..., 2], structural
            )
            fused, weights = self.gates[index](
                (text, audio_context, vision_context), reliability, missing_ratio
            )
            fused_levels.append(fused)
            modality_weights.append(weights)
            audio_attentions.append(audio_attention)
            vision_attentions.append(vision_attention)

        stacked = torch.stack(fused_levels, dim=2)
        level_logits = self.level_score(stacked).squeeze(-1) + self.level_bias
        level_weights = torch.softmax(level_logits, dim=-1)
        fused = torch.sum(level_weights[..., None] * stacked, dim=2)
        # Deepest-level residual mirrors the top-down residual path in PAN/FPN.
        fused = self.norm(fused + fused_levels[-1])
        modality_weight = torch.sum(
            level_weights[..., None] * torch.stack(modality_weights, dim=2), dim=2
        )
        audio_attention = torch.sum(
            level_weights[..., None] * torch.stack(audio_attentions, dim=2), dim=2
        )
        vision_attention = torch.sum(
            level_weights[..., None] * torch.stack(vision_attentions, dim=2), dim=2
        )
        return fused, modality_weight, level_weights, audio_attention, vision_attention


class M2Model(nn.Module):
    """Mask-aware M2 with selectable baseline or VertFound-style fusion."""

    def __init__(self, config: M2Config) -> None:
        super().__init__()
        self.config = config
        if config.text_backend == "hf":
            self.text_encoder: nn.Module = HuggingFaceTextEncoder(config)
        else:
            self.text_encoder = LightweightTextEncoder(config)
        self.audio_encoder = ModalityEncoder(config.audio_dim, config)
        self.vision_encoder = ModalityEncoder(config.vision_dim, config)
        if config.fusion_type == "vertfound_multilevel":
            self.multilevel_fusion: VertFoundMultiLevelFusion | None = (
                VertFoundMultiLevelFusion(config)
            )
            self.text_from_audio = None
            self.text_from_vision = None
            self.gate = None
        else:
            self.multilevel_fusion = None
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
        content = batch["content_mask"].bool()
        denominators = content.sum(dim=1, keepdim=True).clamp_min(1)
        observed_content = reliability & content[..., None]
        missing_ratio = 1.0 - observed_content.sum(dim=1).float() / denominators.float()

        if self.multilevel_fusion is not None:
            text_levels = self.text_encoder(
                batch["input_ids"].long(),
                batch["text_attention_mask"].bool(),
                batch["token_type_ids"].long(),
                structural,
                return_hidden_states=True,
            )
            audio_levels = self.audio_encoder(
                batch["audio"].float(),
                reliability[..., 1],
                structural,
                return_hidden_states=True,
            )
            vision_levels = self.vision_encoder(
                batch["vision"].float(),
                reliability[..., 2],
                structural,
                return_hidden_states=True,
            )
            if not isinstance(text_levels, list):
                raise TypeError("text encoder did not return hidden states")
            if not isinstance(audio_levels, list) or not isinstance(vision_levels, list):
                raise TypeError("modality encoder did not return hidden states")
            (
                fused,
                modality_weights,
                fusion_level_weights,
                audio_attention,
                vision_attention,
            ) = self.multilevel_fusion(
                text_levels,
                audio_levels,
                vision_levels,
                reliability,
                structural,
                missing_ratio.to(text_levels[-1].dtype),
            )
        else:
            text = self.text_encoder(
                batch["input_ids"].long(),
                batch["text_attention_mask"].bool(),
                batch["token_type_ids"].long(),
                structural,
            )
            audio = self.audio_encoder(
                batch["audio"].float(), reliability[..., 1], structural
            )
            vision = self.vision_encoder(
                batch["vision"].float(), reliability[..., 2], structural
            )
            if not torch.is_tensor(text) or not torch.is_tensor(audio) or not torch.is_tensor(vision):
                raise TypeError("baseline encoders must return tensors")
            if self.text_from_audio is None or self.text_from_vision is None or self.gate is None:
                raise RuntimeError("baseline fusion modules are not initialized")
            audio_context, audio_attention = self.text_from_audio(
                text, audio, reliability[..., 1], structural
            )
            vision_context, vision_attention = self.text_from_vision(
                text, vision, reliability[..., 2], structural
            )
            fused, modality_weights = self.gate(
                (text, audio_context, vision_context),
                reliability,
                missing_ratio.to(text.dtype),
            )
            fusion_level_weights = torch.ones_like(fused[..., :1])
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
            "fusion_level_weights": fusion_level_weights,
            "temporal_weights": temporal_weights,
            "audio_attention": audio_attention,
            "vision_attention": vision_attention,
            "missing_ratio": missing_ratio,
        }


def create_ema_teacher(student: M2Model) -> M2Model:
    """Create a non-trainable teacher initialized from the student.

    A frozen Hugging Face BERT is shared because it never receives an EMA
    update. This avoids storing a second 110M-parameter BERT on the GPU.
    """

    teacher = copy.deepcopy(student)
    if student.config.text_backend == "hf" and student.config.freeze_bert:
        teacher.text_encoder.backbone = student.text_encoder.backbone
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(student: M2Model, teacher: M2Model, decay: float) -> None:
    """Update teacher parameters/buffers with an exponential moving average."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must satisfy 0 <= decay < 1")
    student_parameters = dict(student.named_parameters())
    for name, teacher_parameter in teacher.named_parameters():
        student_parameter = student_parameters[name]
        if teacher_parameter.data_ptr() == student_parameter.data_ptr():
            continue
        teacher_parameter.lerp_(student_parameter.detach(), 1.0 - decay)
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        student_buffer = student_buffers[name]
        if teacher_buffer.data_ptr() == student_buffer.data_ptr():
            continue
        if teacher_buffer.is_floating_point():
            teacher_buffer.lerp_(student_buffer.detach(), 1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


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
