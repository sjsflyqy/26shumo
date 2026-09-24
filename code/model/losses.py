from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def js_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first = F.softmax(first_logits, dim=-1)
    second = F.softmax(second_logits, dim=-1)
    mixture = 0.5 * (first + second)
    first_kl = F.kl_div(torch.log(mixture.clamp_min(1e-8)), first, reduction="batchmean")
    second_kl = F.kl_div(torch.log(mixture.clamp_min(1e-8)), second, reduction="batchmean")
    return 0.5 * (first_kl + second_kl)


def soft_target_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled KL used for clean-teacher/masked-student transfer."""

    student_log_probability = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probability = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return (
        F.kl_div(student_log_probability, teacher_probability, reduction="batchmean")
        * temperature**2
    )


class M2Objective(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        classification_weight: float = 0.5,
        regression_weight: float = 1.0,
        clean_weight: float = 0.5,
        consistency_weight: float = 0.1,
        distill_temperature: float = 2.0,
        distill_logit_weight: float = 0.5,
        distill_regression_weight: float = 0.25,
        distill_feature_weight: float = 0.1,
        distill_enabled: bool = True,
        reconstruction_weight: float = 0.0,
        confidence_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.clean_weight = clean_weight
        self.consistency_weight = consistency_weight
        if distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        self.distill_temperature = distill_temperature
        self.distill_logit_weight = distill_logit_weight
        self.distill_regression_weight = distill_regression_weight
        self.distill_feature_weight = distill_feature_weight
        self.distill_enabled = distill_enabled
        if reconstruction_weight < 0 or confidence_weight < 0:
            raise ValueError("generator loss weights must be non-negative")
        self.reconstruction_weight = reconstruction_weight
        self.confidence_weight = confidence_weight

    def _task_loss(
        self,
        output: dict[str, torch.Tensor],
        class_label: torch.Tensor,
        regression_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        classification = F.cross_entropy(
            output["class_logits"], class_label, weight=self.class_weights
        )
        regression = F.smooth_l1_loss(output["intensity"], regression_label)
        total = self.classification_weight * classification + self.regression_weight * regression
        return total, classification, regression

    def forward(
        self,
        clean_output: dict[str, torch.Tensor],
        masked_output: dict[str, torch.Tensor],
        class_label: torch.Tensor,
        regression_label: torch.Tensor,
        teacher_output: dict[str, torch.Tensor] | None = None,
        *,
        clean_batch: dict[str, torch.Tensor] | None = None,
        masked_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        clean, clean_cls, clean_reg = self._task_loss(
            clean_output, class_label, regression_label
        )
        masked, masked_cls, masked_reg = self._task_loss(
            masked_output, class_label, regression_label
        )
        consistency = js_divergence(
            clean_output["class_logits"], masked_output["class_logits"]
        ) + F.smooth_l1_loss(masked_output["intensity"], clean_output["intensity"].detach())
        total = masked + self.clean_weight * clean + self.consistency_weight * consistency
        losses = {
            "loss": total,
            "clean_classification": clean_cls,
            "clean_regression": clean_reg,
            "masked_classification": masked_cls,
            "masked_regression": masked_reg,
            "consistency": consistency,
        }
        if teacher_output is not None and "latent_proposals" in masked_output:
            if clean_batch is None or masked_batch is None:
                raise ValueError("generator loss requires clean_batch and masked_batch")
            if "corruption_mask" in masked_batch:
                artificial = tuple(masked_batch["corruption_mask"][..., index] for index in range(3))
                clean_observed = tuple(clean_batch["reliability_mask"][..., index] for index in range(3))
            else:
                artificial = tuple(
                    masked_batch[f"{name}_corruption_mask"] for name in ("text", "audio", "vision")
                )
                clean_observed = (
                    clean_batch["text_attention_mask"],
                    clean_batch["audio_reliability_mask"],
                    clean_batch["vision_reliability_mask"],
                )
            reconstruction_terms = []
            confidence_terms = []
            reconstructed_positions = 0
            for index in range(3):
                selected = artificial[index].bool() & clean_observed[index].bool()
                if not bool(selected.any()):
                    continue
                reconstructed_positions += int(selected.sum())
                target = teacher_output["latent_targets"][index].detach()[selected]
                proposal = masked_output["latent_proposals"][index][selected]
                blend = masked_output["latent_reconstruction"][index][selected]
                confidence = masked_output["latent_confidence"][index][selected]
                cosine = F.cosine_similarity(proposal, target, dim=-1)
                reconstruction_terms.append(
                    F.smooth_l1_loss(proposal, target)
                    + 0.25 * (1.0 - cosine).mean()
                    + 0.5 * F.smooth_l1_loss(blend, target)
                )
                quality = ((cosine.detach() - 0.2) / 0.6).clamp(0.0, 1.0)
                confidence_terms.append(F.mse_loss(confidence, quality))
            if reconstruction_terms:
                reconstruction = torch.stack(reconstruction_terms).mean()
                confidence_loss = torch.stack(confidence_terms).mean()
            else:
                reconstruction = total.new_zeros(())
                confidence_loss = total.new_zeros(())
            total = (
                total
                + self.reconstruction_weight * reconstruction
                + self.confidence_weight * confidence_loss
            )
            losses.update(
                {
                    "loss": total,
                    "reconstruction": reconstruction,
                    "generator_confidence": confidence_loss,
                    "reconstructed_positions": total.new_tensor(float(reconstructed_positions)),
                }
            )
        if teacher_output is not None and self.distill_enabled:
            distill_logits = soft_target_distillation(
                masked_output["class_logits"],
                teacher_output["class_logits"],
                self.distill_temperature,
            )
            distill_regression = F.smooth_l1_loss(
                masked_output["intensity"], teacher_output["intensity"].detach()
            )
            student_feature = F.normalize(masked_output["fusion_feature"], dim=-1)
            teacher_feature = F.normalize(
                teacher_output["fusion_feature"].detach(), dim=-1
            )
            distill_feature = (1.0 - (student_feature * teacher_feature).sum(dim=-1)).mean()
            total = (
                total
                + self.distill_logit_weight * distill_logits
                + self.distill_regression_weight * distill_regression
                + self.distill_feature_weight * distill_feature
            )
            losses.update(
                {
                    "loss": total,
                    "distill_logits": distill_logits,
                    "distill_regression": distill_regression,
                    "distill_feature": distill_feature,
                }
            )
        return losses
