from __future__ import annotations

from typing import Any

import torch

from .config import SpanMaskConfig


MODALITY_INDEX = {"T": 0, "A": 1, "V": 2}


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator).item())


def _randint(low: int, high: int, generator: torch.Generator) -> int:
    if high <= low:
        return low
    return int(torch.randint(low, high, (), generator=generator).item())


def _choose_start(length: int, span: int, location: str, generator: torch.Generator) -> int:
    maximum = max(0, length - span)
    if location == "begin":
        upper = max(1, maximum // 3 + 1)
        return _randint(0, upper, generator)
    if location == "middle":
        center = maximum // 2
        radius = max(1, length // 10)
        return _randint(max(0, center - radius), min(maximum, center + radius) + 1, generator)
    if location == "end":
        lower = max(0, maximum - max(1, maximum // 3))
        return _randint(lower, maximum + 1, generator)
    return _randint(0, maximum + 1, generator)


def _start_candidates(length: int, span: int, location: str) -> range:
    maximum = max(0, length - span)
    if location == "begin":
        return range(0, max(1, maximum // 3 + 1))
    if location == "middle":
        center = maximum // 2
        radius = max(1, length // 10)
        return range(max(0, center - radius), min(maximum, center + radius) + 1)
    if location == "end":
        return range(max(0, maximum - max(1, maximum // 3)), maximum + 1)
    return range(0, maximum + 1)


def _guided_relative_starts(
    valid_positions: torch.Tensor,
    importance: torch.Tensor,
    rate: float,
    spans: int,
    location: str,
    temperature: float,
    generator: torch.Generator,
) -> list[float]:
    positions = torch.nonzero(valid_positions, as_tuple=False).flatten()
    length = int(positions.numel())
    if length == 0:
        return []
    total = max(1, min(int(round(rate * length)), length))
    spans = max(1, min(spans, total))
    base, remainder = divmod(total, spans)
    scores = importance.float()[positions].clamp_min(0)
    occupied = torch.zeros(length, dtype=torch.bool)
    relative = []
    for span_index in range(spans):
        width = base + int(span_index < remainder)
        candidates = list(_start_candidates(length, width, location))
        available = [
            start for start in candidates if not occupied[start : start + width].any()
        ]
        if not available:
            available = [
                start for start in range(length - width + 1)
                if not occupied[start : start + width].any()
            ]
        if available:
            candidates = available
        window_scores = torch.tensor(
            [scores[start : start + width].mean().item() for start in candidates],
            dtype=torch.float32,
        )
        if float(window_scores.max() - window_scores.min()) <= 1e-8:
            selected = candidates[_randint(0, len(candidates), generator)]
        else:
            scaled = (window_scores - window_scores.min()) / (
                window_scores.max() - window_scores.min()
            )
            probabilities = torch.softmax(scaled / temperature, dim=0)
            selected = candidates[int(torch.multinomial(probabilities, 1, generator=generator))]
        occupied[selected : selected + width] = True
        relative.append(selected / max(1, length - width))
    return relative


def _span_mask(
    valid_positions: torch.Tensor,
    rate: float,
    spans: int,
    location: str,
    generator: torch.Generator,
    relative_starts: list[float] | None = None,
    importance: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    positions = torch.nonzero(valid_positions, as_tuple=False).flatten()
    output = torch.zeros_like(valid_positions, dtype=torch.bool)
    if positions.numel() == 0:
        return output
    total = max(1, min(int(round(rate * positions.numel())), int(positions.numel())))
    spans = max(1, min(spans, total))
    if importance is not None and relative_starts is None:
        relative_starts = _guided_relative_starts(
            valid_positions, importance, rate, spans, location, temperature, generator
        )
    base, remainder = divmod(total, spans)
    occupied = torch.zeros(int(positions.numel()), dtype=torch.bool)
    for span_index in range(spans):
        span_length = base + int(span_index < remainder)
        if not relative_starts:
            proposed = _choose_start(int(positions.numel()), span_length, location, generator)
        else:
            maximum = max(0, int(positions.numel()) - span_length)
            proposed = int(round(relative_starts[span_index % len(relative_starts)] * maximum))
        allowed = [
            value for value in _start_candidates(int(positions.numel()), span_length, location)
            if not occupied[value : value + span_length].any()
        ]
        if not allowed:
            allowed = [
                value for value in range(int(positions.numel()) - span_length + 1)
                if not occupied[value : value + span_length].any()
            ]
        start = min(allowed, key=lambda value: abs(value - proposed)) if allowed else proposed
        occupied[start : start + span_length] = True
        output[positions[start : start + span_length]] = True
    return output


def _relative_start(location: str, generator: torch.Generator) -> float:
    value = _rand(generator)
    if location == "begin":
        return value / 3.0
    if location == "middle":
        return 0.4 + 0.2 * value
    if location == "end":
        return 2.0 / 3.0 + value / 3.0
    return value


def teacher_emotion_importance(
    output: dict[str, Any], batch: dict[str, Any], intensity_weight: float = 0.25
) -> dict[str, torch.Tensor]:
    """Class and intensity attribution on the teacher's clean latent streams.

    This uses only the current training batch. Gradients stop at the detached
    encoder outputs, so teacher parameters and input features are not updated.
    """

    sources = output["latent_saliency_sources"]
    predicted = output["class_logits"].detach().argmax(dim=-1, keepdim=True)
    evidence = output["class_logits"].gather(1, predicted).sum()
    evidence = evidence + intensity_weight * output["intensity"].abs().sum()
    flat = [value for modality in sources for value in modality]
    gradients = torch.autograd.grad(evidence, flat, allow_unused=True)
    result = {}
    offset = 0
    for symbol, modality in zip(("T", "A", "V"), sources):
        values = gradients[offset : offset + len(modality)]
        offset += len(modality)
        saliency = torch.zeros_like(modality[-1][..., 0])
        for feature, gradient in zip(modality, values):
            if gradient is not None:
                saliency = saliency + (gradient * feature).abs().mean(dim=-1)
        if symbol == "T":
            valid = batch["content_mask"].bool() & batch["text_attention_mask"].bool()
        elif "audio_reliability_mask" in batch:
            key = "audio_reliability_mask" if symbol == "A" else "vision_reliability_mask"
            valid = batch[key].bool()
        else:
            index = 1 if symbol == "A" else 2
            valid = batch["content_mask"].bool() & batch["reliability_mask"][..., index].bool()
        saliency = saliency * valid.to(saliency.dtype)
        maximum = saliency.amax(dim=1, keepdim=True).clamp_min(1e-8)
        result[symbol] = (saliency / maximum).detach().cpu()
    return result


def corrupt_aligned_batch(
    batch: dict[str, Any],
    config: SpanMaskConfig,
    generator: torch.Generator,
    *,
    force_rate: float | None = None,
    force_modalities: str | None = None,
    force_location: str | None = None,
    force_spans: int | None = None,
    force_sync: bool | None = None,
    importance: dict[str, torch.Tensor] | None = None,
    emotion_probability: float = 0.0,
    emotion_temperature: float = 0.2,
) -> dict[str, Any]:
    """Create a local-span-corrupted student view without mutating ``batch``.

    Despite the historical name, this supports both a shared aligned timeline
    and independent text/audio/vision timelines.
    """

    output = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    if not 0.0 <= emotion_probability <= 1.0:
        raise ValueError("emotion_probability must be in [0, 1]")
    if emotion_temperature <= 0:
        raise ValueError("emotion_temperature must be positive")
    if "audio_reliability_mask" in output:
        return _corrupt_unaligned_batch(
            output,
            config,
            generator,
            force_rate=force_rate,
            force_modalities=force_modalities,
            force_location=force_location,
            force_spans=force_spans,
            force_sync=force_sync,
            importance=importance,
            emotion_probability=emotion_probability,
            emotion_temperature=emotion_temperature,
        )

    reliability = output["reliability_mask"].bool()
    corruption = torch.zeros_like(reliability)
    batch_size = reliability.shape[0]

    for sample_index in range(batch_size):
        if force_rate is None and _rand(generator) < config.clean_probability:
            continue
        combination = force_modalities
        if combination is None:
            combination = config.modality_combinations[
                _randint(0, len(config.modality_combinations), generator)
            ]
        rate = force_rate
        if rate is None:
            rate = config.min_rate + _rand(generator) * (config.max_rate - config.min_rate)
        location = force_location
        if location is None:
            location = config.locations[_randint(0, len(config.locations), generator)]
        number_of_spans = force_spans
        if number_of_spans is None:
            number_of_spans = _randint(config.min_spans, config.max_spans + 1, generator)
        sync = (
            force_sync
            if force_sync is not None
            else len(combination) > 1 and _rand(generator) < config.sync_probability
        )
        guided = importance is not None and _rand(generator) < emotion_probability
        shared = None
        if sync:
            shared_importance = (
                sum(importance[symbol][sample_index] for symbol in combination)
                if guided else None
            )
            shared = _span_mask(
                output["content_mask"][sample_index].bool(),
                rate,
                number_of_spans,
                location,
                generator,
                importance=shared_importance,
                temperature=emotion_temperature,
            )
        for symbol in combination:
            modality_index = MODALITY_INDEX[symbol]
            selected = shared
            if selected is None:
                selected = _span_mask(
                    output["content_mask"][sample_index].bool(),
                    rate,
                    number_of_spans,
                    location,
                    generator,
                    importance=importance[symbol][sample_index] if guided else None,
                    temperature=emotion_temperature,
                )
            corruption[sample_index, :, modality_index] |= selected

    output["corruption_mask"] = corruption
    output["reliability_mask"] = reliability & ~corruption
    text_corruption = corruption[..., 0]
    output["input_ids"][text_corruption] = 0
    output["token_type_ids"][text_corruption] = 0
    output["text_attention_mask"] = output["text_attention_mask"].bool() & ~text_corruption
    output["audio"][corruption[..., 1]] = 0.0
    output["vision"][corruption[..., 2]] = 0.0
    return output


def _corrupt_unaligned_batch(
    output: dict[str, Any],
    config: SpanMaskConfig,
    generator: torch.Generator,
    *,
    force_rate: float | None,
    force_modalities: str | None,
    force_location: str | None,
    force_spans: int | None,
    force_sync: bool | None,
    importance: dict[str, torch.Tensor] | None,
    emotion_probability: float,
    emotion_temperature: float,
) -> dict[str, Any]:
    """Mask contiguous spans on each modality's own temporal axis."""

    validity = {
        "T": output["content_mask"].bool(),
        "A": output["audio_structural_mask"].bool(),
        "V": output["vision_structural_mask"].bool(),
    }
    corruptions = {
        symbol: torch.zeros_like(mask, dtype=torch.bool)
        for symbol, mask in validity.items()
    }
    batch_size = int(output["input_ids"].shape[0])
    for sample_index in range(batch_size):
        if force_rate is None and _rand(generator) < config.clean_probability:
            continue
        combination = force_modalities
        if combination is None:
            combination = config.modality_combinations[
                _randint(0, len(config.modality_combinations), generator)
            ]
        rate = force_rate
        if rate is None:
            rate = config.min_rate + _rand(generator) * (config.max_rate - config.min_rate)
        location = force_location
        if location is None:
            location = config.locations[_randint(0, len(config.locations), generator)]
        number_of_spans = force_spans
        if number_of_spans is None:
            number_of_spans = _randint(config.min_spans, config.max_spans + 1, generator)
        sync = (
            force_sync
            if force_sync is not None
            else len(combination) > 1 and _rand(generator) < config.sync_probability
        )
        guided = importance is not None and _rand(generator) < emotion_probability
        relative_starts = None
        if sync:
            if guided:
                anchor = combination[_randint(0, len(combination), generator)]
                relative_starts = _guided_relative_starts(
                    validity[anchor][sample_index],
                    importance[anchor][sample_index],
                    rate,
                    number_of_spans,
                    location,
                    emotion_temperature,
                    generator,
                )
            else:
                relative_starts = [
                    _relative_start(location, generator) for _ in range(number_of_spans)
                ]
        for symbol in combination:
            corruptions[symbol][sample_index] |= _span_mask(
                validity[symbol][sample_index],
                rate,
                number_of_spans,
                location,
                generator,
                relative_starts=relative_starts,
                importance=(
                    importance[symbol][sample_index]
                    if guided and relative_starts is None else None
                ),
                temperature=emotion_temperature,
            )

    text_corruption = corruptions["T"]
    audio_corruption = corruptions["A"]
    vision_corruption = corruptions["V"]
    output["text_corruption_mask"] = text_corruption
    output["audio_corruption_mask"] = audio_corruption
    output["vision_corruption_mask"] = vision_corruption
    output["input_ids"][text_corruption] = 0
    output["token_type_ids"][text_corruption] = 0
    output["text_attention_mask"] = output["text_attention_mask"].bool() & ~text_corruption
    output["audio_reliability_mask"] = (
        output["audio_reliability_mask"].bool() & ~audio_corruption
    )
    output["vision_reliability_mask"] = (
        output["vision_reliability_mask"].bool() & ~vision_corruption
    )
    output["audio"][audio_corruption] = 0.0
    output["vision"][vision_corruption] = 0.0
    return output
