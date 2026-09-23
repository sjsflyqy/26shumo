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


def _span_mask(
    valid_positions: torch.Tensor,
    rate: float,
    spans: int,
    location: str,
    generator: torch.Generator,
) -> torch.Tensor:
    positions = torch.nonzero(valid_positions, as_tuple=False).flatten()
    output = torch.zeros_like(valid_positions, dtype=torch.bool)
    if positions.numel() == 0:
        return output
    total = max(1, min(int(round(rate * positions.numel())), int(positions.numel())))
    spans = max(1, min(spans, total))
    base, remainder = divmod(total, spans)
    for span_index in range(spans):
        span_length = base + int(span_index < remainder)
        start = _choose_start(int(positions.numel()), span_length, location, generator)
        output[positions[start : start + span_length]] = True
    return output


def corrupt_aligned_batch(
    batch: dict[str, Any],
    config: SpanMaskConfig,
    generator: torch.Generator,
    *,
    force_rate: float | None = None,
    force_modalities: str | None = None,
    force_location: str | None = None,
    force_spans: int | None = None,
) -> dict[str, Any]:
    """Create a local-span-corrupted student view without mutating ``batch``."""

    output = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
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
        sync = len(combination) > 1 and _rand(generator) < config.sync_probability
        shared = None
        if sync:
            shared = _span_mask(
                output["content_mask"][sample_index].bool(),
                rate,
                number_of_spans,
                location,
                generator,
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
