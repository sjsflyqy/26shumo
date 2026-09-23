#!/usr/bin/env python3
"""可视化多模态 PKL 特征文件。

兼容本题常见的三种数据格式：

1. 附件二：一个 PKL 中包含 train/valid/test 三个数据集；
2. 附件三：单个样本，外层通常由 test 键包装；
3. 附件四：单个样本，特征直接存放在顶层。

示例：
    python visualize_pkl.py ../data/attachment2/unaligned_50.pkl \
        --split train --index 0 --summary

    python visualize_pkl.py ../data/attachment3/aligned/0001.pkl \
        --output ../outputs/pkl_visualizations/attachment3_0001.png

依赖：
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import pickle
import textwrap
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_NAMES = ("train", "valid", "test")
LABEL_KEYS = (
    "classification_labels",
    "regression_labels",
    "classification_labels_A",
    "regression_labels_A",
    "classification_labels_V",
    "regression_labels_V",
    "classification_labels_T",
    "regression_labels_T",
)


def load_pickle(path: Path) -> Any:
    """读取 PKL；附件二文件较大，读取时需要足够内存。"""
    with path.open("rb") as file:
        return pickle.load(file)


def shape_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray{tuple(value.shape)}, {value.dtype}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def print_structure(obj: Any, prefix: str = "root", max_depth: int = 2) -> None:
    """在终端打印简洁的数据结构，避免直接打印大数组。"""
    if max_depth < 0:
        return
    if isinstance(obj, dict):
        print(f"{prefix}: dict({len(obj)} keys)")
        for key, value in obj.items():
            child = f"{prefix}.{key}"
            if isinstance(value, dict) and max_depth > 0:
                print_structure(value, child, max_depth - 1)
            else:
                print(f"  {child}: {shape_text(value)}")
    else:
        print(f"{prefix}: {shape_text(obj)}")


def dataset_size(split_data: dict[str, Any]) -> int | None:
    """从附件二某个 split 中推断样本数。"""
    preferred = ("id", "raw_text", "text", "audio", "vision", "text_bert")
    for key in preferred:
        value = split_data.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def scalarize(value: Any) -> Any:
    """把只含一个元素的数组转成 Python 标量，便于显示。"""
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def take_dataset_sample(split_data: dict[str, Any], index: int) -> dict[str, Any]:
    """从附件二 split 中取第 index 个样本。"""
    size = dataset_size(split_data)
    if size is None:
        raise ValueError("无法从所选 split 推断样本数量。")
    if not -size <= index < size:
        raise IndexError(f"样本索引 {index} 越界；该 split 共 {size} 个样本。")
    index %= size

    sample: dict[str, Any] = {}
    for key, value in split_data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == size:
            sample[key] = scalarize(value[index])
        elif isinstance(value, (list, tuple)) and len(value) == size:
            sample[key] = scalarize(value[index])
        else:
            # 元数据或与样本维无关的字段仍予保留。
            sample[key] = value
    return sample


def squeeze_single_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """移除附件三单样本文件中可能存在的 batch=1 维。"""
    result: dict[str, Any] = {}
    sequence_keys = {"text", "audio", "vision"}
    vector_keys = {"raw_text", "id", "audio_lengths", "vision_lengths", *LABEL_KEYS}

    for key, value in sample.items():
        if not isinstance(value, np.ndarray):
            result[key] = value
            continue

        if key in sequence_keys and value.ndim >= 3 and value.shape[0] == 1:
            result[key] = value[0]
        elif key == "text_bert" and value.ndim == 3 and value.shape[0] == 1:
            # 单样本附件三通常为 (1, 3, L)，附件四通常已经是 (3, L)。
            result[key] = value[0]
        elif key in vector_keys and value.ndim >= 1 and value.shape[0] == 1:
            result[key] = scalarize(value[0])
        else:
            result[key] = value
    return result


def select_sample(
    obj: Any, split: str, index: int
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """统一解析附件二、三、四，并返回样本、完整 split、格式说明。"""
    if not isinstance(obj, dict):
        raise TypeError(f"PKL 顶层应为 dict，实际为 {type(obj).__name__}。")

    available_splits = [key for key in SPLIT_NAMES if isinstance(obj.get(key), dict)]

    # 附件二：通常不止一个 split，或 split 内容的第一维明显大于 1。
    if available_splits:
        chosen = split if split in available_splits else available_splits[0]
        inner = obj[chosen]
        size = dataset_size(inner)
        if len(available_splits) > 1 or (size is not None and size > 1):
            return take_dataset_sample(inner, index), inner, f"数据集格式 / split={chosen}"

        # 附件三：外层常只有 test，里面只有一个样本。
        return squeeze_single_sample(inner), None, f"单样本格式 / 外层键={chosen}"

    # 附件四：字段直接位于顶层。
    return squeeze_single_sample(obj), None, "直接单样本格式"


def as_2d(value: Any) -> np.ndarray | None:
    """把序列特征规范为 (time, feature)，不强行解释高维数据。"""
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    return None


def explicit_length(sample: dict[str, Any], modality: str) -> int | None:
    key = f"{modality}_lengths"
    if key not in sample:
        return None
    value = np.asarray(sample[key]).reshape(-1)
    if value.size == 0:
        return None
    try:
        return int(value[0])
    except (TypeError, ValueError, OverflowError):
        return None


def text_attention_length(sample: dict[str, Any]) -> int | None:
    """text_bert 的第 2 行通常是 attention mask。"""
    value = sample.get("text_bert")
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] >= 2:
        mask = array[1]
        if np.all(np.isin(np.unique(mask), [0, 1])):
            return int(mask.sum())
    return None


def sequence_state(
    array: np.ndarray, declared_length: int | None = None, atol: float = 1e-12
) -> dict[str, np.ndarray | int | bool]:
    """识别每个时间步是有效、全零还是尾部填充。

    全零帧只能称为“候选缺失帧”。没有题目额外说明时，不能仅凭全零
    断言它一定是人为掩码造成的缺失。
    """
    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    observed = np.any(np.abs(finite) > atol, axis=1)
    time_steps = len(observed)
    inferred = declared_length is None

    if declared_length is None:
        nonzero_positions = np.flatnonzero(observed)
        length = int(nonzero_positions[-1] + 1) if nonzero_positions.size else 0
    else:
        length = int(np.clip(declared_length, 0, time_steps))

    valid_region = np.arange(time_steps) < length
    zero_candidate = valid_region & ~observed
    padding = ~valid_region
    return {
        "observed": observed,
        "zero_candidate": zero_candidate,
        "padding": padding,
        "length": length,
        "length_inferred": inferred,
    }


def reduce_feature_axis(array: np.ndarray, max_bins: int) -> np.ndarray:
    """对超高维特征做相邻维度平均，控制热力图尺寸。"""
    if array.shape[1] <= max_bins:
        return array
    edges = np.linspace(0, array.shape[1], max_bins + 1, dtype=int)
    return np.stack(
        [array[:, edges[i] : edges[i + 1]].mean(axis=1) for i in range(max_bins)],
        axis=1,
    )


def standardized_heatmap(
    array: np.ndarray, observed: np.ndarray, max_bins: int
) -> np.ndarray:
    """按特征维标准化，使不同量纲的模态更适合放在热力图中观察。"""
    matrix = np.nan_to_num(array.astype(np.float64, copy=False))
    matrix = reduce_feature_axis(matrix, max_bins)
    reference = matrix[observed] if np.any(observed) else matrix
    if reference.shape[0] == 0:
        return np.zeros_like(matrix)
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return np.clip((matrix - mean) / std, -3.0, 3.0)


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(scalarize(value))


def get_first(sample: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    for key in keys:
        if key in sample:
            return key, scalarize(sample[key])
    return None


def modality_arrays(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in ("text", "audio", "vision"):
        array = as_2d(sample.get(name))
        if array is not None:
            result[name] = array
    return result


def plot_modality(
    fig: Any,
    heat_ax: Any,
    norm_ax: Any,
    name_cn: str,
    array: np.ndarray | None,
    state: dict[str, Any] | None,
    max_feature_bins: int,
) -> None:
    if array is None or state is None:
        heat_ax.text(0.5, 0.5, f"无 {name_cn} 连续特征", ha="center", va="center")
        heat_ax.set_axis_off()
        norm_ax.text(0.5, 0.5, "不可用", ha="center", va="center")
        norm_ax.set_axis_off()
        return

    observed = np.asarray(state["observed"], dtype=bool)
    heat = standardized_heatmap(array, observed, max_feature_bins)
    image = heat_ax.imshow(
        heat.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
    )
    heat_ax.set_title(f"{name_cn}特征热力图  原始形状={tuple(array.shape)}")
    heat_ax.set_xlabel("时间步")
    heat_ax.set_ylabel("特征维/分箱")
    fig.colorbar(image, ax=heat_ax, fraction=0.025, pad=0.02, label="特征内 z-score")

    norm = np.linalg.norm(np.nan_to_num(array.astype(np.float64)), axis=1)
    x = np.arange(len(norm))
    norm_ax.plot(x, norm, color="#3366aa", linewidth=1.2, label="L2 范数")

    missing = np.asarray(state["zero_candidate"], dtype=bool)
    padding = np.asarray(state["padding"], dtype=bool)
    if np.any(missing):
        norm_ax.scatter(
            x[missing], np.zeros(missing.sum()), s=16, color="#e68613",
            marker="x", label="全零候选缺失帧", zorder=3,
        )
    if np.any(padding):
        first_padding = int(np.flatnonzero(padding)[0])
        norm_ax.axvspan(first_padding, len(norm) - 1, color="#bdbdbd", alpha=0.3,
                        label="尾部填充")
    norm_ax.set_title(
        f"帧强度：有效长度={state['length']}，候选缺失={int(missing.sum())}"
    )
    norm_ax.set_xlabel("时间步")
    norm_ax.set_ylabel("L2 范数")
    norm_ax.grid(alpha=0.2)
    handles, labels = norm_ax.get_legend_handles_labels()
    if handles:
        norm_ax.legend(handles, labels, fontsize=8, loc="upper right")


def resample_state(state: dict[str, Any], bins: int = 240) -> np.ndarray:
    observed = np.asarray(state["observed"], dtype=bool)
    missing = np.asarray(state["zero_candidate"], dtype=bool)
    padding = np.asarray(state["padding"], dtype=bool)
    values = np.ones(len(observed), dtype=float)
    values[missing] = 0.0
    values[padding] = -1.0
    if len(values) == 0:
        return np.full(bins, -1.0)
    indices = np.minimum((np.arange(bins) * len(values) / bins).astype(int), len(values) - 1)
    return values[indices]


def plot_sample(
    sample: dict[str, Any],
    source: Path,
    format_description: str,
    output: Path,
    max_feature_bins: int,
    dpi: int,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as exc:
        raise SystemExit(
            "缺少 matplotlib。请先执行：pip install matplotlib"
        ) from exc

    # 尽量使用系统中可用的中文字体；没有时 Matplotlib 会自行回退。
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    arrays = modality_arrays(sample)
    states: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        length = explicit_length(sample, name)
        if name == "text" and length is None:
            length = text_attention_length(sample)
        states[name] = sequence_state(array, length)

    fig = plt.figure(figsize=(17, 14), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, height_ratios=[1.15, 2.0, 2.0, 2.0, 0.8])

    info_ax = fig.add_subplot(grid[0, :])
    info_ax.axis("off")
    sample_id = get_first(sample, ("id", "video_id", "sample_id"))
    class_label = get_first(sample, ("classification_labels", "label", "class_label"))
    regression_label = get_first(sample, ("regression_labels", "score", "sentiment"))
    raw_text = safe_string(sample.get("raw_text"))

    fields = [f"文件：{source}", f"格式：{format_description}"]
    if sample_id:
        fields.append(f"样本 ID：{sample_id[1]}")
    if class_label:
        fields.append(f"分类标签（{class_label[0]}）：{class_label[1]}")
    if regression_label:
        fields.append(f"回归标签（{regression_label[0]}）：{regression_label[1]}")
    fields.append(
        "模态形状：" + ", ".join(
            f"{key}={tuple(value.shape)}" for key, value in arrays.items()
        )
    )
    if raw_text:
        fields.append("原始文本：" + textwrap.fill(raw_text, width=120))
    info_ax.text(
        0.01, 0.98, "\n".join(fields), ha="left", va="top", fontsize=10,
        transform=info_ax.transAxes,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f4f6f8", "edgecolor": "#ccd3da"},
    )

    for row, (key, title) in enumerate(
        (("text", "文本"), ("audio", "音频"), ("vision", "视觉")), start=1
    ):
        heat_ax = fig.add_subplot(grid[row, 0])
        norm_ax = fig.add_subplot(grid[row, 1])
        plot_modality(
            fig, heat_ax, norm_ax, title, arrays.get(key), states.get(key), max_feature_bins
        )

        # 没有 text 连续特征时，补画 text_bert 的 token/attention 信息。
        if key == "text" and key not in arrays and sample.get("text_bert") is not None:
            bert = np.asarray(sample["text_bert"])
            if bert.ndim == 2:
                heat_ax.set_axis_on()
                token_rows = min(bert.shape[0], 3)
                heat_ax.imshow(bert[:token_rows], aspect="auto", interpolation="nearest", cmap="viridis")
                heat_ax.set_title(f"text_bert（形状={tuple(bert.shape)}）")
                heat_ax.set_xlabel("token 位置")
                heat_ax.set_ylabel("0: ids / 1: mask / 2: type")

    availability_ax = fig.add_subplot(grid[4, :])
    ordered = [(key, states[key]) for key in ("text", "audio", "vision") if key in states]
    if ordered:
        matrix = np.stack([resample_state(state) for _, state in ordered])
        cmap = ListedColormap(["#d9d9d9", "#f2a65a", "#4c9f70"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        availability_ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        availability_ax.set_yticks(range(len(ordered)), [key for key, _ in ordered])
        availability_ax.set_xticks(
            [0, matrix.shape[1] // 4, matrix.shape[1] // 2, 3 * matrix.shape[1] // 4, matrix.shape[1] - 1],
            ["0%", "25%", "50%", "75%", "100%"],
        )
        availability_ax.set_xlabel("各模态归一化时间位置（不同模态的时间步不一定一一对齐）")
        availability_ax.set_title("模态可用性：绿色=非零，橙色=有效区内全零，灰色=尾部填充")
    else:
        availability_ax.text(0.5, 0.5, "没有可绘制的连续模态特征", ha="center", va="center")
        availability_ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("多模态 PKL 样本可视化", fontsize=17)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"样本图已保存：{output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def all_zero_ratios(array: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
    """计算每个样本在有效区间内的全零帧比例。"""
    if array.ndim != 3:
        return np.array([], dtype=float)
    zero_rows = ~np.any(np.abs(np.nan_to_num(array)) > 1e-12, axis=2)
    n_samples, time_steps = zero_rows.shape
    if lengths is None:
        # 无显式长度时，仅计算到最后一个非零帧，避免把尾部 padding 算作缺失。
        observed = ~zero_rows
        reverse_index = np.argmax(observed[:, ::-1], axis=1)
        inferred = time_steps - reverse_index
        inferred[~np.any(observed, axis=1)] = 0
        lengths = inferred
    lengths = np.asarray(lengths).reshape(-1).astype(int)
    lengths = np.clip(lengths, 0, time_steps)
    valid = np.arange(time_steps)[None, :] < lengths[:, None]
    denominator = np.maximum(lengths, 1)
    return (zero_rows & valid).sum(axis=1) / denominator


def plot_dataset_summary(
    split_data: dict[str, Any], source: Path, split: str, output: Path, dpi: int, show: bool
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("缺少 matplotlib。请先执行：pip install matplotlib") from exc

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"PKL 数据集概览：{source.name} / {split}", fontsize=16)

    class_item = get_first(split_data, ("classification_labels", "label", "class_label"))
    if class_item and np.asarray(class_item[1]).size > 1:
        labels = np.asarray(class_item[1]).reshape(-1)
        classes, counts = np.unique(labels, return_counts=True)
        axes[0, 0].bar([str(x) for x in classes], counts, color="#4c78a8")
        axes[0, 0].set_title("分类标签分布")
        axes[0, 0].set_xlabel("类别")
        axes[0, 0].set_ylabel("样本数")
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count, str(count), ha="center", va="bottom")
    else:
        axes[0, 0].text(0.5, 0.5, "无分类标签", ha="center", va="center")
        axes[0, 0].set_axis_off()

    reg_item = get_first(split_data, ("regression_labels", "score", "sentiment"))
    if reg_item and np.asarray(reg_item[1]).size > 1:
        values = np.asarray(reg_item[1], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        axes[0, 1].hist(values, bins=25, color="#59a14f", edgecolor="white")
        axes[0, 1].axvline(values.mean(), color="#b22222", linestyle="--", label=f"均值={values.mean():.3f}")
        axes[0, 1].set_title("回归标签分布")
        axes[0, 1].set_xlabel("标签值")
        axes[0, 1].set_ylabel("样本数")
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "无回归标签", ha="center", va="center")
        axes[0, 1].set_axis_off()

    length_found = False
    colors = {"audio": "#f28e2b", "vision": "#e15759", "text": "#4e79a7"}
    for modality in ("text", "audio", "vision"):
        length_key = f"{modality}_lengths"
        if length_key in split_data:
            values = np.asarray(split_data[length_key]).reshape(-1)
            axes[1, 0].hist(values, bins=25, alpha=0.55, label=modality, color=colors[modality])
            length_found = True
    if length_found:
        axes[1, 0].set_title("序列有效长度分布")
        axes[1, 0].set_xlabel("有效时间步")
        axes[1, 0].set_ylabel("样本数")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "无显式长度字段", ha="center", va="center")
        axes[1, 0].set_axis_off()

    names: list[str] = []
    ratio_values: list[np.ndarray] = []
    for modality in ("text", "audio", "vision"):
        value = split_data.get(modality)
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        lengths = split_data.get(f"{modality}_lengths")
        ratios = all_zero_ratios(value, None if lengths is None else np.asarray(lengths))
        if ratios.size:
            names.append(modality)
            ratio_values.append(ratios)
    if ratio_values:
        # labels 参数兼容较老的 Matplotlib；新版本中虽已更名但仍保持兼容。
        axes[1, 1].boxplot(ratio_values, labels=names, showfliers=False)
        axes[1, 1].set_title("有效区间内全零帧比例（候选缺失率）")
        axes[1, 1].set_ylabel("比例")
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[1, 1].grid(axis="y", alpha=0.25)
    else:
        axes[1, 1].text(0.5, 0.5, "无可统计的三维序列数组", ha="center", va="center")
        axes[1, 1].set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"数据集概览已保存：{output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def resolve_output(input_path: Path, output_arg: str | None, split: str, index: int) -> Path:
    default_name = f"{input_path.stem}_{split}_{index:04d}.png"
    if output_arg is None:
        return Path.cwd() / "pkl_visualizations" / default_name
    output = Path(output_arg).expanduser()
    if output.suffix.lower() != ".png":
        return output / default_name
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="可视化附件二/三/四的多模态 PKL 文件。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pkl", type=Path, help="待解析的 PKL 文件路径")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="train", help="附件二的数据划分")
    parser.add_argument("--index", type=int, default=0, help="附件二中要显示的样本下标")
    parser.add_argument("--output", help="输出 PNG 文件或输出目录")
    parser.add_argument("--summary", action="store_true", help="附件二额外生成所选 split 的总体统计图")
    parser.add_argument("--inspect", action="store_true", help="只打印 PKL 结构，不生成图片")
    parser.add_argument("--show", action="store_true", help="保存后打开 Matplotlib 窗口")
    parser.add_argument("--dpi", type=int, default=160, help="输出图片 DPI")
    parser.add_argument("--max-feature-bins", type=int, default=120, help="热力图最多显示多少个特征分箱")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.pkl.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"文件不存在：{input_path}")
    if args.max_feature_bins < 2:
        raise SystemExit("--max-feature-bins 必须至少为 2。")

    print(f"正在读取：{input_path}")
    obj = load_pickle(input_path)
    print_structure(obj)
    if args.inspect:
        return

    sample, split_data, description = select_sample(obj, args.split, args.index)
    output = resolve_output(input_path, args.output, args.split, args.index)
    plot_sample(
        sample=sample,
        source=input_path,
        format_description=description,
        output=output,
        max_feature_bins=args.max_feature_bins,
        dpi=args.dpi,
        show=args.show,
    )

    if args.summary:
        if split_data is None:
            print("当前是单样本 PKL，跳过 --summary 数据集统计图。")
        else:
            summary_path = output.with_name(output.stem + "_summary.png")
            plot_dataset_summary(
                split_data, input_path, args.split, summary_path, args.dpi, args.show
            )


if __name__ == "__main__":
    main()
