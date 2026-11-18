"""Turn the experiment CSVs into docs/exp.md.

Usage:
    python -m experiments.report --results-dir experiments/results \
        --out docs/exp.md --shot 1 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import common  # noqa: E402

BENCHMARKS = ["deepglobe", "isic", "lung", "fss", "suim"]
DISPLAY = {
    "deepglobe": "Deepglobe",
    "isic": "ISIC",
    "lung": "Chest X-ray",
    "fss": "FSS-1000",
    "suim": "SUIM",
}
CD_FSS = ["deepglobe", "isic", "lung", "fss"]
GROUPS = ["A", "B", "C", "X"]
GROUP_TITLE = {
    "A": "Nhóm A — Gộp tầng (Layer Fusion)",
    "B": "Nhóm B — Cắt ngưỡng (Thresholding)",
    "C": "Nhóm C — Hậu xử lý (Post-processing)",
    "X": "Kết hợp A + B + C",
}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_results(results_dir: Path, shot: int, seed: int) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {g: {} for g in GROUPS}
    for group in GROUPS:
        for b in BENCHMARKS:
            rows = read_csv(results_dir / f"{b}_shot{shot}_seed{seed}_group{group}.csv")
            if rows:
                out[group][b] = rows
    return out


def load_commits(results_dir: Path) -> Dict[str, str]:
    path = results_dir / "commits.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    sha = common.git_commit()
    return {g: sha for g in GROUPS}


def index_rows(rows_by_bench: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for b, rows in rows_by_bench.items():
        for r in rows:
            out.setdefault(b, {})[r["name"]] = r
    return out


def avg_metric(
    by_bench: Dict[str, Dict[str, Dict[str, Any]]],
    name: str,
    metric: str,
    benches: Sequence[str] = BENCHMARKS,
) -> Optional[float]:
    vals = []
    for b in benches:
        row = by_bench.get(b, {}).get(name)
        if row is not None:
            vals.append(float(row[metric]))
    return sum(vals) / len(vals) if vals else None


def fmt(v: Optional[float], digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def delta(v: Optional[float], base: Optional[float]) -> str:
    if v is None or base is None:
        return "—"
    d = v - base
    return f"{d:+.2f}"


def commit_cell(commits: Dict[str, str], group: str) -> str:
    sha = commits.get(group, "")
    if not sha or sha == "unknown":
        return "—"
    return f"[{sha[:8]}]({common.commit_link(sha)})"


def auto_note(row: Dict[str, Any], base: Dict[str, Any]) -> str:
    dmiou = float(row["mIoU"]) - float(base["mIoU"])
    dfb = float(row["FB-IoU"]) - float(base["FB-IoU"])
    dfg = float(row["FG_pred"]) - float(base["FG_pred"])
    if abs(dmiou) < 0.3 and abs(dfb) < 0.3:
        return "tương đương baseline"
    if dmiou > 0.3 and dfb < -0.3:
        return "mIoU tăng nhưng FB-IoU giảm — kiểm tra dự đoán thừa foreground"
    if dmiou > 0.3 and dfb >= -0.3:
        return "cải thiện (mIoU và FB-IoU không giảm)"
    if dmiou < -0.3 and dfb > 0.3:
        return "mIoU giảm nhưng FB-IoU tăng — dự đoán ít foreground hơn"
    if dmiou < -0.3:
        return "kém hơn baseline"
    return "thay đổi nhỏ"


def table_group(
    group: str,
    by_bench: Dict[str, Dict[str, Dict[str, Any]]],
    baseline: Dict[str, Any],
    commits: Dict[str, str],
    include_note: bool = True,
) -> str:
    # Preserve config order as it appears in the first benchmark CSV.
    names: List[str] = []
    for b in BENCHMARKS:
        for r in by_bench.get(b, {}).values():
            if r["name"] not in names:
                names.append(r["name"])
    lines = [
        "| Mã TN | Thay đổi | Shot | Seed | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB | Commit | Nhận xét |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        rows = [by_bench[b][name] for b in BENCHMARKS if name in by_bench.get(b, {})]
        if not rows:
            continue
        miou = avg_metric(by_bench, name, "mIoU")
        fb = avg_metric(by_bench, name, "FB-IoU")
        fg = avg_metric(by_bench, name, "FG_pred")
        base_rows = [by_bench.get(b, {}).get(baseline["name"]) for b in BENCHMARKS]
        base_rows = [r for r in base_rows if r is not None]
        if base_rows:
            base_miou = sum(float(r["mIoU"]) for r in base_rows) / len(base_rows)
            base_fb = sum(float(r["FB-IoU"]) for r in base_rows) / len(base_rows)
            base_for_note = base_rows[0]
        else:
            # The group's own baseline is not part of that group's CSV (e.g.
            # group B compares against B0, which lives in group B, while the
            # reference metrics come from group A).  Use the supplied row.
            base_miou = float(baseline.get("mIoU", 0.0))
            base_fb = float(baseline.get("FB-IoU", 0.0))
            base_for_note = baseline
        note = auto_note(rows[0], base_for_note) if include_note else ""
        desc = rows[0].get("note", "")
        lines.append(
            f"| `{name}` | {desc} | {rows[0].get('shot', '1')} | {rows[0].get('seed', '0')} | "
            f"{fmt(miou)} | {fmt(fb)} | {fmt(fg)} | {delta(miou, base_miou)} | "
            f"{delta(fb, base_fb)} | {commit_cell(commits, group)} | {note} |"
        )
    return "\n".join(lines)


def table_per_dataset(
    configs: Sequence[Tuple[str, str]],
    by_group: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
) -> str:
    headers = ["Cấu hình"] + [f"{DISPLAY[b]} (m/FB)" for b in BENCHMARKS] + ["TB 5 tập (m/FB)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for label, name in configs:
        cells = [label]
        mious, fbs = [], []
        found = False
        for b in BENCHMARKS:
            row = None
            for g in GROUPS:
                row = by_group.get(g, {}).get(b, {}).get(name)
                if row:
                    break
            if row:
                found = True
                cells.append(f"{float(row['mIoU']):.1f} / {float(row['FB-IoU']):.1f}")
                mious.append(float(row["mIoU"]))
                fbs.append(float(row["FB-IoU"]))
            else:
                cells.append("—")
        if not found:
            continue
        cells.append(
            f"{sum(mious)/len(mious):.1f} / {sum(fbs)/len(fbs):.1f}" if mious else "—"
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_name(
    by_bench: Dict[str, Dict[str, Dict[str, Any]]],
    metric: str = "mIoU",
    exclude: Sequence[str] = (),
) -> Optional[str]:
    best, best_v = None, -1e9
    for b, rows in by_bench.items():
        for name, row in rows.items():
            if name in exclude:
                continue
            v = avg_metric(by_bench, name, metric)
            if v is not None and v > best_v:
                best, best_v = name, v
    return best


def generate(
    results_dir: Path,
    out_path: Path,
    shot: int = 1,
    seed: int = 0,
) -> str:
    results = load_results(results_dir, shot, seed)
    commits = load_commits(results_dir)
    indexed = {g: index_rows(results[g]) for g in GROUPS}

    base_a = "A0_mean"
    base_c = "C0_crf_dynamic"
    # A0 is the no-post-processing baseline in group A/B; C0 is the full
    # baseline in group C.
    base_row_a = None
    for b in BENCHMARKS:
        if base_a in indexed.get("A", {}).get(b, {}):
            base_row_a = indexed["A"][b][base_a]
            break
    if base_row_a is None:
        # fall back to C1
        for b in BENCHMARKS:
            if "C1_off" in indexed.get("C", {}).get(b, {}):
                base_row_a = indexed["C"][b]["C1_off"]
                break
    base_row_c = None
    for b in BENCHMARKS:
        if base_c in indexed.get("C", {}).get(b, {}):
            base_row_c = indexed["C"][b][base_c]
            break
    if base_row_c is None:
        base_row_c = base_row_a
    base_row_a = base_row_a or {"name": base_a, "mIoU": 0, "FB-IoU": 0, "FG_pred": 0}

    # Episode counts / protocol
    meta = None
    for b in BENCHMARKS:
        p = results_dir / f"{b}_shot{shot}_seed{seed}_groupA.json"
        if p.exists():
            with open(p) as f:
                meta = json.load(f).get("meta")
            break
    n_episodes = meta.get("n_episodes") if meta else "?"
    code_commit = meta.get("code_commit") if meta else "unknown"
    device = meta.get("device") if meta else "?"

    lines: List[str] = []
    lines.append("## 6. Nhật ký thí nghiệm\n")
    lines.append(
        "> Kết quả dưới đây được sinh tự động bởi `experiments/report.py` từ "
        "`experiments/results/*.csv`. Mỗi dòng có link tới commit chứa kết quả tương ứng.\n"
    )
    lines.append("### 6.0 Giao thức chạy\n")
    lines.append(
        f"- **Shot:** {shot}; **seed:** {seed}; **số episode/tập:** {n_episodes} "
        "(lấy mẫu phân tầng đều trên toàn bộ test split; đặt `--max-episodes 0` để chạy toàn bộ).\n"
        f"- **Thiết bị sinh cache adapter:** `{device}`; code commit `{code_commit}`.\n"
        "- **Adapter:** đóng băng ResNet50 + adapter 2 lớp Conv 1×1, SGD 25 epoch, lr 0.01, "
        "2 augmentation shear tối đa 20° (đúng cấu hình gốc). Adapter được fit một lần cho "
        "mỗi lớp (first-episode) và **dùng lại** cho mọi biến thể đầu ra.\n"
        "- **Cache:** 13 bản đồ `q̂_pred_l` ở độ phân giải 100×100 (đúng nơi baseline gộp), "
        "cùng ảnh query/support, mặt nạ GT/support và pseudo-episode của Thuật toán 1. "
        "Nhờ vậy mọi biến thể được so sánh trên **cùng đặc trưng đã thích ứng**.\n"
        "- **Chỉ số:** mIoU (trung bình IoU foreground theo lớp), FB-IoU (trung bình IoU "
        "background và foreground toàn cục), %FG (tỉ lệ foreground dự đoán). "
        "`mIoU TB`/`FB-IoU TB` là trung bình trên 5 tập (Deepglobe, ISIC, Chest X-ray, FSS-1000, SUIM).\n"
        "- **Nhóm C và các biến thể CRF:** DenseCRF chạy CPU rất chậm (~1.7 s/ảnh cho 5 vòng), "
        "nên nhóm C được đánh giá trên một tập con phân tầng "
        f"(`--crf-max-episodes`, mặc định 200) và **mọi biến thể trong nhóm C dùng cùng tập con** "
        "để so sánh có ghép cặp. Nhóm A/B chạy trên toàn bộ episode đã cache.\n"
    )
    lines.append(
        "\n> **Lưu ý:** đây là kết quả tự chạy trên một tập con phân tầng, không phải con số "
        "trong Bảng 4 của bài báo. Dùng cột Δ so với baseline tự chạy, không so trực tiếp "
        "với 60.7/69.2 của bài báo.\n"
    )

    # Baselines
    lines.append("\n### 6.1 Baseline tự chạy\n")
    lines.append(
        table_per_dataset(
            [
                ("Baseline no-pp (A0+B0+C1)", base_a),
                ("Baseline đầy đủ (A0+B0+C0) †", base_c),
            ],
            indexed,
        )
    )
    lines.append(
        "\n† C0 dùng tập con CRF (xem mục 6.0); A0/A0+B0+C1 dùng toàn bộ episode đã cache."
    )
    if base_row_a and base_row_c:
        lines.append(
            f"\nBaseline no-pp trung bình 5 tập: **{float(base_row_a['mIoU']):.2f} / "
            f"{float(base_row_a['FB-IoU']):.2f}**; baseline đầy đủ: **"
            f"{float(base_row_c['mIoU']):.2f} / {float(base_row_c['FB-IoU']):.2f}**.\n"
        )

    # Group tables
    for group in GROUPS:
        if not indexed.get(group):
            continue
        lines.append(f"\n### 6.{GROUPS.index(group)+2} {GROUP_TITLE[group]}\n")
        if group == "C":
            baseline_for_group = base_row_c
        elif group == "B":
            baseline_for_group = None
            for b in BENCHMARKS:
                if "B0_pred_mean" in indexed.get("B", {}).get(b, {}):
                    baseline_for_group = indexed["B"][b]["B0_pred_mean"]
                    break
            baseline_for_group = baseline_for_group or base_row_a
        else:
            baseline_for_group = base_row_a
        lines.append(
            table_group(
                group,
                indexed[group],
                baseline_for_group or base_row_a,
                commits,
                include_note=True,
            )
        )
        if group == "C":
            lines.append(
                "\n*Nhóm C dùng tập con CRF; `C1_off` trong bảng này cũng chạy trên đúng "
                "tập con đó nên so sánh là ghép cặp.*\n"
            )

    # A5 grid compact
    if indexed.get("A"):
        names_a5 = []
        for b in BENCHMARKS:
            for name in indexed["A"].get(b, {}):
                if name.startswith("A5_") and name not in names_a5:
                    names_a5.append(name)
        if names_a5:
            lines.append("\n### 6.6 Lưới trọng số A5 (w_L, w_M, w_H)\n")
            lines.append(
                "| Cấu hình | mIoU TB | FB-IoU TB | % FG | ΔmIoU | ΔFB |"
                "\n|---|---|---|---|---|---|"
            )
            base_miou = avg_metric(indexed["A"], "A0_mean", "mIoU")
            base_fb = avg_metric(indexed["A"], "A0_mean", "FB-IoU")
            for name in names_a5:
                miou = avg_metric(indexed["A"], name, "mIoU")
                fb = avg_metric(indexed["A"], name, "FB-IoU")
                fg = avg_metric(indexed["A"], name, "FG_pred")
                lines.append(
                    f"| `{name}` | {fmt(miou)} | {fmt(fb)} | {fmt(fg)} | "
                    f"{delta(miou, base_miou)} | {delta(fb, base_fb)} |"
                )

    # Detailed per-dataset table for notable configs
    notable: List[Tuple[str, str]] = [
        ("Baseline no-pp (A0+B0+C1)", base_a),
        ("Baseline đầy đủ (A0+B0+C0) †", base_c),
    ]
    for group, label in (("A", "A"), ("B", "B"), ("C", "C"), ("X", "X")):
        name = best_name(indexed.get(group, {}), "mIoU")
        if name and name not in [n for _, n in notable]:
            notable.append((f"Best {label}: {name}", name))
    lines.append("\n### 6.7 Chi tiết theo tập dữ liệu (các cấu hình đáng chú ý)\n")
    lines.append(table_per_dataset(notable, indexed))

    # Findings / caveats
    lines.append("\n### 6.8 Nhận xét & hạn chế\n")
    lines.append(
        "- Mọi biến thể đều là heuristic không học; kết quả phụ thuộc vào chất lượng "
        "đặc trưng đã thích ứng và vào việc ước lượng đúng tỉ lệ foreground.\n"
        "- Các biến thể làm tăng mIoU nhưng giảm FB-IoU hoặc tăng mạnh %FG cần được "
        "coi là **dự đoán thừa foreground**, không phải cải thiện thực sự.\n"
        "- Nhóm C (đặc biệt CRF) mới chạy trên tập con; cần chạy lại toàn bộ episode "
        "nếu muốn kết luận chắc chắn. Nhóm A/B đã chạy trên toàn bộ episode đã cache.\n"
        "- Kết quả 1 seed; để kết luận chắc chắn cần chạy thêm seed và báo cáo "
        "trung bình ± độ lệch chuẩn (xem mục 7 của README).\n"
        "- Deepglobe có nhãn nhiễu; chênh lệch nhỏ trên tập này không nên diễn giải quá mức.\n"
    )
    lines.append(
        "\n### 6.9 Cách chạy lại\n"
        "\n```bash\n"
        "# 1. Cache adapter + 13 layer predictions (một lần cho mỗi tập/shot/seed)\n"
        "python -m experiments.cache_features --benchmark lung --datapath /path/to/lungsegmentation \\\n"
        "    --shot 1 --seed 0 --max-episodes 600\n"
        "\n# 2. Chạy từng nhóm offline trên cache\n"
        "python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group A\n"
        "python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group B\n"
        "python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group C\n"
        "python -m experiments.run_experiments --benchmark lung --shot 1 --seed 0 --group X\n"
        "\n# 3. Sinh lại tài liệu\n"
        "python -m experiments.report --shot 1 --seed 0 --out docs/exp.md\n"
        "```\n"
    )

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[report] wrote {out_path} ({len(text)} bytes)")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "docs" / "exp.md"))
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(Path(args.results_dir), Path(args.out), args.shot, args.seed)


if __name__ == "__main__":
    main()
