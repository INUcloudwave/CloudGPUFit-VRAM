"""Side-by-side table: (1) production CPU baseline, (2) history-matched CPU trace -> xMem,
(3) GPU actual request trace -> xMem and GPU actual reserved (when GPU data exists).
Values are labelled: OBS = observed measurement, PRED = simulator output. MiB = 2^20 bytes."""
import argparse
import glob
import json
from pathlib import Path

EXP = Path("/exp") if Path("/exp").exists() else Path(__file__).resolve().parents[2]
M = 2 ** 20


def mib(x):
    return "—" if x is None else f"{x / M:.1f}"


def prefix_peak(per_step, n_steps):
    """peak reserved/allocated over resident placement + first n inference steps."""
    steps = [s for s in per_step if not s["step"].startswith("resident")]
    res = [s for s in per_step if s["step"].startswith("resident")]
    sel = res + steps[:n_steps]
    return max(s["peak_reserved"] for s in sel), max(s["peak_allocated"] for s in sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.loads((EXP / "manifest/common_manifest.json").read_text())
    lines = ["# reserved 오차 진단 요약 (자동 생성)", "", "- 단위 MiB(2^20). PRED=시뮬레이터 예측, OBS=실측. 용량 80GiB(압박 없음) 고정", ""]
    for c in man["cases"]:
        cid = c["case_id"]
        l4 = c["previous_L4_reference_only"]
        lines += [f"## {cid} ({c['role']})", ""]
        rows = [("이전 L4 memory 모드 피크 reserved / allocated (OBS, 참고치)", l4["memory"]["inference_peak_reserved"], l4["memory"]["inference_peak_allocated"]),
                ("이전 L4 performance 모드: 로딩 직후 reserved / allocated (OBS)", l4["performance"]["after_load_reserved"], l4["performance"]["after_load_allocated"]),
                ("(1) CPU 운영 기준선 reserved / allocated (PRED, 저장값)", c["cpu_production_baseline"]["predicted_reserved_bytes"], c["cpu_production_baseline"]["predicted_allocated_bytes"])]
        f = EXP / f"results/cpu/{cid}.analysis.json"
        if f.exists():
            d = json.loads(f.read_text())
            b = d.get("baseline", {})
            if b:
                rows += [("  A1 운영 방식 재현(시간 그룹) — 저장값 일치=" + str(b["A1_time_grouped_production_equivalent"]["matches_stored"]),
                          b["A1_time_grouped_production_equivalent"]["reserved"], b["A1_time_grouped_production_equivalent"]["allocated"]),
                         ("  A2 같은 trace 순차 재생", b["A2_sequential_xmem"]["reserved"], b["A2_sequential_xmem"]["allocated"]),
                         ("  A3 A2 + PyTorch2.6 10MiB 경계(진단)", b["A3_sequential_diag_pt26_boundary"]["reserved"], b["A3_sequential_diag_pt26_boundary"]["allocated"])]
            for key, h in d.get("history", {}).items():
                fh = h["full_history_replay"]["xmem_pinned"]
                fd = h["full_history_replay"]["diag_pt26_alloc_size_boundary"]
                iso = h.get("isolated_first_generate", {})
                rows.append(("(2) CPU 이력 trace: generate 1회 단독 (GPU식 입력 복제)", iso.get("peak_reserved"), iso.get("peak_allocated")))
                for n, label in [(1, "검증 forward 1회까지"), (2, "+warmup 1회"), (5, "R1 통제(검증1+warmup1+측정3)"), (7, "AWS performance 모드 끝")]:
                    if len([s for s in fh["per_step"] if not s["step"].startswith("resident")]) >= n:
                        r_, a_ = prefix_peak(fh["per_step"], n)
                        rows.append((f"  누적 이력: {label}", r_, a_))
                rows.append(("  AWS 구조 전체(performance→memory) 피크", fh["peak_reserved"], fh["peak_allocated"]))
                mm = fh.get("measured_window_peaks_by_mode", {}).get("memory")
                if mm:
                    rows.append(("  memory 모드 측정 구간 피크 (L4 memory 피크와 대응)", mm["peak_reserved"], mm["peak_allocated"]))
                rows.append(("  AWS 구조 전체, 10MiB 경계 진단 정책", fd["peak_reserved"], fd["peak_allocated"]))
        for rep in sorted(glob.glob(str(EXP / f"results/gpu/{cid}/*/import_report.json"))):
            g = json.loads(Path(rep).read_text())
            run_id = Path(rep).parent.name
            px = g["policies"]["xmem_pinned"]
            pdg = g["policies"]["diag_pt26_alloc_size_boundary"]
            rows.append((f"(3) GPU {run_id}: 실제 reserved 피크 (OBS, trace)", px["comparison"]["observed_peak_reserved_trace"], None))
            rows.append((f"    같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #{px['comparison']['first_decision_divergence_request_index']}, 불일치 {px['comparison']['decision_mismatches']}건",
                         px["replay"]["peak_reserved_bytes"], px["replay"]["peak_allocated_bytes"]))
            rows.append((f"    같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 {pdg['comparison']['decision_mismatches']}건",
                         pdg["replay"]["peak_reserved_bytes"], pdg["replay"]["peak_allocated_bytes"]))
            hy = g.get("hybrid_gpu_prefix_then_cpu_requests")
            if hy:
                rows.append((f"    GPU 로딩 구간 + CPU 요청 (hybrid PRED)", hy["peak_reserved"], hy["peak_allocated"]))
        lines += ["| 항목 | reserved | allocated |", "|---|---:|---:|"]
        lines += [f"| {n} | {mib(r)} | {mib(al)} |" for n, r, al in rows]
        lines.append("")
    Path(a.out).write_text("\n".join(lines))
    print(a.out)


if __name__ == "__main__":
    main()
