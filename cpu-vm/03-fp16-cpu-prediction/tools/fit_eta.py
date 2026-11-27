#!/usr/bin/env python3
"""Fit per-model run-time constants from a calibration run and extrapolate to the 680-case table.

  python3 tools/fit_eta.py --run-name fp32-calib [--jobs 1]
Per model:  t_run = a*F_lin + b*F_att  (least squares, a,b >= 0) on measured invocation seconds;
            post   = c * request_events (export+parse+replay), events/step fitted per model;
            case   = model_load + 4*t_run + post.
FLOP formulas come from tools/estimate_flops.py (M table). Rough planning number only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from estimate_flops import M, GF_VIS  # noqa: E402

EXP = Path("/home/inuos/cgf-fp16-680-20261111")


def flops(c):
    b = c["batch_size"]
    if c["kind"] == "vision_cls":
        return GF_VIS[c["short"]] * 1e9 * b * (c["resolution"] / 224) ** 2, 0.0
    L, H, Hh, dh, P, V = M[c["short"]]
    l, n = c["input_len"], c["new_tokens"]
    if c["kind"] == "decoder_llm":
        return 2 * P * b * (l + n) + 2 * V * H * b * n, 4 * b * Hh * dh * L * (l * l + sum(range(l, l + n)))
    return 2 * (P / 2) * b * (l + n) + 2 * V * H * b * n, 4 * b * Hh * dh * L * (l * l + sum(range(1, n + 1)) + n * l)


def nnls2(rows):
    """min ||a*x1 + b*x2 - y|| with a,b >= 0 (2 unknowns, brute-force on active sets)."""
    best = None
    def fit(use):
        import itertools
        xs = [[r[0][i] for i in use] for r in rows]
        ys = [r[1] for r in rows]
        if len(use) == 1:
            num = sum(x[0] * y for x, y in zip(xs, ys)); den = sum(x[0] ** 2 for x in xs) or 1e-30
            coef = [max(num / den, 0)]
        else:
            s11 = sum(x[0] ** 2 for x in xs); s22 = sum(x[1] ** 2 for x in xs); s12 = sum(x[0] * x[1] for x in xs)
            t1 = sum(x[0] * y for x, y in zip(xs, ys)); t2 = sum(x[1] * y for x, y in zip(xs, ys))
            det = s11 * s22 - s12 ** 2
            if abs(det) < 1e-30:
                return None
            coef = [(t1 * s22 - t2 * s12) / det, (t2 * s11 - t1 * s12) / det]
            if min(coef) < 0:
                return None
        full = [0.0, 0.0]
        for i, u in enumerate(use):
            full[u] = coef[i]
        err = sum((full[0] * r[0][0] + full[1] * r[0][1] - r[1]) ** 2 for r in rows)
        return err, full
    for use in ([0, 1], [0], [1]):
        f = fit(use)
        if f and (best is None or f[0] < best[0]):
            best = f
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--dtype", default=None, help="dtype for the extrapolated table (default: calib dtype)")
    ap.add_argument("--timeout-hours", type=float, nargs="*", default=[1, 2, 4, 6])
    a = ap.parse_args()
    by_model = {}
    for cdir in sorted((EXP / "runs" / a.run_name / "cases").iterdir()):
        rp = cdir / "result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        if r.get("status") != "ok":
            continue
        c = r["case"]
        t = r["timings_s"]
        meas = [x["seconds_incl_profiler"] for x in r["runs"]]
        fl = flops(c)
        steps = c["new_tokens"] or 1
        ev = r["trace_stats"]["memory_events"]
        post = t.get("trace_export", 0) + t.get("trace_parse", 0) + t.get("xmem_replay", 0)
        m = by_model.setdefault(c["short"], {"rows": [], "load": [], "post_per_ev": [], "ev_per_step": [],
                                              "rss_peak": []})
        m["rows"].append(((fl[0], fl[1]), sum(meas) / len(meas)))
        m["load"].append(t["model_load"])
        m["post_per_ev"].append(post / max(ev, 1))
        m["ev_per_step"].append(ev / 4 / steps)
        m["rss_peak"].append((c["batch_size"], c.get("input_len"), c.get("new_tokens"), c.get("resolution"),
                              (r.get("runner") or {}).get("container_memory_peak_bytes")))
    cases = [json.loads(x) for x in (EXP / "manifest/cases.jsonl").read_text().splitlines()]
    tot, per, est = 0, {}, []
    fits = {}
    for short, m in by_model.items():
        coef = nnls2(m["rows"])
        fits[short] = {"a_s_per_flop_lin": coef[0], "b_s_per_flop_att": coef[1],
                       "eff_gflops_lin": 1e-9 / coef[0] if coef[0] else None,
                       "eff_gflops_att": 1e-9 / coef[1] if coef[1] else None,
                       "load_s": max(m["load"]), "post_s_per_event": max(m["post_per_ev"]),
                       "events_per_step": max(m["ev_per_step"]), "calib_points": len(m["rows"]),
                       "rss_peaks": m["rss_peak"]}
    for c in cases:
        f = fits.get(c["short"])
        if f is None:
            continue
        lin, att = flops(c)
        steps = c["new_tokens"] or 1
        s = f["load_s"] + 4 * (f["a_s_per_flop_lin"] * lin + f["b_s_per_flop_att"] * att) + \
            f["post_s_per_event"] * f["events_per_step"] * 4 * steps + 15
        tot += s
        per[c["short"]] = per.get(c["short"], 0) + s
        est.append((s, c["case_id"]))
    est.sort()
    out = {"calib_run": a.run_name, "models_fitted": sorted(fits), "missing_models": sorted({c["short"] for c in cases} - set(fits)),
           "total_sequential_hours": tot / 3600, "per_model_hours": {k: round(v / 3600, 1) for k, v in sorted(per.items(), key=lambda x: -x[1])},
           "largest_cases_hours": [(cid, round(s / 3600, 2)) for s, cid in est[-10:]],
           "timeout_scenarios": {}}
    for h in a.timeout_hours:
        ok = [s for s, _ in est if s <= h * 3600]
        out["timeout_scenarios"][f"{h}h"] = {"cases_done": len(ok), "timeouts": len(est) - len(ok),
                                             "hours": round((sum(ok) + (len(est) - len(ok)) * h * 3600) / 3600, 1)}
    out["fits"] = fits
    dst = EXP / "results" / a.run_name / "eta.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "fits"}, indent=1))
    for k, f in fits.items():
        print(k, "lin GF/s", f["eff_gflops_lin"] and round(f["eff_gflops_lin"], 1), "att GF/s",
              f["eff_gflops_att"] and round(f["eff_gflops_att"], 1), "ev/step", round(f["events_per_step"]))


if __name__ == "__main__":
    main()
