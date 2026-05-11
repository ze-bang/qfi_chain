#!/usr/bin/env python3
"""aggregate_metts.py --- Average per-sample METTS observables and jackknife.

Inputs: a glob of per-sample JSON files written by run_metts_Tfin.jl,
        all sharing the same (phase, N, T).
Output: a single JSON with mean + jackknife errors of the relevant arrays
        (one_pt, two_pt_re, two_pt_im, FQ_Szpi, OXYr_means, s_r, Sigma_rr).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_samples(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern!r}")
    return [json.loads(Path(p).read_text()) for p in files]


def jackknife(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, jackknife std) along axis 0."""
    n = values.shape[0]
    if n < 2:
        return values.mean(axis=0), np.zeros_like(values.mean(axis=0))
    total = values.sum(axis=0)
    jack = (total - values) / (n - 1)            # leave-one-out means
    mean = jack.mean(axis=0)
    var = (n - 1) / n * ((jack - mean) ** 2).sum(axis=0)
    return values.mean(axis=0), np.sqrt(var)


def stack(samples: list[dict], key: str) -> np.ndarray:
    return np.stack([np.asarray(s["observables"][key]) for s in samples], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help="glob pattern, e.g. 'states/Tfin/U_N32_T0.10_s*.json'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = load_samples(args.samples)
    meta0 = samples[0]["metadata"].copy()
    meta0["n_samples"] = len(samples)

    out_obs: dict[str, dict | list | int] = {}
    for key in ("one_pt", "two_pt_re", "two_pt_im",
                "OXYr_means", "s_r", "Sigma_rr"):
        arr = stack(samples, key)
        mean, err = jackknife(arr)
        out_obs[f"{key}_mean"] = mean.tolist()
        out_obs[f"{key}_err"] = err.tolist()

    fq = np.array([s["observables"]["FQ_Szpi"] for s in samples])
    fq_mean, fq_err = jackknife(fq.reshape(-1, 1))
    out_obs["FQ_Szpi_mean"] = float(fq_mean[0])
    out_obs["FQ_Szpi_err"] = float(fq_err[0])
    out_obs["N"] = int(samples[0]["observables"]["N"])
    out_obs["r_max"] = int(samples[0]["observables"]["r_max"])

    payload = {"metadata": meta0, "observables": out_obs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}  ({len(samples)} samples)")


if __name__ == "__main__":
    main()
