import itertools
from pathlib import Path

import numpy as np
import pandas as pd


FILES = {
    "B0": "test_results/BenchData/revise_b0_s300_Bgnn-SB10_20260814_222847.xlsx",
    "B0+Q": "test_results/BenchData/revise_b0q_s300_Bgnn-SB10_20260814_230136.xlsx",
    "B0+L": "test_results/BenchData/revise_b0l_s300_Bgnn-SB10_20260814_232721.xlsx",
    "B0+G": "test_results/BenchData/revise_b0g_s300_Bgnn-SB10_20260814_235336.xlsx",
    "B0+R": "test_results/BenchData/revise_b0r_s300_Bgnn-SB10_20260815_034932.xlsx",
    "B0+Q-s301": "test_results/BenchData/revise_b0q_s301_Bgnn-SB10_20260815_122417.xlsx",
    "B0+Q-s302": "test_results/BenchData/revise_b0q_s302_Bgnn-SB10_20260815_130607.xlsx",
    "B0+G-s301": "test_results/BenchData/revise_b0g_s301_Bgnn-SB10_20260815_132743.xlsx",
    "B0+G-s302": "test_results/BenchData/revise_b0g_s302_Bgnn-SB10_20260815_135157.xlsx",
}

PARAMETERS = {
    "B0": 550162, "B0+Q": 600466, "B0+L": 550418, "B0+G": 557458, "B0+R": 550162,
    "B0+Q-s301": 600466, "B0+Q-s302": 600466, "B0+G-s301": 557458, "B0+G-s302": 557458,
}


def average_ranks(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def exact_signed_rank(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isclose(diff, 0.0)]
    ranks = average_ranks(np.abs(diff))
    observed_pos = ranks[diff > 0].sum()
    total = ranks.sum()
    observed_stat = min(observed_pos, total - observed_pos)
    extreme = 0
    count = 0
    for signs in itertools.product((False, True), repeat=len(ranks)):
        positive = ranks[np.asarray(signs)].sum()
        stat = min(positive, total - positive)
        extreme += stat <= observed_stat + 1e-12
        count += 1
    return observed_stat, extreme / count


def bootstrap_mean_ci(diff, seed=20260815, draws=10000):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    means = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def holm_adjust(p_values):
    names = list(p_values)
    ordered = sorted(names, key=p_values.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (m - index) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return {name: adjusted[name] for name in names}


def main():
    reference = pd.read_csv("data/BenchData/BenchDataSolution.csv")
    labels = reference["benchname"].to_numpy()
    frames = {}
    for name, path in FILES.items():
        frame = pd.read_excel(path)
        assert len(frame) == len(reference) == 130
        assert frame.notna().all().all()
        frame["exact_gap"] = (frame["makespan"] - frame["ref_makespan"]) / frame["ref_makespan"] * 100
        frames[name] = frame

    groups = ["Brandimarte", "Hurink_edata", "Hurink_rdata", "Hurink_vdata"]
    print("SUMMARY")
    for name, frame in frames.items():
        fields = [name]
        for group in groups:
            mask = labels == group
            gaps = frame.loc[mask, "exact_gap"]
            fields.append(f"{group}={gaps.mean():.4f}/{gaps.median():.4f}/{gaps.std(ddof=0):.4f}")
        fields.append(f"time={frame['time'].mean():.6f}")
        fields.append(f"parameters={PARAMETERS[name]}")
        print(" | ".join(fields))

    brand_mask = labels == "Brandimarte"
    base = frames["B0"].loc[brand_mask, "exact_gap"].to_numpy()
    raw_p = {}
    rows = {}
    for index, name in enumerate(("B0+Q", "B0+L", "B0+G", "B0+R")):
        diff = frames[name].loc[brand_mask, "exact_gap"].to_numpy() - base
        ci = bootstrap_mean_ci(diff, seed=20260815 + index)
        statistic, p_value = exact_signed_rank(diff)
        raw_p[name] = p_value
        rows[name] = (diff.mean(), ci[0], ci[1], statistic, p_value)
    adjusted = holm_adjust(raw_p)

    print("BRANDIMARTE_PAIRED_VARIANT_MINUS_B0")
    for name, row in rows.items():
        mean, low, high, statistic, p_value = row
        print(
            f"{name} mean={mean:.6f} ci95=[{low:.6f},{high:.6f}] "
            f"wilcoxon={statistic:.3f} raw_p={p_value:.6f} holm_p={adjusted[name]:.6f}"
        )

    print("STABILITY_ACROSS_TRAINING_SEEDS")
    for variant, names in {
        "B0+Q": ("B0+Q", "B0+Q-s301", "B0+Q-s302"),
        "B0+G": ("B0+G", "B0+G-s301", "B0+G-s302"),
    }.items():
        fields = [variant]
        for group in groups:
            mask = labels == group
            seed_means = np.asarray([frames[name].loc[mask, "exact_gap"].mean() for name in names])
            fields.append(
                f"{group} seeds={','.join(f'{value:.4f}' for value in seed_means)} "
                f"mean={seed_means.mean():.4f} seed_std={seed_means.std(ddof=0):.4f}"
            )
        fields.append(f"time_mean={np.mean([frames[name]['time'].mean() for name in names]):.6f}")
        print(" | ".join(fields))

    for path in FILES.values():
        assert Path(path).is_file()


if __name__ == "__main__":
    main()
