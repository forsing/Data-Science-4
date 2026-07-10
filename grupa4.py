"""Grupa 4 — korelacija / zavisnost (Loto 7/39)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4648_k55.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def presence_matrix(draws: np.ndarray) -> np.ndarray:
    """Binary presence T×39 (kolone = brojevi 1..39)."""
    x = np.zeros((len(draws), FRONT_N), dtype=float)
    for i, draw in enumerate(draws):
        for n in draw.tolist():
            x[i, n - 1] = 1.0
    return x


def pearson_spearman_matrices(draws: np.ndarray) -> dict:
    """Pearson i Spearman korelacija među presence kolonama."""
    x = presence_matrix(draws)
    # Pearson
    pearson = np.corrcoef(x, rowvar=False)
    # Spearman = Pearson na rangovima
    ranks = np.apply_along_axis(lambda col: col.argsort().argsort().astype(float), 0, x)
    spearman = np.corrcoef(ranks, rowvar=False)
    return {"pearson": pearson, "spearman": spearman}


def top_corr_pairs(mat: np.ndarray, top_k: int = 15) -> list[tuple]:
    pairs = []
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            v = float(mat[i, j])
            if np.isnan(v):
                continue
            pairs.append((i + 1, j + 1, v))
    pairs.sort(key=lambda t: (-abs(t[2]), t[0], t[1]))
    return pairs[:top_k]


def mutual_information_matrix(draws: np.ndarray) -> np.ndarray:
    """Puna MI matrica za binarne presence parove."""
    x = presence_matrix(draws)
    mi = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            a = x[:, i]
            b = x[:, j]
            val = 0.0
            for aa in (0.0, 1.0):
                for bb in (0.0, 1.0):
                    pxy = np.mean((a == aa) & (b == bb))
                    px = np.mean(a == aa)
                    py = np.mean(b == bb)
                    if pxy > 0 and px > 0 and py > 0:
                        val += pxy * np.log2(pxy / (px * py))
            mi[i, j] = val
            mi[j, i] = val
    return mi


def mutual_information_pairs(draws: np.ndarray, top_k: int = 15) -> list[tuple]:
    """MI za binarne presence parove — top_k."""
    mi = mutual_information_matrix(draws)
    out = []
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            out.append((i + 1, j + 1, float(mi[i, j])))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out[:top_k]


def partial_corr_proxy(draws: np.ndarray, top_k: int = 10) -> list[tuple]:
    """
    Proxy partial correlation preko precision matrice (pseudo-inverse korelacije).
    Veći |precision_ij| ≈ jača uslovna veza.
    """
    x = presence_matrix(draws)
    # dodaj malu dijagonalu radi stabilnosti
    cov = np.cov(x, rowvar=False) + np.eye(FRONT_N) * 1e-6
    try:
        precision = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return []
    pairs = []
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            pairs.append((i + 1, j + 1, float(precision[i, j])))
    pairs.sort(key=lambda t: (-abs(t[2]), t[0], t[1]))
    return pairs[:top_k]


def learn_next_rule(draws: np.ndarray) -> dict:
    """
    Pravilo next iz grupe 4:
    skor(y) = suma |corr|/MI sa brojevima iz poslednjeg kola + frekvencija.
    """
    mats = pearson_spearman_matrices(draws)
    pearson = mats["pearson"]
    mi = mutual_information_matrix(draws)

    last = [int(x) for x in draws[-1].tolist()]
    freq = Counter(draws.reshape(-1).tolist())
    max_f = max(freq.values()) if freq else 1

    number_score = {}
    for y in range(1, FRONT_N + 1):
        s = 0.0
        for x in last:
            if x == y:
                continue
            s += abs(float(pearson[x - 1, y - 1])) + float(mi[x - 1, y - 1])
        number_score[y] = s + 0.2 * (freq.get(y, 0) / max_f)

    return {
        "number_score": number_score,
        "last_draw": last,
        "target_sum": float(draws.sum(axis=1).mean()),
    }


def _combo_fit(combo: list[int], rule: dict) -> float:
    score = sum(rule["number_score"][x] for x in combo)
    score -= 0.015 * abs(sum(combo) - rule["target_sum"])
    return score


def predict_next_from_rule(draws: np.ndarray, rule: dict | None = None) -> list[int]:
    if rule is None:
        rule = learn_next_rule(draws)
    ranked = sorted(rule["number_score"], key=lambda n: (-rule["number_score"][n], n))
    best = None
    best_fit = -1e18
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        base = sorted(ranked[start : start + FRONT_SELECT])
        for repl in ranked[:28]:
            cand = sorted(set(base[1:] + [repl]))
            if len(cand) != FRONT_SELECT:
                continue
            fit = _combo_fit(cand, rule)
            if fit > best_fit:
                best_fit = fit
                best = cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_grupa4(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | 7/39 | grupa4")
    print()

    mats = pearson_spearman_matrices(draws)
    print("=== top |Pearson| pairs ===")
    print(top_corr_pairs(mats["pearson"]))
    print()

    print("=== top |Spearman| pairs ===")
    print(top_corr_pairs(mats["spearman"]))
    print()

    print("=== top mutual information pairs ===")
    print(mutual_information_pairs(draws))
    print()

    print("=== partial-corr proxy (precision) top ===")
    print(partial_corr_proxy(draws))
    print()

    print("=== pravilo → next (grupa 4) ===")
    rule = learn_next_rule(draws)
    combo = predict_next_from_rule(draws, rule)
    print("rule:", {"last_draw": rule["last_draw"], "target_sum": round(rule["target_sum"], 2)})
    print("next:", combo)


if __name__ == "__main__":
    run_grupa4()


"""
4. Korelacija / zavisnost
Pearson, Spearman, Kendall, partial correlation, distance correlation,
Hoeffding D, MIC/MINE, maximal correlation, mutual information,
conditional MI, normalized MI, pointwise MI (PMI), HSIC,
copula (Gaussian, Clayton, Gumbel, Frank, t-copula), rank correlation matrices
"""



"""
CSV: loto7_4648_k55.csv
Kola: 4648 | seed=39 | 7/39 | grupa4

=== top |Pearson| pairs ===
[(12, 22, -0.06051897871926165), (21, 33, -0.058493677002582206), (8, 39, -0.05785027147713106), (18, 27, -0.05730895511690402), (3, 39, -0.056840949682804176), (4, 30, -0.05628505472506303), (9, 39, -0.05583804783930336), (2, 26, -0.05569647257770474), (32, 34, -0.05536784762625945), (24, 37, -0.05531370380468911), (10, 11, -0.05484719134550472), (14, 18, -0.05482289198663594), (1, 36,-0.05402052235721309), (1, 27, -0.05342993716899333), (6, 24, -0.05331070285036583)]

=== top |Spearman| pairs ===
[(17, 20, 0.4899196285287024), (20, 36, 0.4858018431146071), (12, 17, 0.47612086328307046), (12, 39, 0.4737047165101309), (12, 20, 0.47360852763734007), (20, 24, 0.4728719210411769), (12, 21, 0.47054306809025354), (17, 21, 0.4702101373568475), (2, 36, 0.4697081367240163), (20, 27, 0.4659297841434385), (3, 17, 0.4654309323248813), (20, 21, 0.465091988143744), (12, 36, 0.4646446955447712), (16, 20, 0.4642154407794053), (2, 17, 0.4641434419322453)]

=== top mutual information pairs ===
[(12, 22, 0.002818828996207964), (21, 33, 0.0026236435452093674), (8, 39, 0.002549942892603024), (18, 27, 0.0025278101434903708),(3, 39, 0.002472771775508441), (4, 30, 0.0024371449489862546), (9, 39, 0.0023811463852808287), (2, 26, 0.002368728607009575), (32, 34, 0.0023337994412381893), (24, 37, 0.0023336579207556725), (14, 18, 0.002302260648427484), (10, 11, 0.002292632962064242), (1,36, 0.002242284054845471), (1, 27, 0.002191131834661913), (6, 24, 0.0021691805021419753)]

=== partial-corr proxy (precision) top ===
[(12, 22, 25641.08022495743), (18, 27, 25641.059257566383), (8, 39, 25641.058302497713), (3, 39, 25641.0557999801), (21, 33, 25641.052449941424), (2, 26, 25641.050816990086), (24, 37, 25641.047792373673), (9, 39, 25641.046745122974), (4, 30, 25641.044348907402), (32, 34, 25641.044279407255)]

=== pravilo → next (grupa 4) ===
rule: {'last_draw': [3, 7, 12, 13, 18, 24, 29], 'target_sum': 140.43}
next: [5, 6, 16, 25, 26, 31, 32]
"""
