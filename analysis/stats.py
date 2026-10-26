"""Cluster-bootstrap metrics, calibration fits, reliability bins, BH, shrinkage."""
import numpy as np
import pandas as pd
from scipy import stats as sps

B_DEFAULT = 2000
B_ROW = 1000  # for row-level refits (AUC, logistic slope/intercept)
EPS = 1e-6    # log-loss clip
MIN_EV = 50   # events and non-events per reported cell
MIN_BIN_EV = 30


def rng(seed=20261022):
    return np.random.default_rng(seed)


def _boot_counts(C, B, r):
    return r.multinomial(C, np.full(C, 1.0 / C), size=B)


def cluster_sums(d: pd.DataFrame, cluster="gp_id", extra=()):
    cols = {"n": ("y", "size"), "sy": ("y", "sum"), "sp": ("p", "sum"), "sse": ("se", "sum")}
    for c in extra:
        cols["s_" + c] = (c, "sum")
    return d.assign(se=(d["p"] - d["y"]) ** 2).groupby(cluster).agg(**cols)


def bias_skill(d: pd.DataFrame, B=B_DEFAULT, seed=1, cluster="gp_id"):
    """Bias (mean p - rate) and Brier skill vs in-cell base rate, cluster bootstrap."""
    n, ev = len(d), int(d["y"].sum())
    out = {"n": n, "events": ev, "nonevents": n - ev, "clusters": d[cluster].nunique(),
           "mean_p": d["p"].mean(), "rate": ev / n if n else np.nan}
    s = cluster_sums(d, cluster)
    W = _boot_counts(len(s), B, rng(seed))
    N, SY, SP, SSE = (W @ s[c].to_numpy() for c in ("n", "sy", "sp", "sse"))
    rate_b = SY / N
    bias_b = SP / N - rate_b
    with np.errstate(divide="ignore", invalid="ignore"):
        bss_b = 1 - (SSE / N) / (rate_b * (1 - rate_b))
    brier = ((d["p"] - d["y"]) ** 2).mean()
    base = out["rate"] * (1 - out["rate"])
    est = {"bias": out["mean_p"] - out["rate"], "bss": 1 - brier / base if base else np.nan}
    out.update({"brier": brier, "base_brier": base})
    for k, b in (("bias", bias_b), ("bss", bss_b)):
        b = b[np.isfinite(b)]
        se = b.std(ddof=1)
        out.update({k: est[k], f"{k}_lo": np.quantile(b, .025), f"{k}_hi": np.quantile(b, .975),
                    f"{k}_se": se, f"p_{k}": 2 * sps.norm.sf(abs(est[k]) / se) if se > 0 else np.nan})
    if s.index.nunique() == n:  # one row per cluster: Wilson on the realized rate
        lo, hi = wilson(ev, n)
        out.update({"rate_wilson_lo": lo, "rate_wilson_hi": hi})
    return out


def wilson(k, n, z=1.96):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    m = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def paired_diff(d: pd.DataFrame, a: str, b: str, B=B_DEFAULT, seed=2, cluster="gp_id"):
    """Mean of (d[a] - d[b]) with cluster-bootstrap CI."""
    x = (d[a] - d[b]).groupby(d[cluster]).agg(["sum", "size"])
    W = _boot_counts(len(x), B, rng(seed))
    bt = (W @ x["sum"].to_numpy()) / (W @ x["size"].to_numpy())
    est = x["sum"].sum() / x["size"].sum()
    se = bt.std(ddof=1)
    return {"n": int(x["size"].sum()), "clusters": len(x), "diff": est,
            "lo": np.quantile(bt, .025), "hi": np.quantile(bt, .975),
            "p": 2 * sps.norm.sf(abs(est) / se) if se > 0 else np.nan}


# ---------------------------------------------------------------------------
#  Row-level metrics
# ---------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def log_loss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = sps.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib_fit(p, y, iters=25):
    """(intercept with logit(p) offset, slope from y ~ a + b*logit(p))."""
    x, y = logit(np.asarray(p, float)), np.asarray(y, float)
    a = 0.0  # calibration-in-the-large
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(a + x)))
        g, h = (y - mu).sum(), (mu * (1 - mu)).sum()
        a += g / h
        if abs(g / h) < 1e-10:
            break
    beta = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        w = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return a, beta[1]


def row_metrics(d: pd.DataFrame, B=B_ROW, seed=3, cluster="gp_id"):
    """AUC, calibration intercept & slope with cluster bootstrap CIs, plus log loss."""
    d = d.sort_values(cluster)
    p, y = d["p"].to_numpy(), d["y"].to_numpy()
    codes, starts, lens = np.unique(d[cluster].to_numpy(), return_index=True, return_counts=True)
    est = {"auc": auc(p, y), "log_loss": log_loss(p, y)}
    est["cal_int"], est["cal_slope"] = calib_fit(p, y)
    r = rng(seed)
    bs = {"auc": [], "cal_int": [], "cal_slope": []}
    C = len(codes)
    for _ in range(B):
        pick = r.integers(0, C, C)
        L = lens[pick]
        idx = np.repeat(starts[pick] - np.concatenate([[0], np.cumsum(L)[:-1]]), L) + np.arange(L.sum())
        pb, yb = p[idx], y[idx]
        bs["auc"].append(auc(pb, yb))
        a, b = calib_fit(pb, yb)
        bs["cal_int"].append(a)
        bs["cal_slope"].append(b)
    for k, v in bs.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        est[k + "_lo"], est[k + "_hi"] = np.quantile(v, .025), np.quantile(v, .975)
    return est


def reliability(d: pd.DataFrame, max_bins=10):
    """Equal-count bins; largest k <= max_bins with >= MIN_BIN_EV events and non-events per bin."""
    d = d.sort_values("p").reset_index(drop=True)
    for k in range(max_bins, 1, -1):
        bins = np.array_split(np.arange(len(d)), k)
        ok = all(MIN_BIN_EV <= d["y"].iloc[b].sum() <= len(b) - MIN_BIN_EV for b in bins)
        if ok:
            rows = []
            for b in bins:
                s = d.iloc[b]
                ev = int(s["y"].sum())
                lo, hi = wilson(ev, len(s))
                rows.append({"n": len(s), "events": ev, "mean_p": s["p"].mean(),
                             "rate": ev / len(s), "lo": lo, "hi": hi})
            return pd.DataFrame(rows)
    return pd.DataFrame()


def ece(rel: pd.DataFrame):
    if rel.empty:
        return np.nan
    return float((rel["n"] * (rel["mean_p"] - rel["rate"]).abs()).sum() / rel["n"].sum())


def bh(p: pd.Series, q=0.05):
    """Benjamini-Hochberg: returns (q-values, reject) aligned to p (NaN ignored)."""
    s = p.dropna().sort_values()
    m = len(s)
    qv = (s * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    qv = qv.reindex(p.index)
    return qv, qv <= q


def eb_shrink(est: np.ndarray, se: np.ndarray):
    """DerSimonian-Laird random-effects shrinkage. Returns (shrunk, tau2, grand_mean)."""
    w = 1 / se ** 2
    mu = (w * est).sum() / w.sum()
    Q = (w * (est - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (len(est) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se ** 2 + tau2)
    mu_r = (wr * est).sum() / wr.sum()
    shrink = tau2 / (tau2 + se ** 2) if tau2 > 0 else np.zeros_like(se)
    return mu_r + shrink * (est - mu_r), tau2, mu_r


def _self_check():
    assert abs(auc([.1, .9], [0, 1]) - 1) < 1e-12 and abs(auc([.9, .1], [0, 1])) < 1e-12
    r = rng(0)
    p = r.uniform(.1, .9, 20000)
    y = (r.uniform(size=p.size) < p).astype(int)
    a, b = calib_fit(p, y)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05, (a, b)
    d = pd.DataFrame({"gp_id": np.arange(p.size), "p": p, "y": y})
    o = bias_skill(d, B=200)
    assert o["bias_lo"] < 0 < o["bias_hi"] and o["bss"] > 0
    qv, rej = bh(pd.Series([0.001, 0.02, 0.04, 0.5]))
    assert rej.tolist() == [True, True, False, False]
    sh, tau2, mu = eb_shrink(np.array([0.1, -0.1, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert tau2 == 0 and np.allclose(sh, 0)


if __name__ == "__main__":
    _self_check()
    print("stats self-check ok")
