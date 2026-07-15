"""A zoo of in-game win-probability models, all fit on ~11k historical games.

Model SELECTION happens here, on thousands of games -- NOT on the 16 games of
market ticks. Picking the best of 40 models on 16 games would select the
luckiest, not the best (per-game SD is 0.41; see docs/research-log.md).

Families:
  A  analytic          the frozen physics model (baseline)
  B  analytic_recal    same formula, constants refit by search on train
  C  empirical_*       beta-smoothed state lookup; prior + state-key sweeps
  D  negbin            NEW MATH: runs ~ Negative Binomial, exact convolution.
                       The analytic model assumes final run differential is
                       NORMAL. Real run scoring is discrete, skewed and
                       overdispersed (var > mean). This models each team's
                       remaining runs as NB and convolves them exactly.
  E  logistic          logistic regression on engineered features
  F  gbm_*             gradient-boosted trees, stacked on the analytic output
  G  ensemble_*        log-odds averages + isotonic recalibration

Every model exposes fit(states, y) / predict(states) -> probabilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import zoo_data as Z
from polybot.winprob import _RE24, REGULATION_INNINGS

_RE = np.asarray(_RE24)
FIT_SAMPLE = 150_000        # grid searches only need enough states to rank params


def _subsample(states, y, n=FIT_SAMPLE, seed=0):
    """Scalar grid searches don't need 700k states; this keeps fits to seconds."""
    if len(y) <= n:
        return states, y
    idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
    return states[idx], y[idx]


def _warn_boundary(model, grid) -> None:
    """An optimum on a grid edge means the grid clipped it -- a silent bug."""
    for param, values in grid.items():
        v = getattr(model, param)
        if np.isclose(v, values.min()) or np.isclose(v, values.max()):
            print(f"    !! {model.__class__.__name__}.{param}={v:g} hit a grid "
                  f"BOUNDARY [{values.min():g}, {values.max():g}] -- widen it",
                  flush=True)


class Model:
    name = "base"

    def fit(self, states, y, analytic=None):
        return self

    def predict(self, states, analytic=None) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- A: analytic
class Analytic(Model):
    name = "analytic"

    def predict(self, states, analytic=None):
        return Z.analytic_probs(states) if analytic is None else analytic


# ----------------------------------------------- D: negative-binomial run model
def _nb_pmf(mu: np.ndarray, k: float, n_max: int) -> np.ndarray:
    """Negative-binomial pmf over 0..n_max, vectorized over rows.

    Parameterized by mean mu and dispersion k: var = mu + mu^2/k.
    k -> inf recovers Poisson. Baseball innings are overdispersed, so finite k
    matters -- this is the whole point of the model.
    """
    mu = np.maximum(mu, 1e-9)
    p = k / (k + mu)                       # P(success)
    out = np.empty((len(mu), n_max + 1))
    out[:, 0] = p ** k
    q = 1.0 - p
    for x in range(1, n_max + 1):
        out[:, x] = out[:, x - 1] * ((x + k - 1.0) / x) * q
    return out


class NegBin(Model):
    """Runs remaining ~ NegBinomial per team; P(home win) by exact convolution."""
    name = "negbin"
    N_MAX = 30

    def __init__(self, lam=0.48, k=4.0, home_edge=0.10, tie_home=0.52):
        self.lam, self.k, self.home_edge, self.tie_home = lam, k, home_edge, tie_home

    def _means(self, states):
        inning = states[:, Z.INNING].astype(float)
        is_top = states[:, Z.IS_TOP].astype(bool)
        outs = np.clip(states[:, Z.OUTS], 0, 2).astype(int)
        bases = (states[:, Z.B1] | (states[:, Z.B2] << 1)
                 | (states[:, Z.B3] << 2)).astype(int)
        after = np.maximum(0.0, REGULATION_INNINGS - inning)   # full innings left
        re_now = _RE[outs, bases]                              # runs left THIS half

        # batting team gets the exact base-out run expectancy for the current half;
        # every other half-inning gets the league rate lam.
        mu_away = np.where(is_top, re_now + self.lam * after, self.lam * after)
        mu_home = np.where(is_top, self.lam * (after + 1.0), re_now + self.lam * after)
        total = mu_away + mu_home
        mu_home = mu_home + self.home_edge * (total / (2.0 * REGULATION_INNINGS * self.lam))
        return mu_away, np.maximum(mu_home, 1e-9)

    def predict(self, states, analytic=None):
        mu_a, mu_h = self._means(states)
        n = self.N_MAX
        pa = _nb_pmf(mu_a, self.k, n)
        ph = _nb_pmf(mu_h, self.k, n)
        cdf_h = np.cumsum(ph, axis=1)

        # home must out-score away by more than d = away_score - home_score
        d = (states[:, Z.AS_].astype(int) - states[:, Z.HS].astype(int))
        win = np.zeros(len(states))
        tie = np.zeros(len(states))
        idx = np.arange(len(states))
        for a in range(n + 1):
            need = d + a                    # home needs H > need  (tie at H == need)
            gt = np.where(need < 0, 1.0,
                          np.where(need >= n, 0.0,
                                   1.0 - cdf_h[idx, np.clip(need, 0, n)]))
            eq = np.where((need < 0) | (need > n), 0.0,
                          ph[idx, np.clip(need, 0, n)])
            win += pa[:, a] * gt
            tie += pa[:, a] * eq
        p = win + self.tie_home * tie

        # walk-off: home ahead in the bottom of the 9th or later -> already won
        walkoff = ((states[:, Z.INNING] >= REGULATION_INNINGS)
                   & (states[:, Z.IS_TOP] == 0)
                   & (states[:, Z.HS] > states[:, Z.AS_]))
        p = np.where(walkoff, 1.0, p)
        return np.clip(p, 0.001, 0.999)

    def fit(self, states, y, analytic=None):
        """Coordinate search on train log-loss. No scipy; a coarse grid is plenty.

        Grids are deliberately wider than the plausible range: an optimum landing
        on a boundary means the grid clipped it, which is a silent bug.
        """
        grid = {
            "lam": np.arange(0.28, 0.66, 0.02),
            "k": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 15.0, 30.0]),
            "home_edge": np.arange(-0.10, 0.42, 0.04),
            "tie_home": np.arange(0.44, 0.66, 0.02),
        }
        s, yy = _subsample(states, y)
        best = Z.log_loss(self.predict(s), yy)
        for _ in range(3):                       # a few coordinate passes
            for param, values in grid.items():
                cur = getattr(self, param)
                for v in values:
                    setattr(self, param, float(v))
                    ll = Z.log_loss(self.predict(s), yy)
                    if ll < best:
                        best, cur = ll, float(v)
                setattr(self, param, cur)
        _warn_boundary(self, grid)
        self.name = (f"negbin(lam={self.lam:.2f},k={self.k:g},"
                     f"he={self.home_edge:.2f},tie={self.tie_home:.2f})")
        return self


# ------------------------------------------------- B: recalibrated analytic
class AnalyticRecal(Model):
    """The analytic normal model with its three constants refit to outcomes."""
    name = "analytic_recal"

    def __init__(self, sd=1.06, home_edge=0.12, tie_home=0.52):
        self.sd, self.home_edge, self.tie_home = sd, home_edge, tie_home

    def predict(self, states, analytic=None):
        inning = states[:, Z.INNING].astype(float)
        is_top = states[:, Z.IS_TOP].astype(bool)
        outs = np.clip(states[:, Z.OUTS], 0, 2).astype(int)
        bases = (states[:, Z.B1] | (states[:, Z.B2] << 1)
                 | (states[:, Z.B3] << 2)).astype(int)
        after = np.maximum(0.0, REGULATION_INNINGS - inning)
        away_rem = np.where(is_top, after + 1.0 - states[:, Z.OUTS] / 3.0, after)
        home_rem = np.where(is_top, after + 1.0,
                            after + 1.0 - states[:, Z.OUTS] / 3.0)
        total = away_rem + home_rem

        diff = (states[:, Z.HS] - states[:, Z.AS_]).astype(float)
        re_adj = _RE[outs, bases] - _RE[0, 0]
        diff = diff + np.where(is_top, -re_adj, re_adj)
        diff = diff + self.home_edge * (total / (2.0 * REGULATION_INNINGS))
        sd = self.sd * np.sqrt(np.maximum(total, 0.25))

        from math import erf, sqrt
        phi = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0))))
        ahead = 1.0 - phi((0.5 - diff) / sd)
        tie = phi((0.5 - diff) / sd) - phi((-0.5 - diff) / sd)
        p = ahead + self.tie_home * tie
        walkoff = ((states[:, Z.INNING] >= REGULATION_INNINGS)
                   & (states[:, Z.IS_TOP] == 0) & (states[:, Z.HS] > states[:, Z.AS_]))
        return np.clip(np.where(walkoff, 1.0, p), 0.001, 0.999)

    def fit(self, states, y, analytic=None):
        grid = {"sd": np.arange(0.60, 1.55, 0.05),
                "home_edge": np.arange(-0.10, 0.42, 0.04),
                "tie_home": np.arange(0.44, 0.66, 0.02)}
        s, yy = _subsample(states, y)
        best = Z.log_loss(self.predict(s), yy)
        for _ in range(3):
            for param, values in grid.items():
                cur = getattr(self, param)
                for v in values:
                    setattr(self, param, float(v))
                    ll = Z.log_loss(self.predict(s), yy)
                    if ll < best:
                        best, cur = ll, float(v)
                setattr(self, param, cur)
        _warn_boundary(self, grid)
        self.name = (f"analytic_recal(sd={self.sd:.2f},"
                     f"he={self.home_edge:.2f},tie={self.tie_home:.2f})")
        return self


# ------------------------------------------------------- C: empirical lookup
def _key(states, clip_diff=6, cap_inning=10, use_bases=True):
    inning = np.clip(states[:, Z.INNING], 1, cap_inning).astype(np.int64)
    diff = np.clip(states[:, Z.HS].astype(np.int64) - states[:, Z.AS_], -clip_diff,
                   clip_diff) + clip_diff
    outs = np.clip(states[:, Z.OUTS], 0, 2).astype(np.int64)
    top = states[:, Z.IS_TOP].astype(np.int64)
    bases = ((states[:, Z.B1] | (states[:, Z.B2] << 1)
              | (states[:, Z.B3] << 2)).astype(np.int64) if use_bases
             else np.zeros(len(states), dtype=np.int64))
    nb = 8 if use_bases else 1
    return (((inning * 2 + top) * 3 + outs) * (2 * clip_diff + 1) + diff) * nb + bases


class Empirical(Model):
    """Beta-smoothed empirical state lookup, shrunk toward the analytic prior."""

    def __init__(self, prior=30.0, clip_diff=6, cap_inning=10, use_bases=True):
        self.prior, self.clip_diff = prior, clip_diff
        self.cap_inning, self.use_bases = cap_inning, use_bases
        self.name = (f"empirical(prior={prior:g},clip={clip_diff},"
                     f"inn={cap_inning},bases={int(use_bases)})")

    def _k(self, states):
        return _key(states, self.clip_diff, self.cap_inning, self.use_bases)

    def fit(self, states, y, analytic=None):
        k = self._k(states)
        size = int(k.max()) + 1
        self.count = np.bincount(k, minlength=size).astype(float)
        self.wins = np.bincount(k, weights=y.astype(float), minlength=size)
        return self

    def predict(self, states, analytic=None):
        a = Z.analytic_probs(states) if analytic is None else analytic
        k = self._k(states)
        k = np.clip(k, 0, len(self.count) - 1)
        c, w = self.count[k], self.wins[k]
        return np.clip((w + self.prior * a) / (c + self.prior), 0.001, 0.999)


# ------------------------------------------------------------ E/F: ML models
class Sklearn(Model):
    def __init__(self, est, name):
        self.est, self.name = est, name

    def fit(self, states, y, analytic=None):
        self.est.fit(Z.features(states, analytic), y)
        return self

    def predict(self, states, analytic=None):
        return np.clip(
            self.est.predict_proba(Z.features(states, analytic))[:, 1], 0.001, 0.999)


# ------------------------------------------------------------- G: ensembles
class LogOddsEnsemble(Model):
    """Average the members in LOG-ODDS space (not probability space).

    Averaging probabilities of well-calibrated models makes the result
    UNDER-confident; averaging log-odds preserves sharpness.
    """

    def __init__(self, members, name):
        self.members, self.name = members, name

    def fit(self, states, y, analytic=None):
        for m in self.members:
            m.fit(states, y, analytic)
        return self

    def predict(self, states, analytic=None):
        ps = [np.clip(m.predict(states, analytic), 1e-6, 1 - 1e-6) for m in self.members]
        lo = np.mean([np.log(p / (1 - p)) for p in ps], axis=0)
        return np.clip(1.0 / (1.0 + np.exp(-lo)), 0.001, 0.999)


class Isotonic(Model):
    """Monotonic recalibration of any base model.

    The base is fit on 80% of TRAIN and the calibration curve on the held-out
    20% -- calibrating on the same rows the base was fit on would just learn the
    base's in-sample overfit and report a falsely good ECE.
    """

    def __init__(self, base):
        self.base = base
        self.name = f"isotonic+{base.name.split('(')[0]}"
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)

    def fit(self, states, y, analytic=None):
        n = len(y)
        cut = int(0.8 * n)
        idx = np.random.default_rng(1).permutation(n)
        a, b = idx[:cut], idx[cut:]
        af = None if analytic is None else analytic[a]
        bf = None if analytic is None else analytic[b]
        self.base.fit(states[a], y[a], af)
        self.iso.fit(self.base.predict(states[b], bf), y[b])
        return self

    def predict(self, states, analytic=None):
        return np.clip(self.iso.predict(self.base.predict(states, analytic)),
                       0.001, 0.999)


def build_zoo() -> list[Model]:
    """Every candidate. Fit on TRAIN, compared on VALID."""
    zoo: list[Model] = [Analytic(), AnalyticRecal(), NegBin()]

    # C: empirical sweeps -- smoothing strength and state-key resolution
    for prior in (5, 15, 30, 60, 150, 400):
        zoo.append(Empirical(prior=prior))
    for clip in (3, 4, 8):
        zoo.append(Empirical(prior=30, clip_diff=clip))
    zoo.append(Empirical(prior=30, use_bases=False))
    zoo.append(Empirical(prior=30, cap_inning=12))
    zoo.append(Empirical(prior=100, clip_diff=8, cap_inning=12))

    # E: logistic on engineered features (stacked on the analytic output)
    for C in (0.05, 1.0):
        zoo.append(Sklearn(
            LogisticRegression(C=C, max_iter=2000), f"logistic(C={C})"))

    # F: gradient-boosted trees, several capacities
    for leaves, lr, it in ((15, 0.10, 200), (31, 0.06, 400), (63, 0.06, 400),
                           (31, 0.03, 800)):
        zoo.append(Sklearn(
            HistGradientBoostingClassifier(
                max_leaf_nodes=leaves, learning_rate=lr, max_iter=it,
                early_stopping=True, validation_fraction=0.1, random_state=0),
            f"gbm(leaves={leaves},lr={lr},it={it})"))

    # G: calibration + ensembles across DIFFERENT families (uncorrelated errors
    # are what makes an ensemble worth more than its members)
    zoo.append(Isotonic(NegBin()))
    zoo.append(Isotonic(Empirical(prior=30)))
    zoo.append(Isotonic(Sklearn(
        HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06,
                                       max_iter=400, early_stopping=True,
                                       validation_fraction=0.1, random_state=0),
        "gbm")))
    zoo.append(LogOddsEnsemble(
        [NegBin(), Empirical(prior=30)], "ens(negbin+empirical)"))
    zoo.append(LogOddsEnsemble(
        [NegBin(), Empirical(prior=30),
         Sklearn(HistGradientBoostingClassifier(
             max_leaf_nodes=31, learning_rate=0.06, max_iter=400,
             early_stopping=True, validation_fraction=0.1, random_state=0), "gbm")],
        "ens(negbin+empirical+gbm)"))
    zoo.append(Isotonic(LogOddsEnsemble(
        [NegBin(), Empirical(prior=30),
         Sklearn(HistGradientBoostingClassifier(
             max_leaf_nodes=31, learning_rate=0.06, max_iter=400,
             early_stopping=True, validation_fraction=0.1, random_state=0), "gbm")],
        "ens3")))
    return zoo
