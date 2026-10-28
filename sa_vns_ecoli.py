#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA+VNS hybrid metaheuristic for E. coli protein localization site prediction
=============================================================================
Reproducibility code for the manuscript
  "Using a Hybrid Metaheuristic Model to Predict Protein Localization Sites"
  Eslam E. M. Attia, Osman Ali Sadek Ibrahim, Abdelmgeid A. Ali (Scientific Reports, revision)

Method (manuscript Section 3)
-----------------------------
  * Linear scoring model  L = X W,  X = [Z-scored features, 1]  ->  W in R^{(7+1) x 8}
    (7 feature weights + 1 bias row, 8 localization classes); prediction = argmax_j L_ij.
  * Fitness (to be maximised on the TRAINING data only):
        f(W) = macro-F1(y, argmax(XW)) - lambda * ||W||_F^2 ,  lambda = 1e-4
  * Initial solution: multinomial logistic regression fitted on the training data,
    plus Gaussian noise N(0, 0.2^2)  (warm start).
  * Phase 1 - Simulated Annealing (Algorithm 1): Gaussian perturbation N(0, sigma_SA^2) of all
    weights, Metropolis acceptance exp(delta / T), geometric cooling T <- alpha * T.
    T0 is on the scale of fitness differences (macro-F1 changes of about 0.005-0.05).
  * Phase 2 - Variable Neighborhood Search (Algorithm 2): neighborhood N_k perturbs k randomly
    chosen rows of W (sampling without replacement) with N(0, sigma_VNS^2); on improvement the
    incumbent is updated and k is reset to 1, otherwise k <- k + 1 (k = 1..k_max).
  * Sequential high-level relay hybrid: best SA solution initialises VNS.

Experimental protocol (manuscript Section 4)
--------------------------------------------
  * 10 repeated stratified 80/20 train-test splits (StratifiedShuffleSplit, random_state=42).
  * Z-score statistics, warm start, hyperparameter tuning and all optimisation use the
    training part of each split only; the test part is used once for the final evaluation.
  * SA hyperparameters (T0, alpha) are selected per split by grid search with inner
    3-fold stratified cross-validation on the training data.
  * The same splits are used for SA, VNS, SA+VNS and 13 benchmark classifiers.
  * Statistics over the 10 paired splits: corrected resampled t-test (Nadeau & Bengio, 2003)
    with Holm correction, Wilcoxon signed-rank test, paired Cohen's d (d_z).
  * Robustness: the whole protocol is repeated with several independent seeds for the
    stochastic metaheuristics (default 5) and the variability of the mean is reported.

Usage
-----
  pip install -r requirements.txt
  python sa_vns_ecoli.py                 # full experiment (about 5-10 minutes on a laptop CPU)
  python sa_vns_ecoli.py --quick         # smoke test (about 30 seconds)
  python sa_vns_ecoli.py --data ecoli.data --out results

All tables (CSV), figures (PNG) and a summary (REPORT.md) are written to ./results/.
"""

import argparse
import io
import json
import os
import platform
import sys
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

FEATURES = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
UCI_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
    "https://archive.ics.uci.edu/static/public/39/ecoli.zip",
]


# =============================================================================
# Configuration (Table 6 of the manuscript)
# =============================================================================
@dataclass
class Config:
    seed: int = 42                  # random_state of the splits and of all benchmark models
    n_splits: int = 10              # repeated stratified splits
    test_size: float = 0.20         # 80/20
    n_repeats: int = 5              # independent repetitions of the stochastic search
    # --- solution representation / fitness ---
    l2_lambda: float = 1e-4         # L2 penalty in the fitness
    init_noise: float = 0.2         # std of the noise added to the warm start
    # --- SA (Algorithm 1) ---
    sa_T0: float = 0.01             # used only with --no-tune
    sa_alpha: float = 0.995         # used only with --no-tune
    sa_sigma: float = 0.05          # Gaussian step size of SA (all weights)
    sa_iters_hybrid: int = 1000     # SA phase inside the hybrid
    sa_iters_alone: int = 1500      # SA-only variant (ablation)
    # --- VNS (Algorithm 2) ---
    vns_kmax: int = 5               # largest neighborhood (number of rows of W perturbed)
    vns_sigma: float = 0.15         # Gaussian step size
    vns_iters_hybrid: int = 500     # VNS phase inside the hybrid
    vns_iters_alone: int = 600      # VNS-only variant (ablation)
    # --- hyperparameter tuning (inner CV on training data) ---
    tune: bool = True
    tune_T0_grid: list = field(default_factory=lambda: [0.005, 0.01, 0.02])
    tune_alpha_grid: list = field(default_factory=lambda: [0.990, 0.995, 0.998])
    tune_inner_folds: int = 3
    # --- deep benchmark models ---
    dl_epochs: int = 100
    dl_batch: int = 32
    dl_lr: float = 1e-3
    dl_patience: int = 10
    dl_val_frac: float = 0.10
    out_dir: str = "results"


# =============================================================================
# Data
# =============================================================================
def load_ecoli(path=None):
    """UCI E. coli: sequence name, 7 numeric attributes, localization class (336 rows)."""
    text = None
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        for url in UCI_URLS:
            try:
                raw = urllib.request.urlopen(url, timeout=30).read()
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        name = [n for n in z.namelist() if n.endswith("ecoli.data")][0]
                        raw = z.read(name)
                text = raw.decode("utf-8")
                with open("ecoli.data", "w", encoding="utf-8") as f:
                    f.write(text)
                break
            except Exception as e:
                print(f"[warn] could not download {url}: {e}")
    if text is None:
        sys.exit("Dataset not found. Download ecoli.data from "
                 "https://archive.ics.uci.edu/dataset/39/ecoli and run with --data ecoli.data")
    # The file has no header line -> header=None keeps all 336 proteins.
    return pd.read_csv(io.StringIO(text), sep=r"\s+", header=None,
                       names=["name"] + FEATURES + ["class"])


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


# =============================================================================
# Linear scoring model and fitness (Section 3.3)
# =============================================================================
def predict_linear(W, X):
    return np.argmax(X @ W, axis=1)


def macro_f1_fast(y_true, y_pred, n_classes):
    """Identical to sklearn f1_score(average='macro', zero_division=0) but ~50x faster."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    present = (cm.sum(axis=0) + cm.sum(axis=1)) > 0       # labels in y_true or y_pred
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[present].mean())


def make_fitness(X, y, n_classes, lam):
    def fitness(W):
        return macro_f1_fast(y, predict_linear(W, X), n_classes) - lam * float(np.sum(W * W))
    return fitness


def warm_start(X_feat, y, n_classes, noise, rng, seed):
    """Multinomial logistic regression on the training data + Gaussian noise."""
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit(X_feat, y)
    W = np.zeros((X_feat.shape[1] + 1, n_classes))
    coef = np.zeros((n_classes, X_feat.shape[1]))
    icpt = np.zeros(n_classes)
    coef[clf.classes_] = clf.coef_            # classes absent from a training set keep 0
    icpt[clf.classes_] = clf.intercept_
    icpt[np.setdiff1d(np.arange(n_classes), clf.classes_)] = -10.0
    W[:-1, :] = coef.T
    W[-1, :] = icpt
    return W + rng.normal(0.0, noise, W.shape)


# =============================================================================
# Algorithm 1: Simulated Annealing (global exploration)
# =============================================================================
def simulated_annealing(fitness, W0, T0, alpha, sigma, iters, rng):
    current = W0.copy()
    current_f = fitness(current)
    best, best_f = current.copy(), current_f
    T = T0
    curve = [best_f]
    for _ in range(iters):
        candidate = current + rng.normal(0.0, sigma, current.shape)
        cand_f = fitness(candidate)
        delta = cand_f - current_f
        if delta > 0 or rng.random() < np.exp(delta / T):
            current, current_f = candidate, cand_f
            if current_f > best_f:
                best, best_f = current.copy(), current_f
        T *= alpha
        curve.append(best_f)
    return best, best_f, curve


# =============================================================================
# Algorithm 2: Variable Neighborhood Search (local refinement)
# =============================================================================
def variable_neighborhood_search(fitness, W0, k_max, sigma, iters, rng):
    best = W0.copy()
    best_f = fitness(best)
    n_rows, n_cols = best.shape
    k_max = min(k_max, n_rows)
    curve = [best_f]
    for _ in range(iters):
        k = 1
        while k <= k_max:
            rows = rng.choice(n_rows, size=k, replace=False)      # sampling without replacement
            candidate = best.copy()
            candidate[rows] += rng.normal(0.0, sigma, (k, n_cols))
            cand_f = fitness(candidate)
            if cand_f > best_f:
                best, best_f = candidate, cand_f
                k = 1                                             # new incumbent -> k = 1
            else:
                k += 1                                            # next neighborhood
        curve.append(best_f)
    return best, best_f, curve


def run_variant(variant, Xtr_feat, ytr, n_classes, T0, alpha, cfg, rng, seed):
    """variant in {'SA', 'VNS', 'SA+VNS'}; all start from the same type of warm start."""
    X = add_bias(Xtr_feat)
    fit = make_fitness(X, ytr, n_classes, cfg.l2_lambda)
    W0 = warm_start(Xtr_feat, ytr, n_classes, cfg.init_noise, rng, seed)
    if variant == "SA":
        W, _, c = simulated_annealing(fit, W0, T0, alpha, cfg.sa_sigma, cfg.sa_iters_alone, rng)
        return W, {"sa": c}
    if variant == "VNS":
        W, _, c = variable_neighborhood_search(fit, W0, cfg.vns_kmax, cfg.vns_sigma, cfg.vns_iters_alone, rng)
        return W, {"vns": c}
    W_sa, _, c1 = simulated_annealing(fit, W0, T0, alpha, cfg.sa_sigma, cfg.sa_iters_hybrid, rng)
    W, _, c2 = variable_neighborhood_search(fit, W_sa, cfg.vns_kmax, cfg.vns_sigma, cfg.vns_iters_hybrid, rng)
    return W, {"sa": c1, "vns": c2}


def tune_sa(Xtr_raw, ytr, n_classes, cfg, seed):
    """Grid search over (T0, alpha) with inner stratified CV on the TRAINING data only."""
    skf = StratifiedKFold(n_splits=cfg.tune_inner_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(Xtr_raw, ytr))
    rows = []
    for T0 in cfg.tune_T0_grid:
        for alpha in cfg.tune_alpha_grid:
            scores = []
            for i, (a, b) in enumerate(folds):
                sc = StandardScaler().fit(Xtr_raw[a])
                Xa, Xb = sc.transform(Xtr_raw[a]), sc.transform(Xtr_raw[b])
                rng = np.random.default_rng(seed * 100 + i)
                W, _ = run_variant("SA+VNS", Xa, ytr[a], n_classes, T0, alpha, cfg, rng, seed)
                scores.append(macro_f1_fast(ytr[b], predict_linear(W, add_bias(Xb)), n_classes))
            rows.append({"T0": T0, "alpha": alpha, "inner_cv_macroF1": float(np.mean(scores))})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["inner_cv_macroF1", "T0", "alpha"], ascending=[False, True, True]).iloc[0]
    return float(best["T0"]), float(best["alpha"]), grid


# =============================================================================
# Benchmark models (Section 4.1.1)
# =============================================================================
def sklearn_models(seed):
    return {
        "LR": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed),
        "LDA": LinearDiscriminantAnalysis(solver="svd"),
        "KNN": KNeighborsClassifier(n_neighbors=5, p=2, weights="uniform"),
        "CART": DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=seed),
        "NB": GaussianNB(var_smoothing=1e-9),
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed),
        "L_SVM": SVC(C=1.0, kernel="linear", random_state=seed),
        "ETC": ExtraTreesClassifier(n_estimators=100, criterion="gini", random_state=seed),
        "RFC": RandomForestClassifier(n_estimators=100, criterion="gini", random_state=seed),
        "NN": MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                            learning_rate_init=0.001, max_iter=500, random_state=seed),
    }


def build_torch_models(n_features, n_classes, A_hat):
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * n_features, 64), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(64, n_classes))

        def forward(self, x):
            return self.head(self.conv(x.unsqueeze(1)))

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            _, (h, _) = self.rnn(x.unsqueeze(-1))
            return self.head(h[-1])

    class GCNN(nn.Module):
        """Each protein is a graph of 7 attribute nodes; edge weights are |Pearson r|
        between attributes estimated on the training data (symmetric normalisation)."""
        def __init__(self):
            super().__init__()
            self.register_buffer("A", torch.tensor(A_hat, dtype=torch.float32))
            self.register_buffer("node_id", torch.eye(n_features))
            self.g1 = nn.Linear(1 + n_features, 32)
            self.g2 = nn.Linear(32, 32)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * n_features, n_classes))

        def forward(self, x):
            h = torch.cat([x.unsqueeze(-1), self.node_id.expand(x.shape[0], -1, -1)], dim=-1)
            h = torch.relu(self.A @ self.g1(h))
            h = torch.relu(self.A @ self.g2(h))
            return self.head(h)

    return {"RNN": RNN, "CNN": CNN1D, "GCNN": GCNN}


def normalized_adjacency(X_train):
    C = np.abs(np.nan_to_num(np.corrcoef(X_train, rowvar=False)))
    np.fill_diagonal(C, 1.0)
    d = 1.0 / np.sqrt(C.sum(axis=1))
    return C * d[:, None] * d[None, :]


def train_torch(model, X_tr, y_tr, cfg, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(len(X_tr))
    n_val = max(1, int(round(cfg.dl_val_frac * len(X_tr))))
    v, f = perm[:n_val], perm[n_val:]
    Xf, yf = torch.tensor(X_tr[f], dtype=torch.float32), torch.tensor(y_tr[f], dtype=torch.long)
    Xv, yv = torch.tensor(X_tr[v], dtype=torch.float32), torch.tensor(y_tr[v], dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.dl_lr)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg.dl_epochs):
        model.train()
        order = torch.randperm(len(Xf), generator=g)
        for i in range(0, len(Xf), cfg.dl_batch):
            b = order[i:i + cfg.dl_batch]
            opt.zero_grad()
            loss_fn(model(Xf[b]), yf[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        if vl < best_loss - 1e-6:
            best_loss, wait = vl, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.dl_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def torch_predict(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


# =============================================================================
# Metrics and statistics
# =============================================================================
def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MacroF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected repeated train/test t-test (paired)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / n + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def cohens_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = np.std(d, ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


# =============================================================================
# Experiment
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to a local copy of ecoli.data")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--repeats", type=int, default=None, help="independent repetitions (default 5)")
    ap.add_argument("--no-tune", action="store_true", help="use fixed T0=0.01, alpha=0.995")
    ap.add_argument("--no-deep", action="store_true", help="skip CNN/RNN/GCNN")
    ap.add_argument("--quick", action="store_true", help="smoke test with small budgets")
    args = ap.parse_args()

    cfg = Config(out_dir=args.out, tune=not args.no_tune)
    if args.repeats is not None:
        cfg.n_repeats = args.repeats
    if args.quick:
        cfg.n_splits, cfg.n_repeats, cfg.dl_epochs = 2, 2, 10
        cfg.sa_iters_hybrid, cfg.sa_iters_alone, cfg.vns_iters_hybrid, cfg.vns_iters_alone = 100, 150, 50, 60
        cfg.tune_T0_grid, cfg.tune_alpha_grid = [0.01], [0.995]
    os.makedirs(cfg.out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # ---------------- data ----------------
    df = load_ecoli(args.data)
    X_raw = df[FEATURES].to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(df["class"].to_numpy())
    class_names = [str(c) for c in le.classes_]
    K, D = len(class_names), X_raw.shape[1]
    assert len(df) == 336, "Expected 336 proteins."
    with open(os.path.join(cfg.out_dir, "dataset_check.txt"), "w", encoding="utf-8") as f:
        f.write(f"proteins: {len(df)} | features: {D} | classes: {K}\n")
        f.write(f"label encoding: {dict(zip(class_names, range(K)))}\n")
        f.write(f"class counts: {df['class'].value_counts().to_dict()}\n")
        f.write(f"missing values: {int(df.isna().sum().sum())}\n")
        f.write(f"unique sequence identifiers: {df['name'].nunique()} / {len(df)}\n")
        f.write(f"rows duplicating an earlier 7-feature vector: {int(df.duplicated(subset=FEATURES).sum())}\n\n")
        f.write(df[FEATURES].describe().T.round(5).to_string() + "\n")
    print(f"Loaded {len(df)} proteins, {D} features, {K} classes {class_names}")

    use_deep = not args.no_deep
    if use_deep:
        try:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except ImportError:
            print("[warn] PyTorch not installed -> CNN/RNN/GCNN are skipped (pip install torch)")
            use_deep = False

    splits = list(StratifiedShuffleSplit(n_splits=cfg.n_splits, test_size=cfg.test_size,
                                         random_state=cfg.seed).split(X_raw, y))
    n_train, n_test = len(splits[0][0]), len(splits[0][1])

    rows, tune_rows, repeat_rows = [], [], []
    cm_total = np.zeros((K, K), dtype=int)
    pred_counts = np.zeros(K, dtype=int)
    weights, curves = [], []
    rf_imp, lr_coef, corr_mats = [], [], []

    for s, (tr, te) in enumerate(splits, 1):
        print(f"\n=== Split {s}/{cfg.n_splits} (train={len(tr)}, test={len(te)}) ===")
        scaler = StandardScaler().fit(X_raw[tr])                 # training statistics only
        Xtr, Xte = scaler.transform(X_raw[tr]), scaler.transform(X_raw[te])
        ytr, yte = y[tr], y[te]

        if cfg.tune:
            t0 = time.perf_counter()
            T0, alpha, grid = tune_sa(X_raw[tr], ytr, K, cfg, cfg.seed + s)
            grid.insert(0, "split", s)
            tune_rows.append(grid)
            print(f"  tuning (training data only): T0={T0}, alpha={alpha}  [{time.perf_counter() - t0:.1f}s]")
        else:
            T0, alpha = cfg.sa_T0, cfg.sa_alpha

        # ---- metaheuristics: repetition 1 = main results; all repetitions = robustness ----
        for r in range(cfg.n_repeats):
            for variant in ["SA", "VNS", "SA+VNS"]:
                seed = 1000 * (r + 1) + s
                rng = np.random.default_rng(seed)
                t0 = time.perf_counter()
                W, info = run_variant(variant, Xtr, ytr, K, T0, alpha, cfg, rng, seed)
                yp = predict_linear(W, add_bias(Xte))
                rt = time.perf_counter() - t0
                m = metrics(yte, yp)
                repeat_rows.append({"repeat": r + 1, "split": s, "model": variant, **m, "runtime_s": rt})
                if r == 0:
                    rows.append({"model": variant, "split": s, **m, "runtime_s": rt, "T0": T0, "alpha": alpha})
                    if variant == "SA+VNS":
                        cm_total += confusion_matrix(yte, yp, labels=list(range(K)))
                        pred_counts += np.bincount(yp, minlength=K)
                        weights.append(W)
                        curves.append(info)
        main = {v: next(x for x in rows if x["model"] == v and x["split"] == s) for v in ["SA", "VNS", "SA+VNS"]}
        print("  " + "  ".join(f"{v}: acc={main[v]['Accuracy']:.4f}" for v in main))

        # ---- classical benchmarks ----
        for name, model in sklearn_models(cfg.seed).items():
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            if name == "RFC":
                rf_imp.append(model.feature_importances_)
            if name == "LR":
                lr_coef.append(np.abs(model.coef_).mean(axis=0))
            rows.append({"model": name, "split": s, **metrics(yte, yp), "runtime_s": time.perf_counter() - t0})

        # ---- deep benchmarks ----
        corr_mats.append(np.corrcoef(Xtr, rowvar=False))
        if use_deep:
            import torch
            A_hat = normalized_adjacency(Xtr)
            for name, ctor in build_torch_models(D, K, A_hat).items():
                torch.manual_seed(cfg.seed)
                t0 = time.perf_counter()
                model = train_torch(ctor(), Xtr, ytr, cfg, cfg.seed)
                yp = torch_predict(model, Xte)
                rows.append({"model": name, "split": s, **metrics(yte, yp), "runtime_s": time.perf_counter() - t0})

    # =========================================================================
    # Tables
    # =========================================================================
    out = lambda name: os.path.join(cfg.out_dir, name)
    res = pd.DataFrame(rows)
    res.to_csv(out("per_split_metrics.csv"), index=False)
    order = [m for m in ["SA+VNS", "SA", "VNS", "LR", "LDA", "KNN", "CART", "NB", "SVM", "L_SVM",
                         "ETC", "RFC", "NN", "RNN", "CNN", "GCNN"] if m in set(res["model"])]
    mets = ["Accuracy", "Precision", "Recall", "F1", "MacroF1", "runtime_s"]
    agg = res.groupby("model")[mets].agg(["mean", "std"]).loc[order]
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(out("table_mean_std.csv"))

    rep = pd.DataFrame(repeat_rows)
    rep.to_csv(out("repeats_per_split.csv"), index=False)
    rep_mean = rep.groupby(["model", "repeat"])[["Accuracy", "F1", "MacroF1"]].mean().reset_index()
    rep_mean.to_csv(out("repeats_mean_per_repeat.csv"), index=False)
    rep_summary = rep_mean.groupby("model")[["Accuracy", "F1", "MacroF1"]].agg(["mean", "std", "min", "max"])
    rep_summary.to_csv(out("repeats_summary.csv"))

    if tune_rows:
        pd.concat(tune_rows).to_csv(out("tuning_grid_all_splits.csv"), index=False)
        tg = pd.concat(tune_rows).groupby(["T0", "alpha"])["inner_cv_macroF1"].agg(["mean", "std"]).reset_index()
        tg.to_csv(out("tuning_grid_summary.csv"), index=False)
        res[res.model == "SA+VNS"][["split", "T0", "alpha"]].to_csv(out("tuning_selected_per_split.csv"), index=False)

    acc = res.pivot(index="split", columns="model", values="Accuracy")[order]
    mf1 = res.pivot(index="split", columns="model", values="MacroF1")[order]
    st_rows = []
    for m in order[1:]:
        t, p = corrected_resampled_ttest(acc["SA+VNS"], acc[m], n_train, n_test)
        st_rows.append({"model": m, "SA+VNS_acc": acc["SA+VNS"].mean(), "model_acc": acc[m].mean(),
                        "delta_acc": acc["SA+VNS"].mean() - acc[m].mean(),
                        "delta_rel_%": 100 * (acc["SA+VNS"].mean() - acc[m].mean()) / acc[m].mean(),
                        "t_corrected": t, "p_corrected": p,
                        "p_wilcoxon": wilcoxon_p(acc["SA+VNS"], acc[m]),
                        "cohens_dz": cohens_dz(acc["SA+VNS"], acc[m]),
                        "delta_macroF1": mf1["SA+VNS"].mean() - mf1[m].mean(),
                        "p_corrected_macroF1": corrected_resampled_ttest(mf1["SA+VNS"], mf1[m], n_train, n_test)[1]})
    st = pd.DataFrame(st_rows)
    st["p_holm"] = holm(st["p_corrected"])
    st = st.sort_values("delta_acc", ascending=False)
    st.to_csv(out("table_stats_vs_sa_vns.csv"), index=False)

    pw = pd.DataFrame(np.ones((len(order), len(order))), index=order, columns=order)
    for a in order:
        for b in order:
            if a != b:
                pw.loc[a, b] = corrected_resampled_ttest(acc[a], acc[b], n_train, n_test)[1]
    pw.to_csv(out("pairwise_pvalues.csv"))

    cm_df = pd.DataFrame(cm_total, index=class_names, columns=class_names)
    cm_df.to_csv(out("confusion_matrix_sa_vns_summed.csv"))
    support = cm_total.sum(axis=1)
    per_class = pd.DataFrame({"class": class_names, "test_instances": support, "correct": np.diag(cm_total),
                              "recall": np.where(support > 0, np.diag(cm_total) / np.maximum(support, 1), np.nan),
                              "predicted": pred_counts, "predicted_share_%": 100 * pred_counts / pred_counts.sum()})
    per_class.to_csv(out("per_class_sa_vns.csv"), index=False)

    Wst = np.stack(weights)
    pd.DataFrame(Wst.mean(0), index=FEATURES + ["bias"], columns=class_names).to_csv(out("weights_mean.csv"))
    imp = np.abs(Wst[:, :D, :]).mean(axis=(0, 2))
    imp_sd = np.abs(Wst[:, :D, :]).mean(axis=2).std(axis=0, ddof=1) if len(Wst) > 1 else np.zeros(D)
    fi = pd.DataFrame({"feature": FEATURES, "mean_abs_weight": imp, "sd_over_splits": imp_sd,
                       "relative_%": 100 * imp / imp.sum()}).sort_values("mean_abs_weight", ascending=False)
    fi.to_csv(out("feature_importance_W.csv"), index=False)

    # Feature rankings: SA+VNS |W| vs Random-Forest Gini importance vs LR |coefficients|
    rank = pd.DataFrame({"feature": FEATURES, "SA+VNS_|W|": imp,
                         "RFC_gini": np.mean(rf_imp, axis=0), "LR_|coef|": np.mean(lr_coef, axis=0)})
    for c in ["SA+VNS_|W|", "RFC_gini", "LR_|coef|"]:
        rank[c + "_rank"] = rank[c].rank(ascending=False).astype(int)
    rank.sort_values("SA+VNS_|W|_rank").to_csv(out("feature_rankings_comparison.csv"), index=False)
    rho = {"SA+VNS vs RFC": stats.spearmanr(rank["SA+VNS_|W|"], rank["RFC_gini"])[0],
           "SA+VNS vs LR": stats.spearmanr(rank["SA+VNS_|W|"], rank["LR_|coef|"])[0],
           "RFC vs LR": stats.spearmanr(rank["RFC_gini"], rank["LR_|coef|"])[0]}
    pd.Series(rho, name="spearman_rho").to_csv(out("feature_rankings_spearman.csv"))
    corr = pd.DataFrame(np.mean(corr_mats, axis=0), index=FEATURES, columns=FEATURES)
    corr.to_csv(out("attribute_correlation_graph.csv"))

    # =========================================================================
    # Figures
    # =========================================================================
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out(name), dpi=300)
        plt.close(fig)

    labels4 = ["Accuracy", "Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, v in enumerate(["SA", "VNS", "SA+VNS"]):
        mu = [agg.loc[v, f"{k}_mean"] for k in labels4]
        sd = [agg.loc[v, f"{k}_std"] for k in labels4]
        bars = ax.bar(np.arange(4) + (i - 1) * 0.27, mu, 0.27, yerr=sd, capsize=3, label=v)
        for b_, m_ in zip(bars, mu):
            ax.text(b_.get_x() + b_.get_width() / 2, 0.02, f"{m_:.3f}", ha="center", va="bottom",
                    rotation=90, fontsize=8, color="white")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
    ax.set_ylim(0, 1); ax.set_ylabel(f"Mean ± SD over {cfg.n_splits} splits"); ax.legend(loc="lower right")
    save(fig, "fig_ablation_sa_vns_hybrid.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(order))
    for j, (k, mk) in enumerate(zip(labels4, ["o", "s", "^", "D"])):
        ax.errorbar(xs + (j - 1.5) * 0.15, agg[f"{k}_mean"], yerr=agg[f"{k}_std"], fmt=mk, capsize=2, ms=4,
                    label="F1-score" if k == "F1" else k)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel(f"Mean ± SD over {cfg.n_splits} splits"); ax.grid(alpha=0.3); ax.legend(ncol=4)
    save(fig, "fig_model_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ob = acc.mean().sort_values().index.tolist()
    ax.boxplot([acc[m].to_numpy() for m in ob])
    ax.set_xticks(range(1, len(ob) + 1)); ax.set_xticklabels(ob, rotation=45, ha="right")
    ax.set_ylabel("Test accuracy"); ax.grid(alpha=0.3)
    ax.set_title(f"Test accuracy over {cfg.n_splits} repeated stratified 80/20 splits")
    save(fig, "fig_accuracy_boxplot.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["tab:green" if v > 0 else "tab:red" for v in st["delta_acc"]]
    ax.bar(st["model"], st["delta_acc"], color=colors)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Mean accuracy difference\n(SA+VNS minus model)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig_accuracy_delta.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(st["model"], st["cohens_dz"], color=["tab:green" if v > 0 else "tab:red" for v in st["cohens_dz"]])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Paired Cohen's d (SA+VNS vs model)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig_cohens_d.png")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pw.values, cmap="Blues_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{pw.values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if pw.values[i, j] < 0.35 else "black")
    fig.colorbar(im, ax=ax, label="p-value (corrected resampled t-test, accuracy)")
    save(fig, "fig_pvalue_heatmap.png")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_total, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(K)); ax.set_yticklabels(class_names)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm_total[i, j], ha="center", va="center",
                    color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"SA+VNS confusion matrix (summed over {cfg.n_splits} test sets)")
    fig.colorbar(im, ax=ax)
    save(fig, "fig_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(class_names, per_class["predicted_share_%"])
    for b_, v in zip(bars, per_class["predicted_share_%"]):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("Share of SA+VNS predictions (%)")
    save(fig, "fig_prediction_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    rec = per_class["recall"].fillna(0)
    bars = ax.bar(class_names, rec)
    for b_, v, n in zip(bars, rec, support):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Per-class recall (SA+VNS)")
    save(fig, "fig_per_class_recall.png")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sa_c = np.mean([c["sa"] for c in curves], axis=0)
    vns_c = np.mean([c["vns"] for c in curves], axis=0)
    ax.plot(np.arange(len(sa_c)), sa_c, label="SA phase")
    ax.plot(np.arange(len(sa_c) - 1, len(sa_c) - 1 + len(vns_c)), vns_c, label="VNS phase")
    ax.axvline(len(sa_c) - 1, color="gray", ls="--")
    ax.set_xlabel("Iteration (SA iterations, then VNS outer iterations)")
    ax.set_ylabel("Best training fitness"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Mean convergence over {cfg.n_splits} splits")
    save(fig, "fig_convergence.png")

    # =========================================================================
    # Environment and report
    # =========================================================================
    import sklearn
    import scipy
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor(), "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
           "matplotlib": matplotlib.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    total = time.perf_counter() - t_start
    with open(out("environment_and_config.json"), "w") as f:
        json.dump({"environment": env, "config": asdict(cfg), "n_train": n_train, "n_test": n_test,
                   "total_runtime_s": total}, f, indent=2)

    pd.set_option("display.width", 250)
    lines = ["# SA+VNS on UCI E. coli - results", "",
             f"{cfg.n_splits} repeated stratified 80/20 splits (train={n_train}, test={n_test}), random_state={cfg.seed}",
             f"SA (T0, alpha): {'grid search with inner 3-fold CV on training data' if cfg.tune else f'fixed ({cfg.sa_T0}, {cfg.sa_alpha})'}",
             f"Total runtime: {total:.1f} s", "",
             "## Mean ± SD over splits (repetition 1)", "```", agg.round(4).to_string(), "```", "",
             f"## Robustness over {cfg.n_repeats} independent repetitions (mean of {cfg.n_splits} splits per repetition)",
             "```", rep_summary.round(4).to_string(), "```", "",
             "## SA+VNS versus every other model (accuracy, paired over splits)", "```",
             st.round(4).to_string(index=False), "```", "",
             "## SA+VNS confusion matrix (summed over test sets)", "```", cm_df.to_string(), "```", "",
             "## Per-class (SA+VNS)", "```", per_class.round(4).to_string(index=False), "```", "",
             "## Feature importance from |W|", "```", fi.round(4).to_string(index=False), "```", "",
             "## Feature rankings (SA+VNS |W|, RFC Gini, LR |coef|)", "```",
             rank.sort_values("SA+VNS_|W|_rank").round(4).to_string(index=False), "```",
             "Spearman rho: " + ", ".join(f"{k} = {v:.3f}" for k, v in rho.items()), "",
             "## Mean attribute correlation (training data; GCNN graph edges = |r|)", "```",
             corr.round(3).to_string(), "```"]
    if cfg.tune:
        lines += ["", "## Selected (T0, alpha) per split", "```",
                  res[res.model == "SA+VNS"][["split", "T0", "alpha"]].to_string(index=False), "```"]
    with open(out("REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nAll outputs saved to ./{cfg.out_dir}/")


if __name__ == "__main__":
    main()
