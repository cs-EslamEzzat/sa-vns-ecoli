# SA+VNS hybrid metaheuristic for E. coli protein localization

Code for the article **"Using a Hybrid Metaheuristic Model to Predict Protein Localization Sites"**
(Eslam E. M. Attia, Osman Ali Sadek Ibrahim, Abdelmgeid A. Ali; *Scientific Reports*, under revision).

A single script reproduces every result, table and figure reported in the article.

## Method in brief

- Linear scoring model `L = X W` with `X = [Z-scored 7 attributes, 1]` and `W` of shape (7 + 1) x 8
  (7 feature rows + 1 bias row, 8 localization classes); prediction = `argmax` of the class scores.
- Fitness (training data only): macro-F1 − 1e-4 · ||W||².
- Warm start: multinomial logistic regression on the training data + Gaussian noise N(0, 0.2²).
- **Phase 1 – Simulated Annealing**: Gaussian perturbation (σ = 0.05) of all weights, Metropolis
  acceptance `exp(Δ/T)`, geometric cooling `T ← αT`, 1000 iterations.
- **Phase 2 – Variable Neighborhood Search** (continuous, reduced VNS): neighborhood `N_k` perturbs
  `k` randomly chosen rows of `W` (σ = 0.15), `k = 1…5`; on improvement the incumbent is replaced
  and `k` is reset to 1, otherwise `k ← k + 1`; 500 outer iterations.
- `T0` and `α` are selected inside every split by a grid search with inner 3-fold CV on the
  training data only.

## Evaluation protocol

- UCI E. coli dataset (336 proteins, 7 attributes, 8 classes), downloaded automatically.
- 10 repeated stratified 80/20 train–test splits (`random_state = 42`), identical for all models.
- Z-score statistics, warm start, tuning and optimization use the training part only.
- Benchmarks: LR, LDA, KNN, CART, NB, SVM, L_SVM, ETC, RFC, MLP, CNN, RNN, GCNN.
- Statistics: corrected resampled t-test (Nadeau & Bengio, 2003) with Holm correction,
  Wilcoxon signed-rank test, paired Cohen's d.
- Robustness: the stochastic search is repeated with 5 independent seeds.

## Requirements

Python 3.9 or newer. Install the packages with

```bash
pip install -r requirements.txt
```

(PyTorch is only needed for the CNN, RNN and GCNN benchmarks; without it these three are skipped.)

## Usage

```bash
python sa_vns_ecoli.py                 # full experiment (a few minutes on a laptop CPU)
python sa_vns_ecoli.py --quick         # quick smoke test
python sa_vns_ecoli.py --data ecoli.data --out results   # use a local copy of the dataset
python sa_vns_ecoli.py --help          # all options
```

All outputs are written to `results/`:

| File | Content |
|------|---------|
| `REPORT.md` | Summary of all results |
| `table_mean_std.csv` | Mean ± SD of all metrics and runtime for all models (Tables 7 and 11) |
| `table_stats_vs_sa_vns.csv` | Paired tests and effect sizes (Table 12) |
| `repeats_summary.csv` | Robustness over 5 independent repetitions (Table 8) |
| `feature_rankings_comparison.csv` | SA+VNS weights vs Random Forest vs LR ranking (Table 13) |
| `tuning_*.csv` | Grid-search results and selected `T0`, `α` per split |
| `per_split_metrics.csv` | Every metric of every model on every split |
| `confusion_matrix_sa_vns_summed.csv`, `per_class_sa_vns.csv` | Confusion matrix and per-class results |
| `fig_*.png` | All result figures of the article |
| `environment_and_config.json` | Software versions and every parameter used |

Accuracy-type results are deterministic for a given environment; runtimes depend on the hardware.

## Data

UCI Machine Learning Repository, E. coli dataset: https://archive.ics.uci.edu/dataset/39/ecoli

## License

MIT License (see `LICENSE`).
