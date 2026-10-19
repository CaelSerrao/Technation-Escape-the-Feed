# Escape the Feed

**Your tastes can change. Can your feed?**

A five-page Phase 1 submission for TECHNATION's Behind the Feed Data Challenge, using MovieLens 32M to examine information exposure, user control and trade-offs in one transparent recommender.

## Read the submission

[Open the five-page graphic](reports/Escape_the_Feed_Phase_1.pdf).

In a controlled simulation, full history reached an assigned new-genre target within five added likes for **30/2,000 profiles (1.5%)**, versus **128/2,000 (6.4%)** after keeping the latest 20 starting likes. The paired gain was **4.90 percentage points** (95% user-bootstrap interval **3.85 to 6.05**).

The reset was easier to redirect, but it scored lower on both held-out relevance and effective genre diversity. Its standardized short-versus-long history gap also grew in absolute size. These are different outcomes; we do not claim a reset is a general improvement.

## What the data support

- The steering signals and recommendation lists are simulated. Starting histories and later-rating labels are real MovieLens observations.
- Exposure means a generated top-10 list, not actual views, awareness or a complete feed.
- The steering and later-rating evaluations use different 2,000-user cohorts; see the methodology for their overlap.
- Genre labels measure content variety, not viewpoint diversity. History-length groups are behavioural groups, not demographic communities.
- This equal-weight history model dilutes new signals in long histories by construction. The experiment does not establish a causal effect on commercial platforms.

## Verification

[Read the numerical verification report](verification/VERIFICATION.md). The official input archive matches the recorded SHA-256. A fresh fit reproduced the saved model and all steering outputs exactly. Fresh extension results agree to floating-point precision (largest difference 1.11e-16). An independently written audit passed 8,390 checks, including the individual recommendation rankings and metrics. Exact re-training can vary across platforms or numerical libraries; use the included saved model for the independent audit.

## Reproduce

Python 3.12 with the pinned package versions is recommended. Full training reads all 32 million ratings and needs several GB of memory. Run commands from the repository root. A virtual environment is recommended.

```sh
python -m pip install -r requirements.txt
```

**Rebuild the graphic from the included results (no raw-data download needed):**

```sh
python analysis/build_graphic.py
```

**Download the official dataset and independently verify the saved results:**

```sh
python download_data.py
python analysis/verify_saved_results.py --results results --data data/ml-32m.zip --output verification/local_audit.json
```

**Refit the model and rerun every experiment into a separate folder:**

```sh
python analysis/analyze.py --data data/ml-32m.zip --output .reproduce
python analysis/extend_analysis.py --data data/ml-32m.zip --output .reproduce
python analysis/build_graphic.py --results .reproduce --output reports/Reproduced_Phase_1.pdf
```

To put the public repository link in the report after publishing:

```sh
python analysis/build_graphic.py --repo-url https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY
```

## Files

| Folder | Contents |
|---|---|
| `analysis/` | Portable original analyses, independent verifier, vector PDF builder |
| `results/` | Saved fit, cohort tables, recommendation lists, per-user metrics and summaries |
| `verification/` | Audit evidence and fresh-run comparison |
| `reports/` | Five-page submission PDF |
| `docs/` | Detailed methods and original dataset documentation/conditions |
| `provenance/` | Original package manifest; its PDF hash belongs to an earlier report |
| `data/` | Local raw-data download, excluded from Git |

`SHA256SUMS.txt` records the delivered files. It excludes itself and any Git internals. NPZ recommendation indices refer to the movie order in `results/model.npz`.

## Sources and related work

- [Official MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) and [dataset README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html).
- Harper & Konstan (2015), [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872).
- Curmei, Dean & Recht (ICML 2021), [Quantifying Availability and Discovery in Recommender Systems via Stochastic Reachability](https://proceedings.mlr.press/v139/curmei21a.html). Related prior work; our simple genre-steering test is not an implementation of its optimized reachability method.

See [DATA_NOTICE.md](DATA_NOTICE.md) for provenance and data conditions. The repository is supporting material; the PDF contains the submission's findings, methods and limitations.
