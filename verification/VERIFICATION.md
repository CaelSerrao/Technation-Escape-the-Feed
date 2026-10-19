# Numerical verification

Status: **PASS**. Verification performed on 18 October 2026.

## Source identity

The raw data were downloaded from the official GroupLens MovieLens 32M URL. The 238,950,008-byte archive's SHA-256 matches the identity recorded in the supplied package:

`e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee`

The `ratings.csv` and `movies.csv` MD5 checksums also match the archive's own checksums. No raw ratings are included in the sharing package.

## Two independent routes

1. **Fresh pipeline reproduction:** the supplied model and both analyses were rerun from the official raw archive using NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0 on Python 3.12. The model embeddings, selected users, assigned signals, all steering lists, all benchmark lists and the steering summary reproduced exactly. All five CSV tables agree within numerical precision. Of 946 numeric JSON values, 945 agree exactly and one interval endpoint differs by 1.11e-16. No displayed result changes.
2. **Independently written saved-model audit:** a separate script streamed the raw ratings, checked source/training counts, rebuilt profiles and every saved recommendation list, recalculated per-user metrics, bootstrap intervals, exposure distributions, standardized group rates and example movies. All **8,390 checks passed**. This script uses the saved fit; the first route separately verifies the fit and eligible future-rating cohort.

Machine-readable evidence: [saved_model_audit.json](saved_model_audit.json), [reproduction_comparison.json](reproduction_comparison.json), and the fresh-run logs in this folder. The original supplied package is preserved separately in the working audit folder; analysis scripts in this repository change input/output paths only.

## Numbers appearing in the submission

| Page / finding | Recalculated value | Result |
|---|---|---|
| Steering cohort | 2,000 profiles | Pass |
| Full history success by 5 likes | 30/2,000 = 1.5% | Pass |
| Reset success by 5 likes | 128/2,000 = 6.4% | Pass |
| Paired gain and interval | 4.90 pp; 95% interval 3.85 to 6.05 pp | Pass |
| Full history success by 10 likes | 4.15%, displayed as 4.2% | Pass |
| Reset success by 10 likes | 19.75%, displayed as 19.8% | Pass |
| Entire steering response curve | All 11 checkpoints, steps 0 through 10 | Pass |
| Exposure heatmap sample | 1,955/2,000; groups of 30+ profiles | Pass |
| Heatmap group sizes | Drama 1,301; Comedy 317; Action 168; Adventure 77; Thriller 51; Sci-Fi 41 | Pass |
| Heatmap cells and Other | All fractional genre shares and row totals | Pass |
| Sci-Fi-led exposure | Action 19.8%, Sci-Fi 16.6% | Pass |
| Later-rating evaluation | 2,000 sampled from 2,772 eligible users | Pass |
| nDCG@10 means | Popularity 0.082; Full 0.123; Reset 0.110 (rounded) | Pass |
| Effective genre means | Popularity 8.66; Full 6.98; Reset 6.19 (rounded) | Pass |
| Chart whiskers | All six mean intervals | Pass |
| Full minus popularity relevance | +0.041; interval +0.034 to +0.048 (rounded) | Pass |
| Effective genre reduction | 1.68; interval 1.59 to 1.78 (rounded) | Pass |
| Standardized comparison | 13 strata; 685 short, 490 long, 825 outside | Pass |
| Full history standardized rates | Short 1.0%, long 0.0% (rounded) | Pass |
| Reset standardized rates | Short 2.8%, long 9.1% (rounded) | Pass |
| Signed short-minus-long gaps | +1.0 pp and -6.4 pp (rounded from unrounded rates) | Pass |
| Raw source | 32,000,204 ratings; 200,948 users | Pass |
| Training catalogue | 7,712 movies, 15,113,616 positive interactions, 197,301 users | Pass |
| Methods constants | 32 factors; positive = rating >=4; pre-2023 training; 100+ positives per movie; latest-20 reset; 2,000 bootstrap draws | Pass |

Gaps are calculated before rounding: the reset rates 2.7568653% and 9.1256307% differ by -6.3687654 pp, displayed as -6.4 pp. Subtracting the displayed one-decimal rates instead gives -6.3 pp, which is a rounding artefact.

## What this verification does not establish

Correct arithmetic does not prove harm, demographic unfairness, changed awareness, user satisfaction or commercial-platform behaviour. Steering signals are synthetic. One fixed model and one signal sequence per profile were tested. History truncation changes both amount and content of history; equal weighting dilutes new signals by construction. Genre labels are not viewpoints. Subgroup rates are descriptive; zero observed successes do not imply zero uncertainty. The PDF retains these substantive limitations.

## Report provenance

The supplied package's original manifest contains the hash of an earlier PDF, not the final redesigned graphic. That original manifest is kept under `provenance/` only. The delivered report and code are hashed in the repository's `SHA256SUMS.txt`. Editing notes and pending-reproduction notes were removed from the graphic after numerical reproduction passed. Future validation is identified as future work, not presented as a completed experiment.
