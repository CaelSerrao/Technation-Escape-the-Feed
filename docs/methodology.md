# Methodology

Reproduction commands and file layout are in the repository README.

## Common training setup

Cutoff: 2023-01-01 00:00 UTC. Treat ratings >=4 as positive. Retain movies with >=100 pre-cutoff positive ratings. Fit 32-component TruncatedSVD (7 iterations) to a binary user-movie matrix, columns weighted by inverse square-root positive count. Normalize item vectors. The full-history user profile sums all retained positive item vectors. The reset profile sums the latest 20 starting likes. Timestamp ties break by movie ID. Rank by item-profile dot product and exclude all previously rated movies, even low-rated ones. The sum and mean produce the same within-user ordering.

## Q4: steering experiment

Randomly permute eligible users (>=20 retained pre-cutoff likes) and take the first 2,000 with an eligible target. Uniformly choose a target genre per user, with >=100 catalogue movies, <=10% historical positive membership, <3 initial top-10 matches under both profiles and >=10 unrated candidates. IMAX and missing labels cannot be targets. Uniformly sample ten unrated target-genre movies without replacement. Both methods receive the same sequence of synthetic positive signals. Added movies are excluded from recommendations. The reset starts with 20 likes and then accumulates signals; it is NOT a rolling window. Item vectors stay fixed.

Success means reaching >=3 target-genre movies in the top 10 at least once by the chosen step, not maintaining that level. Multi-genre movies count if they include the target label. The results.json also retains the stricter four-movie threshold and illustrative example, although the report prioritizes the new analyses. User bootstrap intervals use 2,000 draws, conditional on the selected model and synthetic sequences.

## Q1: exposure heatmap

Use the original 2,000 steering-test users and their full-history baseline lists, before synthetic signals. Fractionally allocate each movie across its genre labels after excluding IMAX and missing-label tokens. Movies with no usable genre get an Unlabelled category. Determine each user's dominant historical genre with the same fractional rule; ties follow alphabetically sorted genre labels. Average per-user exposure distributions within each group. Show groups with >=30 users (1,955 users), ten most-exposed genres plus Other. Other pools remaining genres and any unlabelled mass. The full set of groups/genres and their historical distributions remain in extended_results.json. Rows sum to 100% before rounding. These are selected behavioural cohorts, not demographic communities.

## Q3: actual later ratings

Use a SEPARATE random sample of 2,000 of 2,772 users with >=20 pre-cutoff retained likes and >=5 post-cutoff positive ratings in the retained catalogue. All later positives through the archive's October 2023 endpoint are test labels. No future ratings train the model or determine popularity. No synthetic steering signals enter this evaluation.

Compare Popularity (pre-cutoff positive counts, personalized only by excluding rated movies), Full history and 20-item reset. Each ranks the full retained catalogue of 7,712 movies, with previously rated movies excluded; there is no sampled-negative evaluation. Stable sorting breaks score ties by catalogue movie ID.

Relevance: binary nDCG@10 = DCG / ideal DCG, discount 1/log2(rank+1), with a relevant item defined by a later rating >=4. Unobserved items are unlabelled, not known dislikes. Recall@10 and hit rate are also saved.

Diversity: effective genre count = exp(-sum p*log(p)), where p is the mean fractional genre mix of the ten recommended movies, excluding unlabelled mass and renormalizing. It combines breadth and balance; it is not viewpoint diversity. Distinct-genre counts are also saved. Compute metrics per user before averaging. Intervals use 2,000 user-bootstrap draws; between-method differences are paired by user.

## Q5: additional fairness diagnostic

Compare users with 20-50 versus 101+ pre-cutoff positive movies. The middle group (51-100) is excluded from this contrast. Define a stratum by assigned target genre and ORIGINAL full-history baseline target count (0, 1 or 2). Keep strata with >=10 users in both groups. This retains 1,175 users: 685 short histories and 490 long histories, across 13 strata. The other 825 users are outside this comparison.

For each stratum, compute its fraction of the pooled retained sample. Apply those SAME weights to each group's within-stratum success rate. Gap = standardized short-history success minus standardized long-history success. Positive favours shorter histories, negative favours longer histories; absolute distance from zero measures the parity gap. This standardizes two task characteristics, not every potential confounder.

The report treats these as descriptive rates and does not display fairness confidence intervals. Some saved bootstrap intervals are degenerate because no full-history successes occurred in the retained long-history group. That is NOT evidence of zero population uncertainty. Sparse events and lack of model/sequence uncertainty prevent strong population fairness claims. Equal rates can also mean equally poor responsiveness, so parity must be read alongside both group rates and overall performance.

## Interpretation

The baseline recommender is deliberately transparent, not a model of a particular commercial service. Long histories dilute new equal-weight signals by design. Resetting changes historical content as well as length. The data reflect selected MovieLens volunteers, an activity-filtered relevance cohort, and a popularity-filtered catalogue. The later ratings are selective observations rather than complete relevance labels. No demographics, actual awareness changes, stereotypes or causal platform effects are measured. The reset improves steering responsiveness here but scores below full history on held-out relevance and genre diversity, and widens the standardized parity gap in the retained comparison. These are different dimensions; the report does not claim a general improvement.

