# MatchIt 4.7+ — Mahalanobis nearest neighbor, ATT
# Marginal calipers (ideology/tenure/cohort) are enforced by filtering controls before matchit();
# matchit() is run without caliper= to avoid Mahalanobis NN marginal inconsistencies.
# Paper-style bounds used in filter: ideology 0.1, tenure see PHASE2_TENURE_CALIPER (default 2), cohort PHASE2_COHORT_CALIPER (default 1).
library(MatchIt)
matchit(
  treated ~ dw_nom_dim1_baseline + house_tenure_at_event + cohort,
  data = YOUR_MATCH_DATA_FILTERED_TO_MARGINAL_CALIPERS,
  method = "nearest",
  distance = "mahalanobis",
  exact = ~ party,
  ratio = 3,
  replace = TRUE,
  estimand = "ATT"
)
