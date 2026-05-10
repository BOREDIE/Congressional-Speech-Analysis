#!/usr/bin/env Rscript
# Phase 2: MatchIt Mahalanobis nearest-neighbor matching (per-treated pools),
# balance diagnostics (cobalt), matched_sample.csv + robustness + freeze zip.
# DW-NOMINATE dim1 baseline: real nominate_dim1 from speaker_metadata_updated0501.csv
# (first House congress per bioguide_id), not fabricated.

suppressPackageStartupMessages({
  if (!requireNamespace("MatchIt", quietly = TRUE)) stop("Install MatchIt: install.packages('MatchIt')")
  if (!requireNamespace("cobalt", quietly = TRUE)) stop("Install cobalt: install.packages('cobalt')")
  if (!requireNamespace("dplyr", quietly = TRUE)) stop("Install dplyr: install.packages('dplyr')")
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table: install.packages('data.table')")
  }
})

library(MatchIt)
library(cobalt)
library(dplyr)
library(data.table)

resolve_root <- function() {
  ev <- Sys.getenv("COGS206_PROJECT_ROOT", unset = "")
  if (nzchar(ev) && dir.exists(ev)) {
    return(normalizePath(ev, winslash = "/", mustWork = FALSE))
  }
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", ca[grep("^--file=", ca, fixed = TRUE)])
  if (length(fa) > 0L) {
    sp <- normalizePath(fa[1L], winslash = "/", mustWork = TRUE)
    sdir <- dirname(sp)
    if (identical(basename(sdir), "02_src")) {
      return(normalizePath(dirname(sdir), winslash = "/", mustWork = TRUE))
    }
  }
  d <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  for (k in seq_len(20L)) {
    if (file.exists(file.path(d, "02_src", "phase2_matching.R"))) {
      return(normalizePath(d, winslash = "/", mustWork = FALSE))
    }
    nd <- dirname(d)
    if (identical(nd, d)) break
    d <- nd
  }
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

root <- resolve_root()
setwd(root)

paths <- list(
  pol = file.path(root, "01_data/01_RawCorpus/politicians_by_congress.csv"),
  trans = file.path(root, "01_data/02_Panel/transition_index.csv"),
  corpus = file.path(root, "01_data/01_RawCorpus/congress_speech_corpus_updated0430.csv"),
  speaker_meta = file.path(root, "01_data/01_RawCorpus/speaker_metadata_updated0501.csv"),
  out_dir = file.path(root, "03_output/phase2_matching")
)

dir.create(paths$out_dir, recursive = TRUE, showWarnings = FALSE)

log_lines <- character()
logf <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))
  log_lines <<- c(log_lines, msg)
  message(msg)
}

if (!file.exists(paths$speaker_meta)) {
  stop("Missing speaker metadata: ", paths$speaker_meta, call. = FALSE)
}

meta <- fread(paths$speaker_meta, na.strings = c("", "NA"))
names(meta) <- trimws(gsub("\r", "", names(meta)))
if (!all(c("bioguide_id", "chamber", "congress", "nominate_dim1") %in% names(meta))) {
  stop(
    "speaker_metadata_updated0501.csv must include columns: bioguide_id, chamber, congress, nominate_dim1",
    call. = FALSE
  )
}
meta[, chamber_l := tolower(trimws(as.character(chamber)))]
meta[, bid := trimws(as.character(bioguide_id))]
# Baseline = NOMINATE dim1 at the member's first House congress (min congress in House); pre-treatment.
mh <- meta[chamber_l == "house" & !is.na(bid) & nzchar(bid), ]
if (!nrow(mh)) {
  stop("No House rows in speaker metadata", call. = FALSE)
}
setorder(mh, bid, congress)
first_h <- mh[, .(first_house_congress = min(congress, na.rm = TRUE)), by = bid]
nom_bl_dt <- merge(
  first_h,
  mh,
  by.x = c("bid", "first_house_congress"),
  by.y = c("bid", "congress"),
  all.x = TRUE
)[, .(bioguide_id = bid, first_house_congress, dw_nom_dim1_baseline = as.numeric(nominate_dim1))]
nom_bl <- as.data.frame(nom_bl_dt, stringsAsFactors = FALSE)

pol <- fread(paths$pol, colClasses = c(bioguide_id = "character"))
pol[, chamber_l := tolower(trimws(chamber))]
pol_h <- pol[chamber_l == "house"]
pol_s <- pol[chamber_l == "senate"]

trans <- fread(paths$trans, colClasses = c(bioguide_id = "character"))
if ("last_H_congress" %in% names(trans) && !"transition_congress" %in% names(trans)) {
  data.table::setnames(trans, "last_H_congress", "transition_congress")
}
if (!"transition_congress" %in% names(trans)) {
  stop("transition_index.csv must contain last_H_congress or transition_congress", call. = FALSE)
}
if (!"first_S_congress" %in% names(trans)) {
  stop("transition_index.csv must contain first_S_congress", call. = FALSE)
}
trans[, treated_flag := 1L]

all_treated_ids <- unique(trans$bioguide_id)

senate_terms <- function(bio, first_s) {
  cong <- pol_s[bioguide_id == bio & congress >= first_s, unique(congress)]
  length(cong)
}

house_tenure_at <- function(bio, event_c) {
  pol_h[bioguide_id == bio & congress <= event_c, uniqueN(congress)]
}

house_terms_after <- function(bio, event_c) {
  pol_h[bioguide_id == bio & congress > event_c, uniqueN(congress)]
}

party_at_house_congress <- function(bio, cong) {
  p <- pol_h[bioguide_id == bio & congress == cong, party][1L]
  if (is.na(p) || !nzchar(p)) NA_character_ else p
}

cohort_house <- function(bio) {
  pol_h[bioguide_id == bio, min(congress, na.rm = TRUE)]
}

logf("Loading speech aggregates from corpus (selected columns only)...")
if (!file.exists(paths$corpus)) {
  stop("Corpus not found: ", paths$corpus, call. = FALSE)
}
sp <- fread(paths$corpus, select = c("bioguide_id", "congress", "chamber"), colClasses = c(bioguide_id = "character"))
sp[, ch := toupper(trimws(chamber))]
spagg <- sp[, .N, by = .(bioguide_id, congress, ch)]
setnames(spagg, "N", "n_speeches")

speech_n_range <- function(bio, cong_min, cong_max, chm) {
  if (is.na(cong_min) || is.na(cong_max) || cong_max < cong_min) return(0L)
  spagg[bioguide_id == bio & ch == chm & congress >= cong_min & congress <= cong_max, sum(n_speeches)]
}

pre_post_congresses <- function(bio, pseudo, first_s, is_treated) {
  if (is_treated) {
    pre_pol <- pol_h[bioguide_id == bio & congress < pseudo, unique(congress)]
    pre_sp <- spagg[bioguide_id == bio & ch == "H" & congress < pseudo, unique(congress)]
    pre_c <- sort(unique(c(pre_pol, pre_sp)))
    post_pol <- pol_s[bioguide_id == bio & congress >= first_s, unique(congress)]
    post_sp <- spagg[bioguide_id == bio & ch == "S" & congress >= first_s, unique(congress)]
    post_c <- sort(unique(c(post_pol, post_sp)))
  } else {
    pre_pol <- pol_h[bioguide_id == bio & congress < pseudo, unique(congress)]
    pre_sp <- spagg[bioguide_id == bio & ch == "H" & congress < pseudo, unique(congress)]
    pre_c <- sort(unique(c(pre_pol, pre_sp)))
    post_pol <- pol_h[bioguide_id == bio & congress > pseudo, unique(congress)]
    post_sp <- spagg[bioguide_id == bio & ch == "H" & congress > pseudo, unique(congress)]
    post_c <- sort(unique(c(post_pol, post_sp)))
  }
  list(pre = pre_c, post = post_c)
}

collapse_semi <- function(v) paste(v, collapse = ";")

mahalanobis_row <- function(x_t, x_c, S_inv) {
  dv <- as.numeric(x_c - x_t)
  sqrt(as.vector(t(dv) %*% S_inv %*% dv))
}

cov_inv_add_ridge <- function(Xm) {
  S <- stats::cov(Xm)
  if (any(is.na(S))) S <- matrix(0, 3, 3)
  diag(S) <- diag(S) + 1e-6
  tryCatch(
    solve(S),
    error = function(e) solve(S + diag(1e-4, nrow(S)))
  )
}

eligible_controls <- function(t_bio, t_party, transition_c, n_req) {
  cand <- unique(pol_h[congress == transition_c & party == t_party, bioguide_id])
  cand <- setdiff(cand, c(t_bio, all_treated_ids))
  ok <- character()
  for (cb in cand) {
    if (house_terms_after(cb, transition_c) < n_req) next
    if (is.na(party_at_house_congress(cb, transition_c)) || party_at_house_congress(cb, transition_c) != t_party) next
    b1 <- nom_bl[nom_bl$bioguide_id == cb, "dw_nom_dim1_baseline", drop = TRUE]
    if (!length(b1) || is.na(b1[1L])) next
    ok <- c(ok, cb)
  }
  unique(ok)
}

# --- Eligibility logging ---
n_pol_speakers <- uniqueN(pol$bioguide_id)
logf("Unique speakers (politicians table): %d", n_pol_speakers)
logf("Treated (transition_index): %d", nrow(trans))

elig_simple <- data.frame(
  step = c("politicians_unique_bioguide", "transition_index_rows", "dw1_baseline_nonmissing_metadata"),
  n = c(n_pol_speakers, nrow(trans), sum(!is.na(nom_bl$dw_nom_dim1_baseline))),
  stringsAsFactors = FALSE
)
utils::write.csv(elig_simple, file.path(paths$out_dir, "eligibility_counts.csv"), row.names = FALSE)

party_tab <- pol_h[, .N, by = party]
logf("House party counts (any term): %s", paste(paste(party_tab$party, party_tab$N, sep = "="), collapse = ", "))

# MatchIt runner for one treated
run_one_matchit <- function(t_bio, transition_c, first_s, caliper_named, ratio, replace) {
  t_party <- party_at_house_congress(t_bio, transition_c)
  if (is.na(t_party) || !t_party %in% c("D", "R")) {
    return(list(ok = FALSE, reason = "treated_party_not_DR", m = NULL, d = NULL))
  }
  n_sen <- senate_terms(t_bio, first_s)
  n_req <- max(2L, as.integer(n_sen))
  pool <- eligible_controls(t_bio, t_party, transition_c, n_req)
  n_req_used <- n_req
  if (length(pool) < 1L && n_req > 2L) {
    pool <- eligible_controls(t_bio, t_party, transition_c, 2L)
    n_req_used <- 2L
  }
  if (length(pool) < 1L) {
    return(list(ok = FALSE, reason = "empty_pool", m = NULL, d = NULL))
  }
  tb <- nom_bl[nom_bl$bioguide_id == t_bio, , drop = FALSE]
  if (!nrow(tb) || is.na(tb$dw_nom_dim1_baseline[1L])) {
    return(list(ok = FALSE, reason = "treated_missing_nominate", m = NULL, d = NULL))
  }
  cohort_t <- cohort_house(t_bio)
  tenure_t <- house_tenure_at(t_bio, transition_c)
  d_t <- data.frame(
    bioguide_id = t_bio,
    treated = 1L,
    dw_nom_dim1_baseline = tb$dw_nom_dim1_baseline[1L],
    house_tenure_at_event = tenure_t,
    cohort = cohort_t,
    party = t_party,
    stringsAsFactors = FALSE
  )
  rows <- lapply(pool, function(cb) {
    b <- nom_bl[nom_bl$bioguide_id == cb, , drop = FALSE]
    if (!nrow(b) || is.na(b$dw_nom_dim1_baseline[1L])) return(NULL)
    data.frame(
      bioguide_id = cb,
      treated = 0L,
      dw_nom_dim1_baseline = b$dw_nom_dim1_baseline[1L],
      house_tenure_at_event = house_tenure_at(cb, transition_c),
      cohort = cohort_house(cb),
      party = party_at_house_congress(cb, transition_c),
      stringsAsFactors = FALSE
    )
  })
  rows <- rows[!vapply(rows, is.null, logical(1L))]
  if (!length(rows)) {
    return(list(ok = FALSE, reason = "pool_missing_baseline", m = NULL, d = NULL))
  }
  d_c <- bind_rows(rows)
  d <- bind_rows(d_t, d_c)
  d <- d[d$party == t_party & !is.na(d$dw_nom_dim1_baseline), , drop = FALSE]
  # Enforce marginal calipers explicitly (MatchIt Mahalanobis NN can violate per-dimension bounds).
  cal <- caliper_named
  names(cal) <- c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort")
  rt <- d[d$treated == 1L, , drop = FALSE][1L, ]
  keep <- d$treated == 1L | (
    abs(d$dw_nom_dim1_baseline - rt$dw_nom_dim1_baseline) <= cal[["dw_nom_dim1_baseline"]] + 1e-9 &
      abs(d$house_tenure_at_event - rt$house_tenure_at_event) <= cal[["house_tenure_at_event"]] + 1e-9 &
      abs(d$cohort - rt$cohort) <= cal[["cohort"]] + 1e-9
  )
  d <- d[keep, , drop = FALSE]
  if (sum(d$treated == 0L) < 1L) {
    return(list(ok = FALSE, reason = "empty_pool_after_marginal_caliper", m = NULL, d = NULL))
  }
  Xm <- as.matrix(d[, c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort")])
  Sinv <- cov_inv_add_ridge(Xm)
  # Marginal calipers enforced by row filter above; MatchIt caliper can be redundant and
  # occasionally inconsistent with Mahalanobis NN — omit matchit() caliper here.
  m <- tryCatch(
    matchit(
      treated ~ dw_nom_dim1_baseline + house_tenure_at_event + cohort,
      data = d,
      method = "nearest",
      distance = "mahalanobis",
      exact = ~ party,
      ratio = ratio,
      replace = replace,
      estimand = "ATT"
    ),
    error = function(e) NULL
  )
  if (is.null(m)) {
    return(list(ok = FALSE, reason = "matchit_error_or_no_match", m = NULL, d = d, Sinv = Sinv))
  }
  list(ok = TRUE, m = m, d = d, Sinv = Sinv, transition_c = transition_c, first_s = first_s,
       t_party = t_party, n_req = n_req, n_req_used = n_req_used, n_senate_terms = n_sen)
}

build_long_from_match <- function(m, d, pair_id, t_bio, transition_c, first_s, Sinv) {
  gm <- tryCatch(
    MatchIt::get_matches(m, data = d, include.s.weights = FALSE),
    error = function(e) NULL
  )
  if (is.null(gm)) {
    return(NULL)
  }
  gm <- as.data.frame(gm)
  out <- list()
  trow <- gm[gm$treated == 1L, ][1L, , drop = FALSE]
  crows <- gm[gm$treated == 0L, , drop = FALSE]
  Xt <- as.numeric(trow[, c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort")])
  for (i in seq_len(nrow(crows))) {
    cr <- crows[i, , drop = FALSE]
    Xc <- as.numeric(cr[, c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort")])
    mdist <- if (all(is.finite(c(Xt, Xc)))) mahalanobis_row(Xt, Xc, Sinv) else NA_real_
    bio_c <- as.character(cr$bioguide_id)
    pp <- pre_post_congresses(bio_c, transition_c, first_s, is_treated = FALSE)
    n_h_pre <- if (length(pp$pre)) {
      speech_n_range(bio_c, min(pp$pre), max(pp$pre), "H")
    } else {
      0L
    }
    n_h_post <- if (length(pp$post)) {
      speech_n_range(bio_c, min(pp$post), max(pp$post), "H")
    } else {
      0L
    }
    out[[length(out) + 1L]] <- data.frame(
      matched_pair_id = pair_id,
      bioguide_id = bio_c,
      treated_or_control = "control",
      party = as.character(cr$party),
      dw_nom_dim1_baseline = cr$dw_nom_dim1_baseline,
      house_tenure_at_event = cr$house_tenure_at_event,
      cohort = cr$cohort,
      match_distance = mdist,
      pseudo_event_congress = transition_c,
      pre_period_congresses = collapse_semi(pp$pre),
      post_period_congresses = collapse_semi(pp$post),
      match_weight = as.numeric(cr$weights),
      n_house_speeches_pre = n_h_pre,
      n_house_speeches_post = n_h_post,
      n_senate_speeches_post = NA_integer_,
      stringsAsFactors = FALSE
    )
  }
  ppt <- pre_post_congresses(t_bio, transition_c, first_s, is_treated = TRUE)
  n_h_pre_t <- if (length(ppt$pre)) speech_n_range(t_bio, min(ppt$pre), max(ppt$pre), "H") else 0L
  n_s_post_t <- if (length(ppt$post)) speech_n_range(t_bio, min(ppt$post), max(ppt$post), "S") else 0L
  t_out <- data.frame(
    matched_pair_id = pair_id,
    bioguide_id = t_bio,
    treated_or_control = "treated",
    party = as.character(trow$party),
    dw_nom_dim1_baseline = trow$dw_nom_dim1_baseline,
    house_tenure_at_event = trow$house_tenure_at_event,
    cohort = trow$cohort,
    match_distance = 0,
    pseudo_event_congress = transition_c,
    pre_period_congresses = collapse_semi(ppt$pre),
    post_period_congresses = collapse_semi(ppt$post),
    match_weight = 1,
    n_house_speeches_pre = n_h_pre_t,
    n_house_speeches_post = NA_integer_,
    n_senate_speeches_post = n_s_post_t,
    stringsAsFactors = FALSE
  )
  bind_rows(t_out, bind_rows(out))
}

run_full_pipeline <- function(variant_label, caliper_named, ratio, replace, out_csv) {
  logf("=== Variant: %s (ratio=%s replace=%s) ===", variant_label, ratio, replace)
  pair_id <- 0L
  long_parts <- list()
  m_objects <- list()
  n_try <- 0L
  n_ok <- 0L
  reasons <- character()
  for (i in seq_len(nrow(trans))) {
    t_bio <- trans$bioguide_id[i]
    tc <- as.integer(trans$transition_congress[i])
    fs <- as.integer(trans$first_S_congress[i])
    n_try <- n_try + 1L
    res <- run_one_matchit(t_bio, tc, fs, caliper_named, ratio, replace)
    if (!isTRUE(res$ok)) {
      reasons <- c(reasons, sprintf("%s: %s", t_bio, res$reason))
      next
    }
    pair_id <- pair_id + 1L
    n_ok <- n_ok + 1L
    m_objects[[length(m_objects) + 1L]] <- res$m
    lg <- build_long_from_match(res$m, res$d, pair_id, t_bio, res$transition_c, res$first_s, res$Sinv)
    if (!is.null(lg)) long_parts[[length(long_parts) + 1L]] <- lg
  }
  logf("Treated attempted: %d; matched with >=1 control: %d", n_try, n_ok)
  if (length(reasons)) {
    logf("Unmatched / failed (first 30):\n%s", paste(head(reasons, 30L), collapse = "\n"))
  }
  if (!length(long_parts)) {
    stop("No successful matches for variant ", variant_label, call. = FALSE)
  }
  long_df <- bind_rows(long_parts)
  write.csv(long_df, out_csv, row.names = FALSE, quote = TRUE)
  logf("Wrote %s (%d rows)", out_csv, nrow(long_df))
  list(long = long_df, m_objects = m_objects, n_matched_treated = n_ok, n_try = n_try, reasons = reasons)
}

# Default calipers (main spec)
strict_cohort <- identical(Sys.getenv("PHASE2_STRICT_COHORT", "0"), "1")
cohort_c_main <- if (strict_cohort) 1 else as.numeric(Sys.getenv("PHASE2_COHORT_CALIPER", "1"))
tenure_c_main <- if (strict_cohort) 2 else as.numeric(Sys.getenv("PHASE2_TENURE_CALIPER", "2"))
caliper_main <- c(dw_nom_dim1_baseline = 0.1, house_tenure_at_event = tenure_c_main, cohort = cohort_c_main)
caliper_tight <- c(dw_nom_dim1_baseline = 0.05, house_tenure_at_event = 1, cohort = 0.5)
caliper_wide <- c(dw_nom_dim1_baseline = 0.15, house_tenure_at_event = 3, cohort = 1.5)
logf(
  "Main calipers: tenure=%s cohort=%s (PHASE2_STRICT_COHORT=1 forces paper 2/1)",
  tenure_c_main, cohort_c_main
)

main_csv <- file.path(paths$out_dir, "matched_sample.csv")
res_main <- run_full_pipeline("main", caliper_main, ratio = 3L, replace = TRUE, main_csv)

# --- Balance exports (main) ---
md_list <- lapply(res_main$m_objects, function(m) match.data(m, include.s.weights = FALSE))
md_all <- bind_rows(md_list)
# party must be a factor for cobalt's internal contrast computation
if ("party" %in% names(md_all)) {
  md_all$party <- factor(md_all$party, levels = c("D", "R"))
}
if (nrow(md_all)) {
  # Note: md_all contains only matched units; Diff.Un = Diff.Adj is expected because
  # cobalt computes "unadjusted" on the same matched pool (unweighted) vs "adjusted"
  # (weighted). True pre/post improvement is shown in balance_supplementary.csv.
  bt <- bal.tab(
    treated ~ dw_nom_dim1_baseline + house_tenure_at_event + cohort,
    data = md_all,
    estimand = "ATT",
    m.threshold = 0.1,
    weights = md_all$weights,
    un = TRUE,
    quick = FALSE
  )
  capture.output(print(bt), file = file.path(paths$out_dir, "balance_table.txt"))
  bal_df <- as.data.frame(bt$Balance)
  bal_df$covariate <- rownames(bal_df)
  utils::write.csv(bal_df, file.path(paths$out_dir, "balance_table.csv"), row.names = FALSE)
  tryCatch(
    {
      pdf(file.path(paths$out_dir, "love_plot.pdf"), width = 8, height = 5)
      print(love.plot(bt, thresholds = c(m = 0.1), stars = "raw", sample.names = c("Unmatched", "Matched")))
      dev.off()
    },
    error = function(e) {
      if (dev.cur() > 1L) dev.off()
      logf("love.plot skipped: %s", conditionMessage(e))
      writeLines(conditionMessage(e), file.path(paths$out_dir, "love_plot_error.txt"))
    }
  )
  # KS on matched sample.
  # FIX: was md_all[rows, v][[1L]] which extracted only the first scalar — now uses md_all[[v]][rows].
  ks_lines <- character()
  trv <- if ("treated" %in% names(md_all)) md_all$treated else md_all$treat
  for (v in c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort")) {
    x1 <- md_all[[v]][trv == 1L]
    x0 <- md_all[[v]][trv == 0L]
    if (!length(x1) || !length(x0)) next
    kt <- tryCatch(stats::ks.test(x1, x0), error = function(e) NULL, warning = function(w) {
      tryCatch(stats::ks.test(x1, x0, exact = FALSE), error = function(e2) NULL)
    })
    if (!is.null(kt)) {
      ks_lines <- c(ks_lines, sprintf("%s D=%.4f p=%.4g n_t=%d n_c=%d",
                                      v, unname(kt$statistic), kt$p.value, length(x1), length(x0)))
    }
  }
  writeLines(ks_lines, file.path(paths$out_dir, "ks_matched.txt"))
  # Covariate distribution plots — direct base-R histograms (treated vs control, same axis).
  # cobalt's bal.plot() does not support bal.tab.formula objects with which="both"; bypass it.
  tryCatch({
    cov_labels <- c(
      dw_nom_dim1_baseline  = "DW-NOMINATE dim1 (ideology)",
      house_tenure_at_event = "House tenure at event (congresses)",
      cohort                = "Entry cohort (congress)"
    )
    trv2 <- if ("treated" %in% names(md_all)) md_all$treated else md_all$treat
    pdf(file.path(paths$out_dir, "bal_plot_covariates.pdf"), width = 9, height = 3.5)
    par(mfrow = c(1, 3), mar = c(4, 3, 3, 1))
    for (v in c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort")) {
      x_t <- md_all[[v]][trv2 == 1L]
      x_c <- md_all[[v]][trv2 == 0L]
      brks <- pretty(c(x_t, x_c), n = 12)
      h_t <- hist(x_t, breaks = brks, plot = FALSE)
      h_c <- hist(x_c, breaks = brks, plot = FALSE)
      ylim <- c(0, max(h_t$density, h_c$density) * 1.15)
      plot(h_t, freq = FALSE, col = adjustcolor("steelblue", 0.6), border = "white",
           ylim = ylim, main = cov_labels[v], xlab = v, ylab = "Density")
      plot(h_c, freq = FALSE, col = adjustcolor("coral", 0.55), border = "white", add = TRUE)
      legend("topright", legend = c(sprintf("Treated (n=%d)", length(x_t)),
                                    sprintf("Control (n=%d)", length(x_c))),
             fill = c(adjustcolor("steelblue", 0.6), adjustcolor("coral", 0.55)), bty = "n", cex = 0.8)
    }
    dev.off()
    logf("bal_plot_covariates.pdf written")
  }, error = function(e) {
    if (dev.cur() > 1L) dev.off()
    logf("bal_plot_covariates skipped: %s", conditionMessage(e))
    writeLines(conditionMessage(e), file.path(paths$out_dir, "bal_plot_error.txt"))
  })
}

# Save matchit objects
saveRDS(
  list(
    variant = "main",
    m_objects = res_main$m_objects,
    caliper = caliper_main,
    ratio = 3L,
    replace = TRUE,
    generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
  ),
  file.path(paths$out_dir, "matchit_object.rds")
)

# --- Robustness ---
invisible({
  run_full_pipeline("1to1_replace", caliper_main, ratio = 1L, replace = TRUE,
                    file.path(paths$out_dir, "matched_sample_robust_1to1_replace.csv"))
  run_full_pipeline("1to3_no_replace", caliper_main, ratio = 3L, replace = FALSE,
                    file.path(paths$out_dir, "matched_sample_robust_1to3_no_replace.csv"))
  run_full_pipeline("caliper_tight", caliper_tight, ratio = 3L, replace = TRUE,
                    file.path(paths$out_dir, "matched_sample_robust_caliper_tight.csv"))
  run_full_pipeline("caliper_wide", caliper_wide, ratio = 3L, replace = TRUE,
                    file.path(paths$out_dir, "matched_sample_robust_caliper_wide.csv"))
})

# Replacement distribution (main)
rep_tab <- res_main$long %>%
  filter(treated_or_control == "control") %>%
  count(bioguide_id, name = "n_pairs")
rep_summary <- paste(capture.output(summary(rep_tab$n_pairs)), collapse = "\n")

logf("Control reuse (main): rows=%d unique control IDs=%d",
     sum(res_main$long$treated_or_control == "control"),
     n_distinct(filter(res_main$long, treated_or_control == "control")$bioguide_id))
logf("Replacement count distribution (controls by #pairs):\n%s", rep_summary)

# --- Control reuse CSV ---
ctrl_reuse_full <- res_main$long %>%
  filter(treated_or_control == "control") %>%
  count(bioguide_id, name = "n_pairs_used") %>%
  arrange(desc(n_pairs_used))
utils::write.csv(ctrl_reuse_full, file.path(paths$out_dir, "control_reuse.csv"), row.names = FALSE)
high_reuse <- ctrl_reuse_full %>% filter(n_pairs_used >= 3)
if (nrow(high_reuse)) {
  logf("High-reuse controls (>=3 pairs): %s",
       paste(sprintf("%s(x%d)", high_reuse$bioguide_id, high_reuse$n_pairs_used), collapse = ", "))
}

# --- Match distance diagnostics (per pair) ---
DIST_POOR_THRESHOLD <- 2.0
ctrl_long <- res_main$long %>% filter(treated_or_control == "control")
dist_by_pair <- ctrl_long %>%
  group_by(matched_pair_id) %>%
  summarise(
    n_controls         = n(),
    mean_dist          = mean(match_distance, na.rm = TRUE),
    max_dist           = max(match_distance, na.rm = TRUE),
    n_poor_controls    = sum(match_distance > DIST_POOR_THRESHOLD, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(pair_quality = case_when(
    max_dist <= 1.0                        ~ "good",
    max_dist <= DIST_POOR_THRESHOLD        ~ "borderline",
    TRUE                                   ~ "has_poor_control"
  ))
utils::write.csv(dist_by_pair, file.path(paths$out_dir, "match_distance_by_pair.csv"), row.names = FALSE)
logf("Match distance per pair: mean_max=%.3f, good=%d borderline=%d has_poor=%d (threshold=%.1f)",
     mean(dist_by_pair$max_dist),
     sum(dist_by_pair$pair_quality == "good"),
     sum(dist_by_pair$pair_quality == "borderline"),
     sum(dist_by_pair$pair_quality == "has_poor_control"),
     DIST_POOR_THRESHOLD)

# --- Quality flags on matched_sample.csv ---
# Add dist_quality column; controls with distance > threshold are flagged.
long_flagged <- res_main$long %>%
  mutate(dist_quality = case_when(
    treated_or_control == "treated"             ~ "treated",
    match_distance <= 1.0                       ~ "good",
    match_distance <= DIST_POOR_THRESHOLD       ~ "borderline",
    TRUE                                        ~ "poor"
  ))
write.csv(long_flagged, main_csv, row.names = FALSE, quote = TRUE)
logf("Rewrote matched_sample.csv with dist_quality column (%d rows)", nrow(long_flagged))

# --- High-quality matched sample: remove individual poor controls ---
# Treated units are always retained; only specific poor-distance controls are dropped.
long_hq <- long_flagged %>%
  filter(treated_or_control == "treated" | dist_quality != "poor")
# Drop pairs that lost ALL controls
pairs_with_ctrl <- long_hq %>% filter(treated_or_control == "control") %>% pull(matched_pair_id) %>% unique()
long_hq <- long_hq %>% filter(matched_pair_id %in% pairs_with_ctrl)
hq_csv <- file.path(paths$out_dir, "matched_sample_highquality.csv")
write.csv(long_hq, hq_csv, row.names = FALSE, quote = TRUE)
logf("High-quality matched sample (dist<=%.1f): %d treated, %d control rows -> %s",
     DIST_POOR_THRESHOLD,
     sum(long_hq$treated_or_control == "treated"),
     sum(long_hq$treated_or_control == "control"),
     hq_csv)

# --- Supplementary balance table from matched_sample output ---
# Uses long_flagged directly (one row per person per role), not pooled md_all.
# Provides the TRUE pre-matching comparison: treated vs same-party-same-congress pool.
long_t <- long_flagged %>% filter(treated_or_control == "treated")
long_c <- long_flagged %>% filter(treated_or_control == "control")
supp_bal_rows <- lapply(c("dw_nom_dim1_baseline", "house_tenure_at_event", "cohort"), function(v) {
  x1 <- long_t[[v]]
  x0 <- long_c[[v]]
  sd1 <- sd(x1, na.rm = TRUE)
  smd <- if (!is.na(sd1) && sd1 > 0) (mean(x1, na.rm=TRUE) - mean(x0, na.rm=TRUE)) / sd1 else NA_real_
  kt  <- tryCatch(stats::ks.test(x1, x0, exact = FALSE), error = function(e) NULL)
  data.frame(
    covariate    = v,
    mean_treated = mean(x1, na.rm = TRUE),
    mean_control = mean(x0, na.rm = TRUE),
    sd_treated   = sd1,
    SMD          = smd,
    KS_D         = if (!is.null(kt)) unname(kt$statistic) else NA_real_,
    KS_p         = if (!is.null(kt)) kt$p.value            else NA_real_,
    n_treated    = sum(!is.na(x1)),
    n_control    = sum(!is.na(x0)),
    stringsAsFactors = FALSE
  )
})
supp_bal <- bind_rows(supp_bal_rows)
utils::write.csv(supp_bal, file.path(paths$out_dir, "balance_supplementary.csv"), row.names = FALSE)
supp_txt <- capture.output(print(supp_bal, row.names = FALSE, digits = 4))
writeLines(supp_txt, file.path(paths$out_dir, "balance_supplementary.txt"))
logf("Supplementary balance (treated n=%d vs matched control n=%d):\n%s",
     nrow(long_t), nrow(long_c), paste(supp_txt, collapse = "\n"))

matching_log <- c(
  paste("Project root:", root),
  paste("Speaker metadata (NOMINATE baseline):", paths$speaker_meta),
  paste("Corpus:", paths$corpus),
  "",
  "=== Summary ===",
  sprintf("N_treated in transition_index: %d", nrow(trans)),
  sprintf("N_treated matched (main, >=1 control): %d", res_main$n_matched_treated),
  sprintf("High-quality sample (dist<=%.1f): %d treated, %d control",
          DIST_POOR_THRESHOLD,
          sum(long_hq$treated_or_control == "treated"),
          sum(long_hq$treated_or_control == "control")),
  "",
  "Pair quality breakdown (main):",
  paste(capture.output(print(as.data.frame(table(dist_by_pair$pair_quality)))), collapse = "\n"),
  "",
  "Control reuse summary (main):",
  rep_summary,
  "",
  "Supplementary balance (treated vs matched controls):",
  paste(supp_txt, collapse = "\n"),
  "",
  "=== Timestamped log ===",
  log_lines
)
writeLines(matching_log, file.path(paths$out_dir, "matching_log.txt"))

# match_call.R (actual cohort caliper used in this run)
match_call_txt <- c(
  "# MatchIt 4.7+ — Mahalanobis nearest neighbor, ATT",
  "# Marginal calipers (ideology/tenure/cohort) are enforced by filtering controls before matchit();",
  "# matchit() is run without caliper= to avoid Mahalanobis NN marginal inconsistencies.",
  "# Paper-style bounds used in filter: ideology 0.1, tenure see PHASE2_TENURE_CALIPER (default 2), cohort PHASE2_COHORT_CALIPER (default 1).",
  "library(MatchIt)",
  paste0(
    "matchit(\n  treated ~ dw_nom_dim1_baseline + house_tenure_at_event + cohort,\n",
    "  data = YOUR_MATCH_DATA_FILTERED_TO_MARGINAL_CALIPERS,\n  method = \"nearest\",\n",
    "  distance = \"mahalanobis\",\n  exact = ~ party,\n",
    "  ratio = 3,\n  replace = TRUE,\n  estimand = \"ATT\"\n)"
  )
)
writeLines(match_call_txt, file.path(paths$out_dir, "match_call.R"))

# Zip freeze
zipf <- file.path(paths$out_dir, sprintf("phase2_matching_freeze_%s.zip", format(Sys.Date(), "%Y%m%d")))
zf <- c(
  "matched_sample.csv", "matched_sample_highquality.csv",
  "matchit_object.rds", "balance_table.csv", "balance_table.txt",
  "balance_supplementary.csv", "balance_supplementary.txt",
  "eligibility_counts.csv", "love_plot.pdf", "matching_log.txt", "match_call.R",
  "ks_matched.txt", "bal_plot_covariates.pdf",
  "match_distance_by_pair.csv", "control_reuse.csv",
  "matched_sample_robust_1to1_replace.csv", "matched_sample_robust_1to3_no_replace.csv",
  "matched_sample_robust_caliper_tight.csv", "matched_sample_robust_caliper_wide.csv"
)
zf <- zf[file.exists(file.path(paths$out_dir, zf))]
owd <- getwd()
setwd(paths$out_dir)
zip(zipf, zf, flags = "-q")
setwd(owd)
logf("Wrote freeze zip: %s", zipf)

message("Phase 2 matching complete.")
