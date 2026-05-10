#!/usr/bin/env Rscript
# Optional: export House (bioguide_id, congress, dim1) from speaker_metadata_updated0501.csv
# into dw_nominate_house_dim1.csv for external tools. Values are real nominate_dim1 only — no fabrication.

resolve_root <- function() {
  fa <- sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])
  if (length(fa)) {
    sp <- normalizePath(fa[1L], winslash = "/", mustWork = TRUE)
    sdir <- dirname(sp)
    if (identical(basename(sdir), "02_src")) {
      return(normalizePath(dirname(sdir), winslash = "/", mustWork = TRUE))
    }
  }
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

root <- resolve_root()
meta_path <- file.path(root, "01_data/01_RawCorpus/speaker_metadata_updated0501.csv")
out_path <- file.path(root, "01_data/01_RawCorpus/dw_nominate_house_dim1.csv")

stopifnot(file.exists(meta_path))
suppressPackageStartupMessages(library(data.table))
meta <- fread(meta_path, na.strings = c("", "NA"))
names(meta) <- trimws(gsub("\r", "", names(meta)))
meta[, chamber_l := tolower(trimws(as.character(chamber)))]
meta[, bid := trimws(as.character(bioguide_id))]
h <- meta[chamber_l == "house" & !is.na(bid) & nzchar(bid) & !is.na(nominate_dim1), .(bioguide_id = bid, congress, dim1 = as.numeric(nominate_dim1))]
h <- unique(h, by = c("bioguide_id", "congress"))
utils::write.csv(h, out_path, row.names = FALSE, quote = TRUE)
message("Wrote ", nrow(h), " rows from real nominate_dim1 to ", out_path)
