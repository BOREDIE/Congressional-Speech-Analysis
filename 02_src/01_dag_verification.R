#!/usr/bin/env Rscript
# Publication DAG for a double-column layout.
# - DAG only on the figure canvas (3.4 x 3.2 in, fits a single column).
# - Legend exported separately to LaTeX for the figure caption / note.
# - Sizes (node radius, font, arrow head) specified in points, not relative
#   units, so the DAG stays legible when shrunk to column width.
# install.packages("dagitty", repos = "https://cloud.r-project.org")

resolve_project_root <- function() {
  ev <- Sys.getenv("COGS206_PROJECT_ROOT", unset = "")
  if (nzchar(ev) && dir.exists(ev)) {
    return(normalizePath(ev, winslash = "/", mustWork = FALSE))
  }
  sp <- NA_character_
  ca <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", ca[grep("^--file=", ca, fixed = TRUE)])
  if (length(fa) > 0L) {
    sp <- normalizePath(fa[1L])
  } else if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) &&
             isTRUE(rstudioapi::isAvailable())) {
    ap <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nzchar(ap) && identical(tolower(basename(ap)), "01_dag_verification.r")) {
      sp <- normalizePath(ap)
    }
  }
  if (!is.na(sp)) {
    sdir <- dirname(sp)
    return(if (identical(basename(sdir), "02_src")) {
      normalizePath(dirname(sdir), winslash = "/", mustWork = FALSE)
    } else {
      normalizePath(sdir, winslash = "/", mustWork = FALSE)
    })
  }
  d <- tryCatch(normalizePath(getwd(), winslash = "/", mustWork = FALSE), error = function(e) getwd())
  for (.k in seq_len(20L)) {
    if (file.exists(file.path(d, "02_src", "01_dag_verification.R"))) {
      return(normalizePath(d, winslash = "/", mustWork = FALSE))
    }
    nd <- dirname(d)
    if (identical(nd, d)) {
      break
    }
    d <- nd
  }
  getwd()
}

root <- resolve_project_root()
setwd(root)

if (!requireNamespace("dagitty", quietly = TRUE)) {
  stop(
    "Missing dagitty. Run:\n  install.packages(\"dagitty\", repos=\"https://cloud.r-project.org\")\n",
    "Linux: install V8 deps first (e.g. libnode-dev, libcurl4-openssl-dev).",
    call. = FALSE
  )
}
library(dagitty)

DAG_TXT <- r"(
dag {
  L [latent,pos="2.5,-0.75"]
  M [latent,pos="3.5,-0.75"]
  J [latent,pos="4.05,0.35"]
  K [latent,pos="4.05,1.15"]
  A [pos="0,0"]
  B [pos="0,0.85"]
  C [pos="0,1.7"]
  D [pos="0,2.55"]
  E [pos="1,3.45"]
  F [pos="1,4.25"]
  G [exposure,pos="2,1.7"]
  H [pos="3,0.35"]
  I [pos="3,1.15"]
  N [outcome,pos="5.05,1.7"]
  A -> G
  A -> N
  B -> G
  B -> N
  C -> G
  C -> N
  D -> G
  D -> N
  E -> N
  E -> G
  F -> N
  F -> G
  L -> G
  L -> N
  G -> H
  G -> I
  G -> J
  G -> K
  M -> H
  M -> I
  M -> J
  M -> K
  M -> N
  H -> N
  I -> N
  J -> N
  K -> N
}
)"

dag <- dagitty(DAG_TXT)
LATENT <- c("J", "K", "L", "M")

# ---- Geometry helpers ------------------------------------------------------

hypot <- function(dx, dy) sqrt(dx * dx + dy * dy)

# Convert a length in points to user (data) units for the current device.
pt_to_usr <- function(pt) {
  # 1 inch = 72.27 pt; par("cin")[1L] is character width in inches.
  in_per_usr_x <- par("pin")[1L] / diff(par("usr")[1:2])
  (pt / 72.27) / in_per_usr_x
}

node_ring <- function(xc, yc, r, n = 96L) {
  th <- seq(0, 2 * pi, length.out = n)
  cbind(xc + r * cos(th), yc + r * sin(th))
}

short_arrow <- function(x1, y1, x2, y2, r, head_in = 0.05) {
  dx <- x2 - x1
  dy <- y2 - y1
  len <- hypot(dx, dy)
  if (len < 1e-9) return(invisible())
  ux <- dx / len
  uy <- dy / len
  arrows(
    x1 + ux * r,
    y1 + uy * r,
    x2 - ux * r,
    y2 - uy * r,
    length = head_in,
    angle  = 20,
    col    = "gray30",
    lwd    = 0.8,
    lty    = 1
  )
}

# ---- DAG drawing -----------------------------------------------------------

plot_dag_letters <- function(node_pt = 14, label_cex = 0.75) {
  cd <- coordinates(dag)
  xv <- unname(cd$x); yv <- -unname(cd$y)
  names(xv) <- names(cd$x); names(yv) <- names(cd$y)
  
  ed <- as.data.frame(edges(dag), stringsAsFactors = FALSE)
  ed <- ed[ed$e == "->", , drop = FALSE]
  
  # Compute padding in data units once a window is open. We open a provisional
  # window, measure pt->usr, then re-open with proper padding.
  plot.new()
  plot.window(
    xlim = range(xv), ylim = range(yv),
    asp = 1, xaxs = "i", yaxs = "i"
  )
  r_usr <- pt_to_usr(node_pt)         # node radius in data units
  pad   <- 1.8 * r_usr
  
  plot.new()
  plot.window(
    xlim = c(min(xv) - pad, max(xv) + pad),
    ylim = c(min(yv) - pad, max(yv) + pad),
    asp = 1, xaxs = "i", yaxs = "i"
  )
  r_usr <- pt_to_usr(node_pt)
  
  # Edges first so node fills cover the arrow tails cleanly.
  for (k in seq_len(nrow(ed))) {
    v <- as.character(ed$v[k]); w <- as.character(ed$w[k])
    short_arrow(xv[v], yv[v], xv[w], yv[w], r_usr * 1.02, head_in = 0.045)
  }
  
  for (nm in names(xv)) {
    lt   <- if (nm %in% LATENT) 2L else 1L
    fill <- if (nm == "G") "gray92" else if (nm == "N") "gray82" else "white"
    ring <- node_ring(xv[nm], yv[nm], r_usr)
    polygon(ring[, 1L], ring[, 2L],
            border = "gray15", lty = lt, lwd = 1.1, col = fill)
    text(xv[nm], yv[nm], nm, font = 2L, cex = label_cex, col = "gray5")
  }
}

# ---- Legend (LaTeX, for caption / note) ------------------------------------

KEY <- c(
  A = "party",
  B = "baseline ideology",
  C = "House tenure",
  D = "cohort",
  E = "ideology (|NOM|)",
  G = "treatment",
  H = "speech length",
  I = "audience",
  N = "outcome",
  F = "House seniority",
  J = "committee (latent)",
  K = "leadership (latent)",
  L = "ambition (latent)",
  M = "peer influence (latent)"
)

write_legend_tex <- function(path) {
  parts <- paste0("\\textbf{", names(KEY), "}~", KEY)
  body  <- paste(parts, collapse = "; ")
  txt <- paste0(
    "% Auto-generated by 01_dag_verification.R\n",
    "\\textit{Notes:} Solid ring: observed; dashed ring: latent. ",
    "G = treatment, N = outcome. ",
    body, ".\n"
  )
  writeLines(txt, path)
}

# ---- Output ----------------------------------------------------------------

out_dir  <- file.path(root, "03_output")
dir.create(out_dir, FALSE, TRUE)
pdf_path <- file.path(out_dir, "dag_congressional_language.pdf")
png_path <- file.path(out_dir, "dag_congressional_language.png")
tex_path <- file.path(out_dir, "dag_legend.tex")

# Column-width figure: 3.4 in wide x 3.2 in tall.
w_in <- 3.4
h_in <- 3.2

draw <- function() {
  par(family = "sans", mar = c(0.15, 0.15, 0.15, 0.15))
  # node_pt sized so letters read comfortably at column width.
  plot_dag_letters(node_pt = 9.5, label_cex = 0.78)
}

if (capabilities("cairo")) {
  grDevices::cairo_pdf(pdf_path, width = w_in, height = h_in, family = "sans")
} else {
  pdf(pdf_path, width = w_in, height = h_in)
}
draw(); dev.off()

png(png_path, width = w_in, height = h_in, units = "in", res = 600)
draw(); dev.off()

write_legend_tex(tex_path)

message("Wrote ", normalizePath(pdf_path, winslash = "/"))
message("Wrote ", normalizePath(png_path, winslash = "/"))
message("Wrote ", normalizePath(tex_path, winslash = "/"))