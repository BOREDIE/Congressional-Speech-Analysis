#!/usr/bin/env Rscript
# Letter-labeled DAG + minimal English key. dagitty parses graph; base R draws
# (latent = dashed node border, observed = solid border; edges solid).
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

node_ring <- function(xc, yc, r, n = 72L) {
  th <- seq(0, 2 * pi, length.out = n)
  cbind(xc + r * cos(th), yc + r * sin(th))
}

short_arrow <- function(x1, y1, x2, y2, r) {
  dx <- x2 - x1
  dy <- y2 - y1
  len <- hypot(dx, dy)
  if (len < 1e-9) {
    return()
  }
  ux <- dx / len
  uy <- dy / len
  arrows(
    x1 + ux * r,
    y1 + uy * r,
    x2 - ux * r,
    y2 - uy * r,
    length = 0.07,
    angle = 18,
    col = "gray38",
    lwd = 1.25,
    lty = 1
  )
}

hypot <- function(dx, dy) {
  sqrt(dx * dx + dy * dy)
}

plot_dag_letters <- function() {
  cd <- coordinates(dag)
  xv <- unname(cd$x)
  yv <- -unname(cd$y)
  names(xv) <- names(cd$x)
  names(yv) <- names(cd$y)
  
  ed <- as.data.frame(edges(dag), stringsAsFactors = FALSE)
  ed <- ed[ed$e == "->", , drop = FALSE]
  
  rx <- 0.042 * (max(xv) - min(xv))
  pad <- 2.2 * rx
  plot.new()
  plot.window(
    xlim = c(min(xv) - pad, max(xv) + pad),
    ylim = c(min(yv) - pad, max(yv) + pad),
    asp = 1,
    xaxs = "i",
    yaxs = "i"
  )
  
  for (k in seq_len(nrow(ed))) {
    v <- as.character(ed$v[k])
    w <- as.character(ed$w[k])
    short_arrow(xv[v], yv[v], xv[w], yv[w], rx * 1.05)
  }
  
  for (nm in names(xv)) {
    lt <- if (nm %in% LATENT) {
      2L
    } else {
      1L
    }
    ring <- node_ring(xv[nm], yv[nm], rx)
    polygon(ring[, 1L], ring[, 2L], border = "gray25", lty = lt, lwd = 1.6, col = "white")
    text(xv[nm], yv[nm], nm, font = 2L, cex = 0.92, col = "gray10")
  }
}

KEY <- c(
  "A party",
  "B baseline ideology",
  "C House tenure",
  "D cohort",
  "E ideology |nom|",
  "F House seniority",
  "G treatment",
  "H speech length",
  "I audience",
  "J committee (latent)",
  "K leadership (latent)",
  "L ambition (latent)",
  "M peer influence (latent)",
  "N outcome"
)

out_dir <- file.path(root, "03_output")
dir.create(out_dir, FALSE, TRUE)
pdf_path <- file.path(out_dir, "dag_congressional_language.pdf")
png_path <- file.path(out_dir, "dag_congressional_language.png")

draw <- function() {
  layout(matrix(1:2, nrow = 2L), heights = c(0.64, 0.36))
  par(family = "sans")
  par(mar = c(0.2, 0.5, 0.35, 0.5))
  plot_dag_letters()
  par(mar = c(0.2, 0.55, 0.15, 0.55), cex = 0.5)
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")
  text(0.5, 0.98, "Key", font = 2L, cex = 1.05)
  text(0.5, 0.91, "Solid ring: observed. Dashed ring: latent (unobserved).", cex = 0.78, col = "gray35")
  n <- length(KEY)
  cols <- 2L
  rows <- as.integer(ceiling(n / cols))
  dy <- 0.78 / max(rows, 1L)
  ytop <- 0.84
  for (i in seq_len(n)) {
    col <- (i - 1L) %/% rows
    row <- (i - 1L) %% rows
    x <- 0.02 + col * 0.49
    y <- ytop - row * dy
    text(x, y, KEY[i], pos = 4, family = "sans")
  }
  layout(1L)
}

w_in <- 10
h_in <- 7.6

if (capabilities("cairo")) {
  grDevices::cairo_pdf(pdf_path, width = w_in, height = h_in)
} else {
  pdf(pdf_path, width = w_in, height = h_in)
}
draw()
dev.off()

png(png_path, width = w_in, height = h_in, units = "in", res = 320)
draw()
dev.off()

message("Wrote ", normalizePath(pdf_path, winslash = "/"))
message("Wrote ", normalizePath(png_path, winslash = "/"))
