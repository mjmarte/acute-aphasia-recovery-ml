suppressPackageStartupMessages(library(ggplot2))

REV <- Sys.getenv("REV_ROOT", ".")
args <- commandArgs(trailingOnly = TRUE)
OUT <- if (length(args)) args[1] else file.path(REV, "figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

A <- read.csv(file.path(REV, "results", "rerun", "all_tasks.csv"))
best_r2 <- function(variant) {
  d <- A[A$variant == variant & A$fs == "FS4", ]
  d <- d[order(-round(d$f1, 2), -d$mcc), ]
  d$r2[1]
}
R2 <- c(aq_aq = best_r2("c_wab_own"),
        cu_cu = best_r2("c_nct_own"),
        aq_cu = best_r2("c_nct_aq"),
        cu_aq = best_r2("c_wab_cu"))
pct <- function(x) paste0(round(100 * x), "%")
message(sprintf("R2: %s", paste(names(R2), signif(R2, 5), collapse = "  ")))

FAM    <- "Arial"
INK    <- "#222222"
MUTED  <- "#6B6B6B"
PURPLE <- "#400286"
ORANGE <- "#E87722"
TINT_P <- "#ECE4F4"
TINT_O <- "#FCEADB"
pt <- function(p) p / .pt

bx <- list(l = c(5, 36), r = c(84, 115))
by <- list(t = c(67, 87), b = c(35, 55))
cy <- c(t = mean(by$t), b = mean(by$b))
off <- 4.2
gap <- 0.9

arrow_poly <- function(x0, y0, x1, y1, w, hl = 4.6, hw = NULL, id) {
  if (is.null(hw)) hw <- w + 3.8
  L <- sqrt((x1 - x0)^2 + (y1 - y0)^2); ux <- (x1 - x0) / L; uy <- (y1 - y0) / L
  px <- -uy; py <- ux; bxh <- x1 - ux * hl; byh <- y1 - uy * hl
  data.frame(id = id,
             x = c(x0 + px * w / 2, bxh + px * w / 2, bxh + px * hw / 2, x1, bxh - px * hw / 2, bxh - px * w / 2, x0 - px * w / 2),
             y = c(y0 + py * w / 2, byh + py * w / 2, byh + py * hw / 2, y1, byh - py * hw / 2, byh - py * w / 2, y0 - py * w / 2))
}
K <- 6.0
xs <- bx$l[2]; x1 <- bx$r[1] - gap
tail <- 3
arr <- list(
  aq_cu = list(xs = xs, ys = cy["t"] - off, x1 = x1, y1 = cy["b"] + off, col = PURPLE),
  cu_aq = list(xs = xs, ys = cy["b"] + off, x1 = x1, y1 = cy["t"] - off, col = ORANGE),
  aq_aq = list(xs = xs, ys = cy["t"] + off, x1 = x1, y1 = cy["t"] + off, col = PURPLE),
  cu_cu = list(xs = xs, ys = cy["b"] - off, x1 = x1, y1 = cy["b"] - off, col = ORANGE)
)
polys <- do.call(rbind, lapply(names(arr), function(k) {
  a <- arr[[k]]; L <- sqrt((a$x1 - a$xs)^2 + (a$y1 - a$ys)^2)
  x0 <- a$xs - tail * (a$x1 - a$xs) / L; y0 <- a$ys - tail * (a$y1 - a$ys) / L
  arrow_poly(x0, y0, a$x1, a$y1, w = K * R2[[k]], id = k)
}))
polys$col <- vapply(polys$id, function(k) arr[[k]]$col, "")

lab <- do.call(rbind, lapply(names(arr), function(k) {
  a <- arr[[k]]; t <- switch(k, aq_aq = 0.5, cu_cu = 0.5, aq_cu = 0.27, cu_aq = 0.73)
  data.frame(id = k, x = a$xs + t * (a$x1 - a$xs), y = a$ys + t * (a$y1 - a$ys), col = a$col, lab = pct(R2[[k]]))
}))
rrect <- function(x0, x1, y0, y1, r, id, n = 48) {
  a <- function(cx, cy, from, to) { th <- seq(from, to, length.out = n); cbind(cx + r * cos(th), cy + r * sin(th)) }
  m <- rbind(a(x1 - r, y0 + r, -pi / 2, 0), a(x1 - r, y1 - r, 0, pi / 2),
             a(x0 + r, y1 - r, pi / 2, pi), a(x0 + r, y0 + r, pi, 3 * pi / 2))
  data.frame(id = id, x = m[, 1], y = m[, 2])
}
pill_w <- 12.4; pill_h <- 6.6
pills <- do.call(rbind, lapply(seq_len(nrow(lab)), function(i) {
  d <- rrect(lab$x[i] - pill_w / 2, lab$x[i] + pill_w / 2, lab$y[i] - pill_h / 2, lab$y[i] + pill_h / 2, r = 2.3, id = lab$id[i])
  d$col <- lab$col[i]; d
}))

boxes <- rbind(
  transform(rrect(bx$l[1], bx$l[2], by$t[1], by$t[2], r = 2.6, id = "l_t"), fill = TINT_P),
  transform(rrect(bx$l[1], bx$l[2], by$b[1], by$b[2], r = 2.6, id = "l_b"), fill = TINT_O),
  transform(rrect(bx$r[1], bx$r[2], by$t[1], by$t[2], r = 2.6, id = "r_t"), fill = TINT_P),
  transform(rrect(bx$r[1], bx$r[2], by$b[1], by$b[2], r = 2.6, id = "r_b"), fill = TINT_O)
)
box_txt <- data.frame(
  x = c(mean(bx$l), mean(bx$l), mean(bx$r), mean(bx$r)),
  y = c(cy["t"], cy["b"], cy["t"], cy["b"]),
  lab = c("Aphasia\nseverity", "Discourse\ncontent", "Aphasia\nseverity", "Discourse\ncontent")
)

g <- ggplot() +
  annotate("text", x = 60, y = 115.5, label = "Forecasting language one year after stroke\nfrom measures taken within days of onset",
           family = FAM, fontface = "bold", size = pt(14), colour = INK, lineheight = 0.95, vjust = 1) +
  annotate("text", x = mean(bx$l), y = 94.2, label = "Within days", family = FAM, size = pt(12), colour = MUTED) +
  annotate("text", x = mean(bx$r), y = 94.2, label = "One year later", family = FAM, size = pt(12), colour = MUTED) +
  annotate("segment", x = 45, xend = 75, y = 94.2, yend = 94.2, colour = MUTED, linewidth = 0.35,
           arrow = arrow(length = unit(1.8, "mm"), type = "closed")) +
  geom_polygon(data = polys[polys$id %in% c("aq_cu"), ], aes(x, y, group = id, fill = I(col)), colour = NA) +
  geom_polygon(data = polys[polys$id %in% c("cu_aq"), ], aes(x, y, group = id, fill = I(col)), colour = NA) +
  geom_polygon(data = polys[polys$id %in% c("aq_aq", "cu_cu"), ], aes(x, y, group = id, fill = I(col)), colour = NA) +
  geom_polygon(data = boxes, aes(x, y, group = id, fill = I(fill)), colour = NA) +
  geom_text(data = box_txt, aes(x, y, label = lab), family = FAM, fontface = "bold", size = pt(13.5), colour = INK, lineheight = 0.92) +
  geom_polygon(data = pills, aes(x, y, group = id, colour = I(col)), fill = "white", linewidth = 0.45, linejoin = "round") +
  geom_text(data = lab, aes(x, y, label = lab, colour = I(col)), family = FAM, fontface = "bold", size = pt(13)) +
  annotate("text", x = 60, y = 27.6, label = "Percent of one-year variance explained",
           family = FAM, size = pt(12), colour = MUTED) +
  annotate("segment", x = 8, xend = 112, y = 22.6, yend = 22.6, colour = "#D9D9D9", linewidth = 0.3) +
  annotate("text", x = 60, y = 16.4, label = "Brain imaging added little beyond acute severity",
           family = FAM, size = pt(13), colour = INK) +
  annotate("text", x = 60, y = 8.6, label = "Discourse recovery is partly separable from severity",
           family = FAM, size = pt(13), colour = INK) +
  coord_fixed(xlim = c(0, 120), ylim = c(0, 120), expand = FALSE, clip = "off") +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", colour = NA), plot.margin = margin(0, 0, 0, 0))

stem <- file.path(OUT, "graphical_abstract")
ggsave(paste0(stem, ".pdf"), g, width = 120, height = 120, units = "mm", device = cairo_pdf)
ggsave(paste0(stem, ".tiff"), g, width = 120, height = 120, units = "mm", dpi = 600, device = ragg::agg_tiff, compression = "lzw")
ggsave(paste0(stem, "_preview.png"), g, width = 120, height = 120, units = "mm", dpi = 200, device = ragg::agg_png)
message("written: ", stem, ".{pdf,tiff,_preview.png}")
