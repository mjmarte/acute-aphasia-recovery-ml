suppressPackageStartupMessages({library(tidyverse); library(patchwork); library(scales); library(ragg)})
REV <- Sys.getenv("REV_ROOT", "."); FD <- file.path(REV, "results/figdata"); OUT <- file.path(REV, "manuscript/figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
outcome_colors <- c("aphasia_resolution" = "#400286", "discourse_content" = "#E87722")
class_colors <- c("1" = "#2E7D32", "0" = "#C62828")
tag_theme <- theme(plot.tag = element_text(face = "bold", size = 14, family = "Helvetica"), plot.tag.position = c(0, 1))
set.seed(42)
modality_colors <- c("Clinical" = "#4393C3", "Volume" = "#999999", "Cortical" = "#E87722", "WM" = "#74C476", "Network" = "#400286")
theme_marte <- function() theme_classic(base_size = 11, base_family = "Helvetica") + theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 10), legend.position = "bottom", legend.title = element_text(size = 9), legend.text = element_text(size = 8), axis.title = element_text(size = 10), plot.title = element_text(face = "bold", size = 12), panel.grid.major = element_line(color = "gray92"))
save_both <- function(name, plot, w, h) { ggsave(file.path(OUT, paste0(name, ".png")), plot, width = w, height = h, dpi = 300, device = ragg::agg_png); ggsave(file.path(OUT, paste0(name, ".tif")), plot, width = w, height = h, dpi = 600, compression = "lzw"); cat("saved", name, "\n") }
pred <- read_csv(file.path(FD, "best_predictions.csv"), show_col_types = FALSE)
labs_out <- c(aphasia_resolution = "Aphasia resolution (12-month WAB-AQ)", discourse_content = "Discourse content (12-month content units)")
p_dot <- function(oc, panel) {
  d <- pred %>% filter(outcome == oc) %>% arrange(y_true) %>% mutate(idx = row_number(), err = abs(y_pred_mean - y_true))
  ggplot(d, aes(x = idx)) + geom_linerange(aes(ymin = pred_min, ymax = pred_max), color = "gray70", linewidth = 0.4) + geom_point(aes(y = y_true), shape = 4, size = 1.4, color = "black") +
    geom_point(aes(y = y_pred_mean, color = err), size = 1.8) + geom_hline(yintercept = unique(d$thr), linetype = "dotted") + scale_color_viridis_c(name = "Absolute error", option = "C") +
    labs(x = "Patients ordered by observed 12-month score", y = if (oc == "aphasia_resolution") "WAB-AQ" else "Content units", title = panel) + theme_marte()
}
p_box <- function(oc, panel) {
  d <- pred %>% filter(outcome == oc) %>% mutate(grp = ifelse(correct, "Correctly classified", "Misclassified"))
  ggplot(d, aes(x = grp, y = pred_sd, fill = grp)) + geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) + geom_jitter(width = 0.12, size = 1.2, alpha = 0.8) +
    scale_fill_manual(values = c("Correctly classified" = "#2E7D32", "Misclassified" = "#C62828")) + labs(x = NULL, y = "Within-patient SD of predictions", title = panel) + theme_marte() + theme(legend.position = "none")
}
fig2 <- (p_dot("aphasia_resolution", "Aphasia resolution") | p_box("aphasia_resolution", NULL)) / (p_dot("discourse_content", "Discourse content") | p_box("discourse_content", NULL)) + plot_layout(widths = c(3, 1)) + plot_annotation(tag_levels = "A") & tag_theme
save_both("fig2", fig2, 10, 7.5)
p_scatter <- function(oc, panel) {
  d <- pred %>% filter(outcome == oc) %>% mutate(cls = factor(cls_true), shape = ifelse(correct, "Correctly classified", "Misclassified"))
  fit <- lm(y_true ~ y_pred_mean, d); lim <- range(c(d$y_true, d$y_pred_mean)); thr <- unique(d$thr)
  ggplot(d, aes(x = y_pred_mean, y = y_true)) + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") + geom_vline(xintercept = thr, linetype = "dotted") + geom_hline(yintercept = thr, linetype = "dotted") +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE, color = "#4393C3", fill = "#4393C340", linewidth = 0.8) + geom_point(aes(color = cls, shape = shape), size = 2.2, alpha = 0.9) +
    scale_color_manual(values = class_colors, labels = c("0" = "Not resolved/normalized", "1" = "Resolved/normalized"), name = NULL) + scale_shape_manual(values = c("Correctly classified" = 16, "Misclassified" = 17), name = NULL) +
    coord_equal(xlim = lim, ylim = lim) + labs(x = "Predicted 12-month score", y = "Observed 12-month score", title = panel, subtitle = sprintf("calibration slope %.2f, intercept %.1f", coef(fit)[2], coef(fit)[1])) + theme_marte()
}
fig4 <- (p_scatter("aphasia_resolution", "Aphasia resolution") + p_scatter("discourse_content", "Discourse content")) + plot_layout(guides = "collect") + plot_annotation(tag_levels = "A") & theme(legend.position = "bottom") & tag_theme
save_both("fig4", fig4, 10, 5.5)
cells_raw <- read_csv(file.path(FD, "cells.csv"), show_col_types = FALSE)
if (nrow(cells_raw) > 0) {
cells <- cells_raw %>% mutate(predictor = factor(predictor, levels = c("Own", "Other", "Both")), outcome = factor(outcome, levels = c("12-month WAB-AQ", "12-month content units")))
p_cells <- function(metric, lo, hi, ylab, panel) {
  ggplot(cells, aes(x = predictor, y = .data[[metric]], fill = fs)) + geom_col(position = position_dodge(0.8), width = 0.7) + geom_errorbar(aes(ymin = .data[[lo]], ymax = .data[[hi]]), position = position_dodge(0.8), width = 0.25) +
    facet_wrap(~outcome) + scale_fill_manual(values = c(FS1 = "#C6ACDF", FS4 = "#400286"), name = "Feature set") + labs(x = "Acute language predictor(s)", y = ylab, title = panel) + theme_marte()
}
fig3 <- p_cells("r2", "r2_lo", "r2_hi", expression(R^2), NULL) / p_cells("f1", "f1_lo", "f1_hi", "F1", NULL) + plot_layout(guides = "collect") + plot_annotation(tag_levels = "A") & theme(legend.position = "bottom") & tag_theme
save_both("fig3", fig3, 8, 8)
} else cat("fig3 skipped: no cells yet\n")
o <- read_csv(file.path(FD, "outcomes.csv"), show_col_types = FALSE)
hist_p <- function(x, title, thr, binw) ggplot(tibble(x = na.omit(x)), aes(x)) + geom_histogram(binwidth = binw, fill = "#9970AB", color = "white") + geom_vline(xintercept = thr, linetype = "dashed") + labs(x = NULL, y = "Patients", title = title) + theme_marte()
s1 <- (hist_p(o$wab_aq_acute, "Acute WAB-AQ (n = 73)", 93.8, 5) | hist_p(o$wab_aq_12mo, "12-month WAB-AQ (n = 73)", c(93.8, 96.7), 5)) / (hist_p(o$nct_total_cu_acute, "Acute content units (n = 52)", 22.1, 3) | hist_p(o$nct_total_cu_12mo, "12-month content units (n = 61)", 22.1, 3)) + plot_annotation(tag_levels = "A") & tag_theme
save_both("supp_s1", s1, 9, 7)
if (file.exists(file.path(FD, "shap_ranks.csv"))) {
  sh <- read_csv(file.path(FD, "shap_ranks.csv"), show_col_types = FALSE)
  nice_map <- c(wab_aq_acute = "Acute WAB-AQ", lesion_volume_mL = "Lesion volume", age_at_stroke = "Age", sex = "Sex", education_yrs = "Education", prior_stroke = "Prior stroke",
                BPM_IFG_opercularis_L = "L pars opercularis", BPM_IFG_triangularis_L = "L pars triangularis", BPM_STG_L = "L STG", BPM_STG_L_pole = "L STG pole", BPM_MTG_L = "L MTG", BPM_SMG_L = "L SMG", BPM_AG_L = "L angular gyrus", BPM_Ins_L = "L insula", BPM_FuG_L = "L fusiform", BPM_MFG_L = "L MFG",
                SLF_L_prob = "SLF", SLFt_L_prob = "Arcuate (SLFt)", IFOF_L_prob = "IFOF", ILF_L_prob = "ILF", UF_L_prob = "UF")
  nice <- function(f) ifelse(f %in% names(nice_map), nice_map[f], str_replace(f, "^pair_", ""))
  modality <- function(f) case_when(f %in% c("wab_aq_acute", "age_at_stroke", "sex", "education_yrs", "prior_stroke") ~ "Clinical", f == "lesion_volume_mL" ~ "Volume", str_starts(f, "BPM_") ~ "Cortical", str_starts(f, "pair_") ~ "Network", TRUE ~ "WM")
  p_shap <- function(oc, panel) {
    d <- sh %>% filter(outcome == oc) %>% arrange(desc(mean_abs_shap)) %>% head(15) %>%
      mutate(feat = fct_reorder(nice(feature), mean_abs_shap), mod = modality(feature), retention = replace_na(retention, 0))
    top <- max(d$mean_abs_shap); second <- sort(d$mean_abs_shap, decreasing = TRUE)[2]
    truncated <- top / second > 8
    mx <- if (truncated) second * 1.3 else top
    d <- d %>% mutate(bar = pmin(mean_abs_shap, mx), lab = ifelse(truncated & mean_abs_shap > mx, sprintf("%.1f (axis truncated)", mean_abs_shap), ""))
    ggplot(d, aes(y = feat)) + geom_col(aes(x = bar, fill = mod)) +
      geom_text(aes(x = mx * 0.98, label = lab), hjust = 1, size = 3, colour = "white", fontface = "bold") +
      geom_point(aes(x = retention * mx), shape = 21, fill = "white", size = 2.2) +
      scale_x_continuous(name = "Mean |SHAP| across 100 outer folds (bars)", limits = c(0, mx * 1.02), expand = expansion(mult = c(0, 0.01)),
                         sec.axis = sec_axis(~ . / mx, name = "RFE retention, proportion of folds (points)", breaks = c(0, 0.5, 1))) +
      scale_fill_manual(values = modality_colors, name = NULL) + labs(y = NULL, title = panel) + theme_marte() }
  fig5 <- p_shap("aphasia_resolution", "A. Aphasia resolution") + p_shap("discourse_content", "B. Discourse content") + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  save_both("fig5", fig5, 11, 6)
}
cat("done\n")
