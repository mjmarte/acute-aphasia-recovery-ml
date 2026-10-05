
suppressPackageStartupMessages({
  library(tidyverse); library(ggseg); library(patchwork)
  library(viridis); library(glue); library(sf); library(ggrepel)
})

OUT <- "panels"
shap <- read_csv("shap/shap_rank_summary.csv", show_col_types = FALSE)

bpm_to_dk <- tribble(
  ~feature,                    ~label,
  "BPM_IFG_opercularis_L",    "lh_parsopercularis",
  "BPM_IFG_triangularis_L",   "lh_parstriangularis",
  "BPM_STG_L",                "lh_superiortemporal",
  "BPM_STG_L_pole",           "lh_temporalpole",
  "BPM_MTG_L",                "lh_middletemporal",
  "BPM_SMG_L",                "lh_supramarginal",
  "BPM_AG_L",                 "lh_inferiorparietal",
  "BPM_Ins_L",                "lh_insula",
  "BPM_FuG_L",                "lh_fusiform",
  "BPM_MFG_L",                "lh_caudalmiddlefrontal",
)

pair_node_to_dk <- tribble(
  ~code,     ~label,
  "POP",     "lh_parsopercularis",
  "PTR",     "lh_parstriangularis",
  "STG",     "lh_superiortemporal",
  "MTG",     "lh_middletemporal",
  "SMG",     "lh_supramarginal",
  "IPG",     "lh_inferiorparietal",
  "ITG",     "lh_inferiortemporal",
  "IN",      "lh_insula",
  "PrCG",    "lh_precentral",
  "SFG",     "lh_superiorfrontal",
  "FG",      "lh_fusiform",
)

dk_nice <- c(
  "lh_parsopercularis"     = "L. Pars Oper.",
  "lh_parstriangularis"    = "L. Pars Tri.",
  "lh_superiortemporal"    = "L. Sup. Temp.",
  "lh_temporalpole"        = "L. Temp. Pole",
  "lh_middletemporal"      = "L. Mid. Temp.",
  "lh_supramarginal"       = "L. Supramarg.",
  "lh_inferiorparietal"    = "L. Inf. Par.",
  "lh_insula"              = "L. Insula",
  "lh_fusiform"            = "L. Fusiform",
  "lh_caudalmiddlefrontal" = "L. Mid. Front.",
  "lh_inferiortemporal"    = "L. Inf. Temp.",
  "lh_precentral"          = "L. Precentral",
  "lh_superiorfrontal"     = "L. Sup. Front."
)

aggregate_shap <- function(outcome, fs_val, model_val) {
  shap %>%
    filter(outcome_label == outcome, fs == fs_val, model == model_val) %>%
    inner_join(bpm_to_dk, by = "feature") %>%
    select(label, shap = mean_abs_shap)
}

lh_lat <- dk()
lh_lat$data$sf <- lh_lat$data$sf %>%
  filter(grepl("^lh_", label), view == "lateral")

WAB_VMAX <- as.numeric(readLines(glue("{OUT}/wab_max_shap.txt")))
NCT_VMAX <- as.numeric(readLines(glue("{OUT}/nct_max_shap.txt")))

make_fill <- function(vmax) {
  scale_fill_viridis_c(
    option = "inferno", direction = -1,
    name = "Mean |SHAP|", na.value = "gray92",
    limits = c(0, vmax),
    breaks = c(0, vmax/2, vmax),
    labels = c("0.0", format(round(vmax/2,2), nsmall=2), format(vmax, nsmall=1)),
    guide = guide_colorbar(frame.colour="black", frame.linewidth=0.8,
                           ticks.colour="black",
                           barwidth=unit(0.4,"cm"), barheight=unit(2.0,"cm"),
                           title.position="top", title.hjust=0.5))
}

get_centroids <- function(labels_vec, sample_data) {
  p_dummy <- ggplot(sample_data) +
    geom_brain(atlas = lh_lat, mapping = aes(fill = shap))
  built <- ggplot_build(p_dummy)$data[[1]]
  built_sf <- sf::st_as_sf(built[, c("label", "geometry")])

  built_sf %>%
    filter(label %in% labels_vec, !sf::st_is_empty(geometry)) %>%
    group_by(label) %>%
    summarise(geometry = sf::st_union(geometry), .groups = "drop") %>%
    mutate(centroid = sf::st_centroid(geometry),
           cx = sf::st_coordinates(centroid)[, 1],
           cy = sf::st_coordinates(centroid)[, 2]) %>%
    sf::st_drop_geometry() %>% select(label, cx, cy)
}

build_top3 <- function(shap_dk, fill_scale, vmax, title_str) {
  top3 <- shap_dk %>%
    filter(grepl("^lh_", label)) %>%
    arrange(desc(shap)) %>% slice_head(n = 3) %>%
    mutate(rank = row_number(),
           nice = coalesce(dk_nice[label], label))

  centroids <- get_centroids(top3$label, shap_dk)
  top3 <- top3 %>% inner_join(centroids, by = "label")

  pal <- viridis::viridis(256, option = "inferno", direction = -1)
  top3 <- top3 %>% mutate(
    line_col = pal[pmin(pmax(round(shap / vmax * 255) + 1, 1), 256)]
  )

  p <- ggplot(shap_dk) +
    geom_brain(atlas = lh_lat, mapping = aes(fill = shap),
               colour = "gray40", size = 0.3) +
    fill_scale +
    ggrepel::geom_label_repel(
      data = top3,
      aes(x = cx, y = cy, label = paste0(rank, ". ", nice)),
      size = 2.8, fontface = "bold", family = "Helvetica",
      colour = "gray15", fill = alpha("white", 0.95),
      label.padding = unit(0.18, "lines"),
      segment.colour = top3$line_col,
      segment.size = 1.0,
      min.segment.length = 0,
      box.padding = 0.6, point.padding = 0.4,
      force = 8, max.overlaps = Inf,
      nudge_x = 250, nudge_y = 0,
      direction = "y",
      seed = 42,
      inherit.aes = FALSE) +
    theme_void(base_family = "Helvetica") +
    theme(legend.position = "right",
          plot.margin = margin(6, 6, 6, 6),
          plot.background = element_rect(fill = "white", colour = NA),
          plot.title = element_text(face = "bold", size = 10, hjust = 0.5)) +
    labs(title = title_str)

  p
}

cat("--- Top-3 cortical (ggseg brain + centroids for the shared labeling routine) ---\n")
export_brain <- function(shap_dk, vmax, stem) {
  top3 <- shap_dk %>% filter(grepl("^lh_", label)) %>% arrange(desc(shap)) %>% slice_head(n = 3) %>%
    mutate(rank = row_number(), nice = coalesce(dk_nice[label], label))
  top3 <- top3 %>% inner_join(get_centroids(top3$label, shap_dk), by = "label")
  p <- ggplot(shap_dk) + geom_brain(atlas = lh_lat, mapping = aes(fill = shap), colour = "gray40", size = 0.3) +
    scale_fill_viridis_c(option = "inferno", direction = -1, na.value = "gray92", limits = c(0, vmax), guide = "none") +
    theme_void() + theme(plot.margin = margin(0, 0, 0, 0), plot.background = element_rect(fill = "white", colour = NA))
  W_IN <- 4.0; b <- ggplot_build(p)$layout$panel_params[[1]]
  xr <- b$x_range; yr <- b$y_range; H_IN <- W_IN * diff(yr) / diff(xr)
  ggsave(glue("{OUT}/{stem}_brain.png"), p, width = W_IN, height = H_IN, dpi = 600)
  wpx <- round(W_IN * 600); hpx <- round(H_IN * 600)
  top3 %>% mutate(px = (cx - xr[1]) / diff(xr) * wpx, py = (1 - (cy - yr[1]) / diff(yr)) * hpx) %>%
    select(rank, label, nice, shap, px, py) %>% write_csv(glue("{OUT}/{stem}_centroids.csv"))
  cat(glue("  Saved {stem}_brain.png and {stem}_centroids.csv ({wpx} x {hpx} px)"), "\n")
}
export_brain(aggregate_shap("aphasia_resolution", "FS4", "rf"), WAB_VMAX, "top3_cortical_wab")
export_brain(aggregate_shap("discourse_content",  "FS4", "svr"), NCT_VMAX, "top3_cortical_nct")
