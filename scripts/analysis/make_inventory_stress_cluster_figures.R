#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: make_inventory_stress_cluster_figures.R EARLY_TABLE DETAIL_DIR FIGURE_DIR")
}

early_path <- normalizePath(args[[1]], mustWork = TRUE)
detail_dir <- normalizePath(args[[2]], mustWork = TRUE)
figure_dir <- args[[3]]
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

early <- read.csv(early_path)
recovery <- read.csv(file.path(detail_dir, "cluster_recovery_summary.csv"))
combined <- merge(
  early,
  recovery,
  by = c("risk_limit_per_asset", "cluster_id"),
  all.x = TRUE
)
write.csv(combined, file.path(detail_dir, "cluster_financial_summary.csv"), row.names = FALSE)

correlations <- do.call(
  rbind,
  lapply(c(800, 1600), function(risk) {
    subset_data <- combined[combined$risk_limit_per_asset == risk, ]
    depth_test <- suppressWarnings(cor.test(
      subset_data$relative_depth_effect_percent_mean,
      subset_data$baseline_mean_top_depth,
      method = "spearman",
      exact = FALSE
    ))
    spread_test <- suppressWarnings(cor.test(
      subset_data$relative_depth_effect_percent_mean,
      subset_data$baseline_mean_spread_bps,
      method = "spearman",
      exact = FALSE
    ))
    data.frame(
      risk_limit_per_asset = risk,
      predictor = c("baseline top depth", "baseline spread"),
      spearman_rho = c(unname(depth_test$estimate), unname(spread_test$estimate)),
      p_value = c(depth_test$p.value, spread_test$p.value),
      cluster_count = nrow(subset_data)
    )
  })
)
write.csv(
  correlations,
  file.path(detail_dir, "cluster_characteristic_correlations.csv"),
  row.names = FALSE
)

palette <- c("800" = "#B33A3A", "1600" = "#2468A2")
labels <- c("800" = "Capacity 800", "1600" = "Capacity 1,600")
combined$risk <- factor(combined$risk_limit_per_asset)
combined$cluster_label <- as.character(combined$cluster_id)

theme_thesis <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "#E1E1E1", linewidth = 0.35),
    axis.text = element_text(colour = "#333333"),
    axis.title = element_text(colour = "#222222"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(colour = "#222222"),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, colour = "#444444"),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    legend.background = element_rect(fill = "white", colour = NA)
  )

save_plot <- function(plot, stem, width, height) {
  ggsave(
    file.path(figure_dir, paste0(stem, ".pdf")), plot,
    width = width, height = height, units = "in", device = "pdf", bg = "white"
  )
  ggsave(
    file.path(figure_dir, paste0(stem, ".png")), plot,
    width = width, height = height, units = "in", dpi = 300, bg = "white"
  )
}

p_sensitivity <- ggplot(
  combined,
  aes(
    x = baseline_mean_top_depth,
    y = relative_depth_effect_percent_mean,
    colour = risk,
    label = cluster_label
  )
) +
  geom_hline(yintercept = 0, colour = "#555555", linewidth = 0.35) +
  geom_errorbar(
    aes(
      ymin = relative_depth_effect_percent_lower,
      ymax = relative_depth_effect_percent_upper
    ),
    width = 0,
    linewidth = 0.5,
    alpha = 0.8
  ) +
  geom_point(size = 2.5) +
  geom_text(nudge_y = 0.012, size = 3.1, show.legend = FALSE) +
  scale_x_log10(labels = scales::comma) +
  scale_colour_manual(values = palette, labels = labels) +
  labs(
    x = "Baseline mean top depth (log scale)",
    y = "Mean depth deterioration, 2--10 s (%)",
    title = "Shallower liquidity clusters exhibit larger relative withdrawals",
    subtitle = "Labels identify clusters; error bars are paired 95% confidence intervals"
  ) +
  theme_thesis
save_plot(p_sensitivity, "cluster_depth_sensitivity", 7.2, 4.8)

recovery_plot_data <- combined[!is.na(combined$recovery_seconds), ]
p_recovery <- ggplot(
  recovery_plot_data,
  aes(x = factor(cluster_id), y = recovery_seconds, colour = risk, group = risk)
) +
  geom_linerange(
    aes(ymin = 0, ymax = recovery_seconds),
    position = position_dodge(width = 0.4),
    linewidth = 0.55,
    alpha = 0.7
  ) +
  geom_point(position = position_dodge(width = 0.4), size = 2.5) +
  scale_colour_manual(values = palette, labels = labels) +
  scale_y_continuous(breaks = seq(0, 40, 5), limits = c(0, 40)) +
  labs(
    x = "Liquidity-cluster identifier",
    y = "Initial recovery time (s)",
    title = "Initial recovery horizon by liquidity cluster",
    subtitle = paste0(
      "First point followed by five consecutive seconds whose paired interval includes zero;\n",
      "cluster 9 at capacity 1,600 had no significant initial episode"
    )
  ) +
  theme_thesis
save_plot(p_recovery, "cluster_recovery_time", 7.2, 5.0)

message("Detailed cluster figures written to ", normalizePath(figure_dir))
