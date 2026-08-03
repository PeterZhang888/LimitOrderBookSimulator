#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: make_inventory_stress_figures.R TABLE_DIR FIGURE_DIR")
}

table_dir <- normalizePath(args[[1]], mustWork = TRUE)
figure_dir <- args[[2]]
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

palette <- c("800" = "#B33A3A", "1600" = "#2468A2")
labels <- c("800" = "Capacity 800", "1600" = "Capacity 1,600")

theme_thesis <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "#E1E1E1", linewidth = 0.35),
    axis.title = element_text(colour = "#222222"),
    axis.text = element_text(colour = "#333333"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(colour = "#222222"),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, colour = "#444444"),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    legend.background = element_rect(fill = "white", colour = NA),
    strip.background = element_rect(fill = "white", colour = NA)
  )

save_plot <- function(plot, stem, width, height) {
  ggsave(file.path(figure_dir, paste0(stem, ".pdf")), plot,
         width = width, height = height, units = "in", device = "pdf", bg = "white")
  ggsave(file.path(figure_dir, paste0(stem, ".png")), plot,
         width = width, height = height, units = "in", dpi = 300, bg = "white")
}

mechanism <- read.csv(file.path(table_dir, "mechanism_checkpoint_summary.csv"))
mechanism$risk <- factor(mechanism$risk_limit_per_asset)

long_mechanism <- rbind(
  data.frame(
    risk = mechanism$risk,
    seconds = mechanism$post_shock_seconds,
    metric = "Shared gross exposure: shock - control",
    mean = mechanism$gross_exposure_delta_mean,
    lower = mechanism$gross_exposure_delta_lower,
    upper = mechanism$gross_exposure_delta_upper
  ),
  data.frame(
    risk = mechanism$risk,
    seconds = mechanism$post_shock_seconds,
    metric = "Shared quote scale: shock - control",
    mean = mechanism$quote_scale_delta_mean,
    lower = mechanism$quote_scale_delta_lower,
    upper = mechanism$quote_scale_delta_upper
  ),
  data.frame(
    risk = mechanism$risk,
    seconds = mechanism$post_shock_seconds,
    metric = "Unshocked requested depth: control - shock",
    mean = mechanism$requested_depth_reduction_mean,
    lower = mechanism$requested_depth_reduction_lower,
    upper = mechanism$requested_depth_reduction_upper
  )
)
long_mechanism$metric <- factor(
  long_mechanism$metric,
  levels = c(
    "Shared gross exposure: shock - control",
    "Shared quote scale: shock - control",
    "Unshocked requested depth: control - shock"
  )
)

p_mechanism <- ggplot(
  long_mechanism,
  aes(x = seconds, y = mean, colour = risk, fill = risk, group = risk)
) +
  geom_hline(yintercept = 0, colour = "#555555", linewidth = 0.35) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.14, colour = NA) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 1.6) +
  scale_x_log10(
    breaks = c(1, 2, 5, 10, 30, 300, 1800),
    labels = c("1", "2", "5", "10", "30", "300", "1,800")
  ) +
  scale_colour_manual(values = palette, labels = labels) +
  scale_fill_manual(values = palette, labels = labels) +
  facet_wrap(~metric, scales = "free_y", ncol = 1) +
  labs(
    x = "Seconds after the shock (log scale)",
    y = NULL,
    title = "Shared-dealer mechanism after the inventory-adverse shock",
    subtitle = "Means and paired 95% confidence intervals across 20 seeds"
  ) +
  theme_thesis
save_plot(p_mechanism, "mechanism_response", 7.2, 8.1)

time_data <- read.csv(file.path(table_dir, "marketwide_time_summary.csv"))
time_data$risk <- factor(time_data$risk_limit_per_asset)

make_depth_plot <- function(data, title, subtitle, stem, width, height) {
  plot <- ggplot(
    data,
    aes(
      x = post_shock_seconds,
      y = relative_depth_effect_percent_mean,
      colour = risk,
      fill = risk,
      group = risk
    )
  ) +
    geom_hline(yintercept = 0, colour = "#555555", linewidth = 0.35) +
    geom_ribbon(
      aes(
        ymin = relative_depth_effect_percent_lower,
        ymax = relative_depth_effect_percent_upper
      ),
      alpha = 0.13,
      colour = NA
    ) +
    geom_line(linewidth = 0.75) +
    scale_colour_manual(values = palette, labels = labels) +
    scale_fill_manual(values = palette, labels = labels) +
    labs(
      x = "Seconds after the shock",
      y = "Unshocked top-depth deterioration (%)",
      title = title,
      subtitle = subtitle
    ) +
    theme_thesis
  save_plot(plot, stem, width, height)
}

make_depth_plot(
  subset(time_data, post_shock_seconds <= 30),
  "Immediate cross-asset liquidity response",
  "Difference-in-differences; means and paired 95% confidence intervals across 20 seeds",
  "marketwide_depth_early",
  7.2,
  4.5
)

make_depth_plot(
  time_data,
  "Full post-shock depth response",
  "The immediate withdrawal is small relative to later stochastic dispersion",
  "marketwide_depth_full",
  7.2,
  4.5
)

clusters <- read.csv(file.path(table_dir, "cluster_early_response_summary.csv"))
clusters$risk <- factor(clusters$risk_limit_per_asset)
clusters$cluster <- factor(clusters$cluster_id)

p_clusters <- ggplot(
  clusters,
  aes(
    x = cluster,
    y = relative_depth_effect_percent_mean,
    colour = risk,
    group = risk
  )
) +
  geom_hline(yintercept = 0, colour = "#555555", linewidth = 0.35) +
  geom_errorbar(
    aes(
      ymin = relative_depth_effect_percent_lower,
      ymax = relative_depth_effect_percent_upper
    ),
    width = 0.16,
    position = position_dodge(width = 0.42),
    linewidth = 0.55
  ) +
  geom_point(position = position_dodge(width = 0.42), size = 2.1) +
  scale_colour_manual(values = palette, labels = labels) +
  labs(
    x = "Liquidity-cluster identifier",
    y = "Mean top-depth deterioration, 2--10 s (%)",
    title = "Early response by empirical liquidity cluster",
    subtitle = "Cluster identifiers are categories, not an ordered liquidity ranking; 95% paired intervals"
  ) +
  theme_thesis
save_plot(p_clusters, "cluster_early_response", 7.2, 4.7)

message("Figures written to ", normalizePath(figure_dir))
