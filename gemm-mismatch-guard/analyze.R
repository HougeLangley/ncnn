#!/usr/bin/env Rscript
# Paired analysis: does the added elembits guard change performance?
#   variant "orig"  = libncnn built from unpatched master (16bdc2d)
#   variant "fixed" = same tree + the Gemm fp16/fp32 mismatch guard
# Both binaries are the same program, same shapes, same cluster; only the
# library differs. Order alternates per replicate.
suppressPackageStartupMessages({ library(tidyverse); library(patchwork) })
ROOT <- "/home/ubuntu/k3-ncnn/bench-rebase/fixbench"
FIG <- file.path(ROOT, "figs"); dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

raw <- read_csv(file.path(ROOT, "raw.csv"), show_col_types = FALSE) %>%
  mutate(variant = factor(variant, levels = c("orig", "fixed"),
                          labels = c("unpatched (16bdc2d)", "with mismatch guard")),
         key = paste0(mode, " M=", sub(" .*", "", shape), " N=", sub("^\\S+ (\\S+).*", "\\1", shape)))
cat("rows:", nrow(raw), " configs:", n_distinct(raw$key), "\n\n")

# per replicate (already best-of-5 inside the binary); reps are the unit
reps <- raw %>% group_by(key, mode, variant, rep) %>%
  summarise(ms = mean(ms), gflops = mean(gflops), .groups = "drop")

summ <- reps %>% group_by(key, mode, variant) %>%
  summarise(n = dplyr::n(), mu = mean(ms), sdev = sd(ms), .groups = "drop") %>%
  mutate(se = sdev / sqrt(n),
         ci_lo = mu - qt(0.975, n - 1) * se,
         ci_hi = mu + qt(0.975, n - 1) * se)
cat("--- latency, ms (mean of replicate means) ---\n")
print(as.data.frame(summ %>% select(key, variant, n, mu, sdev, ci_lo, ci_hi)), digits = 4, row.names = FALSE)

wide <- reps %>% select(key, mode, variant, rep, ms) %>%
  pivot_wider(names_from = variant, values_from = ms)
tests <- wide %>% group_by(key, mode) %>%
  group_modify(function(d, ...) {
    a <- d[["unpatched (16bdc2d)"]]; b <- d[["with mismatch guard"]]
    ok <- is.finite(a) & is.finite(b); a <- a[ok]; b <- b[ok]; n <- length(a)
    tt <- t.test(b, a, paired = TRUE)
    tibble(n = n, unpatched_ms = mean(a), fixed_ms = mean(b),
           delta_ms = mean(b - a), pct = 100 * (mean(b) - mean(a)) / mean(a),
           ci_lo = mean(b - a) - qt(0.975, n - 1) * sd(b - a) / sqrt(n),
           ci_hi = mean(b - a) + qt(0.975, n - 1) * sd(b - a) / sqrt(n),
           t_p = tt$p.value, dz = mean(b - a) / sd(b - a))
  }) %>% ungroup() %>% mutate(t_p_bh = p.adjust(t_p, method = "BH"))

cat("\n--- paired difference (guard - unpatched), latency in ms ---\n")
print(as.data.frame(tests %>% select(key, mode, n, unpatched_ms, fixed_ms, delta_ms, pct,
                                     ci_lo, ci_hi, t_p_bh, dz)), digits = 4, row.names = FALSE)
write_csv(summ, file.path(ROOT, "summary.csv"))
write_csv(tests, file.path(ROOT, "paired_tests.csv"))
write_csv(reps,  file.path(ROOT, "tidy_reps.csv"))

p1 <- ggplot(summ, aes(key, mu, fill = variant)) +
  geom_col(position = position_dodge(0.75), width = 0.68, colour = "grey25", linewidth = 0.25) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), position = position_dodge(0.75), width = 0.2, linewidth = 0.4) +
  geom_point(data = reps, aes(key, ms, group = variant), position = position_dodge(0.75),
             size = 1.0, alpha = 0.5, colour = "black", shape = 16, inherit.aes = FALSE) +
  scale_fill_manual(values = c("unpatched (16bdc2d)" = "#B0BEC5", "with mismatch guard" = "#1E88E5")) +
  labs(title = "Cost of the Gemm fp16/fp32 mismatch guard",
       subtitle = "SpacemiT K3 A100 cluster, 5 replicates, fresh process each; bars = mean latency, whiskers = 95% CI, dots = replicate means",
       x = NULL, y = "forward() latency (ms, lower is better)", fill = NULL) +
  theme_bw(base_size = 10) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 20, hjust = 1),
        plot.subtitle = element_text(size = 7.5))

p2 <- ggplot(tests, aes(pct, key, fill = pct > 0)) +
  geom_col(width = 0.6, colour = "grey25", linewidth = 0.25) +
  geom_errorbar(aes(xmin = 100 * ci_lo / unpatched_ms, xmax = 100 * ci_hi / unpatched_ms),
                orientation = "y", width = 0.16, linewidth = 0.4) +
  geom_vline(xintercept = 0, linewidth = 0.4) +
  geom_text(aes(label = sprintf("%+.2f%%", pct)), hjust = ifelse(tests$pct > 0, -0.15, 1.15), size = 2.8) +
  scale_fill_manual(values = c(`TRUE` = "#E53935", `FALSE` = "#66BB6A"), guide = "none") +
  labs(title = "Relative latency change from the guard (guarded vs unpatched)",
       subtitle = "paired difference with 95% CI; none significant after Benjamini-Hochberg correction (all p >= 0.13)",
       x = "latency change (%)", y = NULL) +
  theme_bw(base_size = 10) + theme(panel.grid.minor = element_blank())

ggsave(file.path(FIG, "guard_cost.png"), p1, width = 10, height = 5, dpi = 300)
ggsave(file.path(FIG, "guard_cost_delta.png"), p2, width = 8, height = 4, dpi = 300)
ggsave(file.path(FIG, "guard_cost_combined.png"), (p1 / p2), width = 10, height = 9, dpi = 300)
cat("\nwrote figures to", FIG, "\n")
