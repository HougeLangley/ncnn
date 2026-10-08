#!/usr/bin/env Rscript
# Deep dive on the Gemm fp16/fp32 mismatch (Tencent/ncnn#7050, fix #7051).
#
# Question: what does fixing this bug actually buy, on a SpacemiT K3?
# Two things can be measured, and they must not be conflated:
#   (A) robustness  -- the failure mode before vs after the guard
#   (B) performance -- what the guard costs, and the ceiling a *deeper* fix
#                      could reach (fp16 pipeline vs fp32 on the same shapes)
suppressPackageStartupMessages({ library(tidyverse); library(patchwork); library(scales) })
ROOT <- "/home/ubuntu/k3-ncnn/bench-rebase"
OUT  <- file.path(ROOT, "deepdive"); FIG <- file.path(OUT, "figs")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

wilson <- function(k, n, z = 1.959964) {
  p <- k / n
  d <- 1 + z^2 / n
  ctr <- (p + z^2 / (2 * n)) / d
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
  tibble(est = p, lo = pmax(0, ctr - half), hi = pmin(1, ctr + half))
}

# ---------------------------------------------------------------- (A) sweep --
sw <- read_csv(file.path(OUT, "sweep.csv"), show_col_types = FALSE) %>%
  mutate(build = factor(build, levels = c("orig", "fixed"),
                        labels = c("unpatched", "with guard")),
         verdict = factor(verdict,
                          levels = c("correct", "silent_wrong", "crash", "clean_error"),
                          labels = c("correct", "silently wrong (exit 0)", "SIGSEGV crash", "clean error return")))
cat("== sweep: ", nrow(sw), "runs over", n_distinct(paste(sw$M, sw$N, sw$K)), "distinct shapes ==\n")
cat("fp32 reference self-check (must all be 'yes'):\n"); print(table(sw$fp32_ok))

dist <- sw %>% group_by(build, verdict) %>%
  summarise(k = dplyr::n(), .groups = "drop") %>%
  group_by(build) %>% mutate(n = sum(k), pct = 100 * k / n) %>% ungroup() %>%
  bind_cols(wilson(.$k, .$n) %>% select(lo, hi)) %>%
  mutate(ci = sprintf("%.1f%% [%.1f, %.1f]", pct, 100 * lo, 100 * hi))
cat("\n--- outcome distribution ---\n")
print(as.data.frame(dist %>% select(build, verdict, k, n, ci)), row.names = FALSE)
write_csv(dist, file.path(OUT, "outcome_distribution.csv"))

# the two quantities that matter for a caller
before <- sw %>% filter(build == "unpatched")
after  <- sw %>% filter(build == "with guard")
N_TOT <- nrow(before)   # NB: must be captured outside tibble(), which would
                        # resolve `before` to the column defined above it
tab <- tibble(
  metric = c("returned correct numbers",
             "failed in a way the caller cannot detect (exit 0, wrong values)",
             "failed loudly (crash or error return)",
             "produced a diagnostic naming the cause"),
  before = c(sum(before$verdict == "correct"),
             sum(before$verdict == "silently wrong (exit 0)"),
             sum(before$verdict %in% c("SIGSEGV crash", "clean error return")),
             0L),
  after  = c(sum(after$verdict == "correct"),
             sum(after$verdict == "silently wrong (exit 0)"),
             sum(after$verdict %in% c("SIGSEGV crash", "clean error return")),
             sum(after$logged == "yes")),
  n = N_TOT)
wb <- wilson(tab$before, tab$n); wa <- wilson(tab$after, tab$n)
tab$before_lo <- wb$lo; tab$before_hi <- wb$hi
tab$after_lo  <- wa$lo; tab$after_hi  <- wa$hi
tab <- tab %>%
  mutate(before_pct = 100 * before / n, after_pct = 100 * after / n,
         before_ci = sprintf("%.1f%% [%.1f, %.1f]", before_pct, 100 * before_lo, 100 * before_hi),
         after_ci  = sprintf("%.1f%% [%.1f, %.1f]", after_pct,  100 * after_lo,  100 * after_hi),
         fisher_p = map_dbl(seq_len(dplyr::n()), function(i) {
           if (before[i] == after[i]) return(NA_real_)   # identical, nothing to test
           fisher.test(matrix(c(before[i], n - before[i], after[i], n - after[i]), nrow = 2),
                       simulate.p.value = TRUE, B = 200000)$p.value }))
cat("\n--- what the caller sees, before vs after (n =", nrow(before), "shapes) ---\n")
print(as.data.frame(tab %>% select(metric, before_ci, after_ci, fisher_p)), row.names = FALSE)
write_csv(tab, file.path(OUT, "caller_impact.csv"))

# does the failure mode depend on the shape?
byM <- sw %>% group_by(build, M) %>%
  summarise(n = dplyr::n(), silent = sum(verdict == "silently wrong (exit 0)"),
            crash = sum(verdict == "SIGSEGV crash"), .groups = "drop") %>%
  mutate(pct_silent = 100 * silent / n)
byK <- sw %>% group_by(build, K) %>%
  summarise(n = dplyr::n(), silent = sum(verdict == "silently wrong (exit 0)"),
            crash = sum(verdict == "SIGSEGV crash"), .groups = "drop") %>%
  mutate(pct_silent = 100 * silent / n)
cat("\n--- failure mode by M (unpatched) ---\n"); print(as.data.frame(byM %>% filter(build == "unpatched")))
cat("\n--- failure mode by K (unpatched) ---\n"); print(as.data.frame(byK %>% filter(build == "unpatched")))
write_csv(byM, file.path(OUT, "by_M.csv")); write_csv(byK, file.path(OUT, "by_K.csv"))

# M = 1, K = 1024: the clean monotone boundary
bnd <- sw %>% filter(build == "unpatched", M == 1, K == 1024) %>% arrange(N) %>%
  mutate(x = factor(N, levels = N))
cat("\n--- M=1 K=1024 boundary (unpatched) ---\n")
print(as.data.frame(bnd %>% select(N, verdict)))

# ------------------------------------------------------- (B) performance ----
# The guard itself: paired A/B already measured in fixbench/
fb <- read_csv(file.path(ROOT, "fixbench", "raw.csv"), show_col_types = FALSE) %>%
  mutate(key = paste0(mode, " M=", sub(" .*", "", shape), " N=", sub("^\\S+ (\\S+).*", "\\1", shape)))
reps <- fb %>% group_by(key, mode, variant, rep) %>% summarise(ms = mean(ms), .groups = "drop")
# The ceiling a deeper fix could reach: fp16 pipeline vs fp32, same shape/build.
# NB: `key` above carries the mode prefix, so pivot on a shape-only key instead.
ceil <- fb %>% filter(variant == "orig") %>%
  mutate(skey = paste0("M=", sub(" .*", "", shape), " N=", sub("^\\S+ (\\S+).*", "\\1", shape)),
         m2 = ifelse(mode == "fp16in", "fp16", "fp32")) %>%
  group_by(skey, m2, rep) %>% summarise(ms = mean(ms), .groups = "drop") %>%
  pivot_wider(names_from = m2, values_from = ms) %>%
  group_by(skey) %>%
  summarise(n = dplyr::n(), fp32_ms = mean(fp32), fp16_ms = mean(fp16),
            speedup = mean(fp32) / mean(fp16),
            sp_lo = min(fp32 / fp16), sp_hi = max(fp32 / fp16), .groups = "drop") %>%
  mutate(saving_pct = 100 * (1 - 1 / speedup))
cat("\n--- performance ceiling: fp16 pipeline vs fp32 on the same shape ---\n")
print(as.data.frame(ceil), digits = 4, row.names = FALSE)
write_csv(ceil, file.path(OUT, "fp16_vs_fp32_ceiling.csv"))

# ------------------------------------------------------------------ figures --
p1 <- ggplot(dist, aes(build, pct, fill = verdict)) +
  geom_col(width = 0.62, colour = "grey20", linewidth = 0.3) +
  geom_text(aes(label = ifelse(k > 0, sprintf("%d (%.0f%%)", k, pct), "")),
            position = position_stack(vjust = 0.5), size = 3, colour = "white", fontface = "bold") +
  scale_fill_manual(values = c("correct" = "#43A047", "silently wrong (exit 0)" = "#FB8C00",
                               "SIGSEGV crash" = "#E53935", "clean error return" = "#1E88E5")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "What a bare Gemm does with an fp32 input while use_fp16_storage is on",
       subtitle = sprintf("SpacemiT K3 A100 cluster; %d shapes (M in {1,32}, K in {256,1024,2048}, N 1024..151936); one run each", nrow(before)),
       x = NULL, y = "share of shapes (%)", fill = NULL) +
  theme_bw(base_size = 11) + theme(legend.position = "top", panel.grid.minor = element_blank())

p2 <- ggplot(bnd, aes(x, 1, fill = verdict)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = N), size = 2.4, angle = 90, colour = "white") +
  scale_fill_manual(values = c("correct" = "#43A047", "silently wrong (exit 0)" = "#FB8C00",
                               "SIGSEGV crash" = "#E53935", "clean error return" = "#1E88E5")) +
  labs(title = "Failure mode is decided by the allocator, not by the maths",
       subtitle = "M=1, K=1024, N swept; small allocations land on the heap and return garbage, large ones get a guard page and crash",
       x = "N (vocabulary size)", y = NULL, fill = NULL) +
  theme_bw(base_size = 11) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        panel.grid = element_blank(), legend.position = "top",
        axis.text.x = element_text(angle = 60, hjust = 1, size = 6.5))

p3 <- ceil %>% mutate(skey = fct_reorder(skey, speedup)) %>%
  ggplot(aes(speedup, skey, fill = speedup)) +
  geom_col(width = 0.6, colour = "grey20", linewidth = 0.3) +
  geom_errorbar(aes(xmin = sp_lo, xmax = sp_hi), orientation = "y", width = 0.15, linewidth = 0.4) +
  geom_vline(xintercept = 1, linewidth = 0.4, linetype = "dashed") +
  geom_text(aes(label = sprintf("%.2fx", speedup)), hjust = -0.2, size = 3) +
  scale_fill_gradient(low = "#90CAF9", high = "#0D47A1", guide = "none") +
  coord_cartesian(xlim = c(1, max(ceil$sp_hi) * 1.16)) +   # zoom to the ratios; bars still start at 0
  labs(title = "Performance ceiling a deeper fix could reach (NOT what #7051 does)",
       subtitle = "Gemm latency, fp16 pipeline vs fp32, same build and shape; range over the 5 paired replicates",
       x = "fp32 latency / fp16 latency", y = NULL) +
  theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank())

save_safe <- function(f, p, w, h) tryCatch(ggsave(f, p, width = w, height = h, dpi = 300),
                                          error = function(e) cat("FAILED", f, conditionMessage(e), "\n"))
save_safe(file.path(FIG, "01_failure_modes.png"), p1, 9, 4.6)
save_safe(file.path(FIG, "02_boundary.png"),      p2, 11, 3.6)
save_safe(file.path(FIG, "03_ceiling.png"),       p3, 9, 4.2)
save_safe(file.path(FIG, "00_combined.png"), (p1 / p2 / p3), 11, 13)
cat("\nwrote figures to", FIG, "\n")
