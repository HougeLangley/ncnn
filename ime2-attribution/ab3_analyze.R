#!/usr/bin/env Rscript
# =============================================================================
# ab3_analyze.R -- three-arm attribution analysis
#
#   arm "off"  : upstream default (use_fp16_arithmetic=true -> fp16sa Gemm)
#   arm "gemm" : + Gemm IME2 only            == ncnn PR #7032
#   arm "both" : + Gemm IME2 + SDPA IME2     == PR #7032 + PR #7037
#
# Attribution:
#   effect(PR #7032, Gemm)  = gemm - off
#   effect(PR #7037, SDPA)  = both - gemm
#
# Input : results3/rep<NN>_<arm>.json   Output: results3/*.csv and figs3/*.png
# =============================================================================
suppressPackageStartupMessages({
  library(jsonlite); library(tidyverse); library(patchwork); library(scales)
})

ROOT <- "/home/ubuntu/k3-ncnn/bench-rebase"
RES  <- file.path(ROOT, "results3")
FIG  <- file.path(ROOT, "figs3")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

ARM_LEVELS <- c("off", "gemm", "both")
ARM_LABELS <- c(off = "off (upstream fp16sa)", gemm = "+Gemm IME2 (#7032)", both = "+Gemm+SDPA (#7037)")

files <- list.files(RES, pattern = "^rep[0-9]+_(off|gemm|both)\\.json$", full.names = TRUE)
if (!length(files)) stop("no json in ", RES)
cat("found", length(files), "result files\n")

`%||%` <- function(a, b) if (is.null(a)) b else a

parse_one <- function(f) {
  base <- basename(f)
  rep  <- as.integer(sub("^rep([0-9]+)_.*$", "\\1", base))
  arm  <- sub("^rep[0-9]+_(off|gemm|both)\\.json$", "\\1", base)
  j    <- fromJSON(f, simplifyVector = FALSE)
  rows <- list()
  for (b in j$benchmarks %||% list()) {
    add <- function(metric, kind) {
      m <- b[[metric]]
      if (is.null(m) || is.null(m$values)) return(invisible(NULL))
      vals <- as.numeric(unlist(m$values))
      if (!length(vals)) return(invisible(NULL))
      rows[[length(rows) + 1]] <<- tibble(
        rep = rep, arm = arm, kind = kind,
        prompt_size = as.integer(b$prompt_size %||% NA),
        response_size = as.integer(b$response_size %||% NA),
        run = seq_along(vals), value = vals)
      invisible(NULL)
    }
    add("pp_throughput", "prefill")
    add("tg_throughput", "decode")
  }
  if (!length(rows)) return(NULL)
  bind_rows(rows)
}

tidy <- map_dfr(files, parse_one) %>%
  mutate(test = if_else(kind == "prefill", paste0("pp", prompt_size), paste0("tg", response_size)),
         arm  = factor(arm, levels = ARM_LEVELS))
write_csv(tidy, file.path(RES, "tidy_runs.csv"))

cat("\n--- measurements per test x arm ---\n")
print(tidy %>% count(kind, test, arm) %>% pivot_wider(names_from = arm, values_from = n))

reps <- tidy %>% group_by(test, kind, arm, rep) %>%
  summarise(rmean = mean(value), rsd = sd(value), n = dplyr::n(), .groups = "drop")
write_csv(reps, file.path(RES, "tidy_reps.csv"))

summ <- reps %>% group_by(test, kind, arm) %>%
  summarise(n_rep = dplyr::n(), mu = mean(rmean), sdev = sd(rmean), .groups = "drop") %>%
  mutate(se = sdev / sqrt(n_rep),
         ci_lo = mu - qt(0.975, n_rep - 1) * se,
         ci_hi = mu + qt(0.975, n_rep - 1) * se)
write_csv(summ, file.path(RES, "summary.csv"))
cat("\n--- summary (tokens/s, replicate means) ---\n")
print(as.data.frame(summ %>% select(test, arm, n_rep, mu, sdev, ci_lo, ci_hi)), digits = 5, row.names = FALSE)

# ---- paired comparisons between arms -------------------------------------
wide <- reps %>% select(test, kind, arm, rep, rmean) %>%
  pivot_wider(names_from = arm, values_from = rmean)

paired_test <- function(d, a, b) {
  x <- d[[a]]; y <- d[[b]]
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]; n <- length(x)
  if (n < 2) return(tibble(n = n))
  tt <- t.test(x, y, paired = TRUE)
  wt <- suppressWarnings(wilcox.test(x, y, paired = TRUE, exact = FALSE))
  tibble(n = n, base_mean = mean(y), new_mean = mean(x),
         delta = mean(x - y), pct_change = 100 * (mean(x) - mean(y)) / mean(y),
         sd_diff = sd(x - y), se_diff = sd(x - y) / sqrt(n),
         ci_lo = mean(x - y) - qt(0.975, n - 1) * sd(x - y) / sqrt(n),
         ci_hi = mean(x - y) + qt(0.975, n - 1) * sd(x - y) / sqrt(n),
         t_p = tt$p.value, wilcox_p = wt$p.value,
         cohen_dz = mean(x - y) / sd(x - y))
}

comparisons <- list(
  "PR #7032 (Gemm IME2)"        = c("gemm", "off"),
  "PR #7037 (SDPA IME2)"        = c("both", "gemm"),
  "total (both vs off)"         = c("both", "off")
)

tests <- map_dfr(names(comparisons), function(nm) {
  ab <- comparisons[[nm]]
  wide %>% group_by(test, kind) %>%
    group_modify(~ paired_test(.x, ab[1], ab[2])) %>%
    ungroup() %>% mutate(contrast = nm, .before = 1)
}) %>% group_by(contrast) %>%
  mutate(t_p_bh = p.adjust(t_p, method = "BH")) %>% ungroup()

write_csv(tests, file.path(RES, "paired_tests.csv"))
cat("\n--- paired contrasts ---\n")
print(as.data.frame(tests %>% select(contrast, test, n, base_mean, new_mean, pct_change,
                                     ci_lo, ci_hi, t_p, t_p_bh, cohen_dz)),
      digits = 4, row.names = FALSE)

# ---------------------------------------------------------------- plots
pd <- position_dodge(width = 0.8)
p1 <- ggplot(summ, aes(test, mu, fill = arm)) +
  geom_col(position = pd, width = 0.72, colour = "grey25", linewidth = 0.25) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), position = pd, width = 0.2, linewidth = 0.4) +
  geom_point(data = reps, aes(test, rmean, group = arm), position = position_dodge(width = 0.8),
             size = 1.0, alpha = 0.5, colour = "black", shape = 16, inherit.aes = FALSE) +
  facet_wrap(~ kind, scales = "free") +
  scale_fill_manual(values = c("off" = "#B0BEC5", "gemm" = "#66BB6A", "both" = "#1E88E5"),
                    labels = ARM_LABELS) +
  labs(title = "SpacemiT K3 A100: attributing the gain between two separate PRs",
       subtitle = sprintf("Qwen3-0.6B | threads=8 pinned cpu8-15 | %d replicates per arm, fresh process each; bars=mean, whiskers=95%% CI, dots=replicate means", max(reps$rep)),
       x = NULL, y = "throughput (tokens/s)", fill = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.subtitle = element_text(size = 8))

p2 <- ggplot(tests, aes(pct_change, test, fill = contrast)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72,
           colour = "grey25", linewidth = 0.25) +
  geom_vline(xintercept = 0, linewidth = 0.4) +
  geom_text(aes(label = sprintf("%+.0f%%", pct_change)),
            position = position_dodge(width = 0.8), hjust = -0.12, size = 2.9) +
  scale_fill_manual(values = c("PR #7032 (Gemm IME2)" = "#66BB6A",
                               "PR #7037 (SDPA IME2)" = "#1E88E5",
                               "total (both vs off)"  = "#8E24AA")) +
  labs(title = "Attribution: which PR produced which gain",
       subtitle = "paired difference vs the preceding arm, 5 replicates",
       x = "change in throughput (%)", y = NULL, fill = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position = "top", panel.grid.minor = element_blank())

ggsave_safe <- function(file, plot, w, h) {
  tryCatch({ ggsave(file, plot, width = w, height = h, dpi = 300); cat("wrote", basename(file), "\n") },
           error = function(e) cat("PLOT FAILED:", basename(file), "-", conditionMessage(e), "\n"))
}
ggsave_safe(file.path(FIG, "attr_bars.png"),     p1, 10, 5)
ggsave_safe(file.path(FIG, "attr_attrib.png"),   p2, 9,  4.5)
ggsave_safe(file.path(FIG, "attr_combined.png"), (p1 / p2) + plot_layout(heights = c(1, 1)), 10, 9.5)
cat("\nwrote figures to", FIG, "\n")
