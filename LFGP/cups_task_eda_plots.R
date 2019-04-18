library(R.matlab)
library(tidyverse)

# Data Parameters --------------------------------------------------
n_subjs <- 259
n_ROIs <- 350
n_vols <- 268
dir_dat <- "dat/time_series/cup/cleaned"
files_subj <- list.files(dir_dat)

# Read and Smooth fMRI Data ----------------------------------------
dat_array <- array(dim = c(n_subjs, n_ROIs, n_vols))

for (i in seq(files_subj)) {
  cat(i)
  subj <- files_subj[i]
  dat <- t(readMat(glue::glue("{dir_dat}/{subj}"))$TC.Data[, 1:n_ROIs])
  dat <- t(apply(dat, 1, function(x) scale(lm(x ~ c(1:n_vols))$residuals, center = TRUE, scale = TRUE)))
  global_mean <- colMeans(dat, na.rm = TRUE)
  dat[is.na(dat)] <- 0
  dat <- t(apply(dat, 1, function(x) scale(lm(x ~ global_mean)$residuals, center = TRUE, scale = TRUE)))
  dat_array[i, , ] <- t(apply(dat, 1, stats::filter, c(0.1, 0.2, 0.4, 0.20, 1)))
}
dat_array <- dat_array[, , 3:(n_vols - 2)]

# Calculated Sliding Window Connectivity for Selected ROIs ----------
dim(dat_array)
ROIs <- c(186, 90, 71, 14, 247, 20, 110, 288, 193, 121)
dat_array <- dat_array[, ROIs, ]

L <- 30
n_windows <- dim(dat_array)[3] - L
sw <- array(dim = c(n_subjs, length(ROIs), length(ROIs), n_windows))
sw_log <- sw
for (i in 1:n_subjs) {
  cat(i)
  for (w in 1:n_windows) {
    sw[i, , , w] <- cor(t(dat_array[i, ROIs, w:(w + L)]))
    sw_log[i, , , w] <- expm::logm(sw[i, , , w])
  }
}



# Plots ---------------------------------------------------------------
i <- 1
par(mfrow = c(3, 2))
plot(sw[i, 1, 2, ], ty = "l")
plot(sw_log[i, 1, 2, ], ty = "l")
plot(sw[i, 1, 3, ], ty = "l")
plot(sw_log[i, 1, 3, ], ty = "l")
plot(sw_log[i, 2, 3, ], ty = "l")
plot(sw_log[i, 2, 3, ], ty = "l")

i <- 2
par(mfrow = c(3, 2))
plot(sw[i, 1, 2, ], ty = "l")
plot(sw_log[i, 1, 2, ], ty = "l")
plot(sw[i, 1, 3, ], ty = "l")
plot(sw_log[i, 1, 3, ], ty = "l")
plot(sw_log[i, 2, 3, ], ty = "l")
plot(sw_log[i, 2, 3, ], ty = "l")

par(mfrow = c(1, 1))
ts.plot(t(sw[1:6, 1, 2, ]), col = 1:6)

# vectorized log SW --------------------------------------------------
sw_log_vec <- array(dim = c(n_subjs, choose(length(ROIs) + 1, 2), n_windows))
for (i in 1:n_subjs) {
  for (w in 1:n_windows) {
    sw_log_vec[i, , w] <- sw_log[i, , , w][lower.tri(sw_log[i, , , w], diag = TRUE)]
  }
}

# SW array to data frame ----------------------------------------------
q <- choose(length(ROIs) + 1, 2)
dat_sw_log <- as.tibble(expand.grid(1:259, 1:q, 1:n_windows))
colnames(dat_sw_log) <- c("subj", "cell", "w")
dat_sw_log$value <- NA
for (k in 1:nrow(dat_sw_log)) {
  if (k %% 1000 == 0) {cat(k / 1000)}
  dat_sw_log$value[k] <- sw_log_vec[dat_sw_log$subj[k], dat_sw_log$cell[k], dat_sw_log$w[k]]
}

# Plots ---------------------------------------------------------------
selected_subj <- 2
for (selected_cell in 1:q) {
  plt <- ggplot(dat_sw_log %>% filter(subj == selected_subj, cell == selected_cell)) + 
    geom_line(aes(x = w, y = value)) + 
    facet_grid(rows = vars(factor(cell))) + 
    ylab("Log SW Covariance") +
    xlab("Window") + 
    ggtitle(glue::glue("SW Covariance Cell {selected_cell}"))
  ggsave(device = "png", plot = plt, filename = glue::glue("img/log_sw_cov_{selected_subj}_{selected_cell}"))
}


