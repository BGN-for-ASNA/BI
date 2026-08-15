library(mgcv)
library(dplyr)
library(tidyr)

# Load data
bird_move <- read.csv("data/bird_move.csv", stringsAsFactors = TRUE)

# Prepare data for mgcv
bird_move$species <- as.factor(bird_move$species)

# Define models as per Pedersen et al. 2019
# Reducing k to speed up the benchmark comparison
cat("Fitting Model G...\n")
bird_modG <- bam(count ~ te(week, latitude, bs=c("cc", "tp"), k=c(5, 5)),
                 data=bird_move, method="fREML", family="poisson",
                 knots=list(week=c(0, 52)))

cat("Fitting Model GS...\n")
bird_modGS <- bam(count ~ te(week, latitude, bs=c("cc", "tp"), k=c(5, 5), m=2) +
                     t2(week, latitude, species, bs=c("cc", "tp", "re"),
                        k=c(5, 5, 6), m=2, full=TRUE),
                   data=bird_move, method="fREML", family="poisson", 
                   knots=list(week=c(0, 52)))

cat("Fitting Model GI...\n")
bird_modGI <- bam(count ~ species + 
                     te(week, latitude, bs=c("cc", "tp"), k=c(5, 5), m=2) +
                     te(week, latitude, by=species, bs= c("cc", "tp"),
                        k=c(5, 5), m=1),
                  data=bird_move, method="fREML", family="poisson",
                  knots=list(week=c(0, 52)))

cat("Fitting Model S...\n")
bird_modS <- bam(count ~ t2(week, latitude, species, bs=c("cc", "tp", "re"),
                             k=c(5, 5, 6), m=2, full=TRUE),
                  data=bird_move, method="fREML", family="poisson",
                  knots=list(week=c(0, 52)))

cat("Fitting Model I...\n")
bird_modI <- bam(count ~ species + te(week, latitude, by=species,
                                       bs=c("cc", "tp"), k=c(5, 5), m=2),
                  data=bird_move, method="fREML", family="poisson",
                  knots=list(week=c(0, 52)))

# Function to export model data for BF
export_bi_data <- function(model, name) {
  # Design matrix
  X <- predict(model, type="lpmatrix")
  y <- model$y
  
  # Coefficients
  coefs <- coef(model)
  ses <- sqrt(diag(vcov(model)))
  
  dir.create(paste0("BF_data/", name), recursive=TRUE, showWarnings=FALSE)
  
  write.csv(X, paste0("BF_data/", name, "/X.csv"), row.names=FALSE)
  write.csv(data.frame(y=y), paste0("BF_data/", name, "/y.csv"), row.names=FALSE)
  write.csv(data.frame(coef=coefs, se=ses), paste0("BF_data/", name, "/results_r.csv"), row.names=FALSE)
  
  # Export penalty information
  # We'll save the smoothing parameters and the S matrices
  lambdas <- model$sp
  write.csv(data.frame(lambda=lambdas), paste0("BF_data/", name, "/lambdas.csv"), row.names=TRUE)
  
  # Export each smooth's S and its range in X
  smooth_info <- list()
  for (i in seq_along(model$smooth)) {
    sm <- model$smooth[[i]]
    # Range of coefficients this smooth covers
    first <- sm$first.para
    last <- sm$last.para
    
    # S matrices
    for (k in seq_along(sm$S)) {
      s_mat <- sm$S[[k]]
      # Save S matrix
      write.csv(as.matrix(s_mat), paste0("BF_data/", name, "/S_", i, "_", k, ".csv"), row.names=FALSE)
    }
    
    smooth_info[[i]] <- list(label=sm$label, first=first, last=last, n_S = length(sm$S))
  }
  
  # Save smooth info summary
  saveRDS(smooth_info, paste0("BF_data/", name, "/smooth_info.rds"))
  # Also write as CSV for easy reading in Python
  info_df <- do.call(rbind, lapply(seq_along(smooth_info), function(i) {
    data.frame(id=i, label=smooth_info[[i]]$label, first=smooth_info[[i]]$first, last=smooth_info[[i]]$last, n_S=smooth_info[[i]]$n_S)
  }))
  write.csv(info_df, paste0("BF_data/", name, "/smooth_info.csv"), row.names=FALSE)
}

cat("Exporting data...\n")
# We'll refine the export to be more comprehensive for all models
models <- list(G=bird_modG, GS=bird_modGS, GI=bird_modGI, S=bird_modS, I=bird_modI)

for (n in names(models)) {
  export_bi_data(models[[n]], n)
}

cat("All models fitted and exported successfully.\n")
