library(ape)
library(phylolm)

data_dir <- "/home/sosa/work/BI/Test/GGMM (WIP)/graphical_mixed_model/data"
out_dir  <- "/home/sosa/work/BI/Test/GMM (WIP)"

# Load and prepare tree
tree <- read.tree(file.path(data_dir, "VertTree_mammals.tre"))
max_edge <- max(tree$edge.length)
tree$edge.length <- tree$edge.length / max_edge

# Load trait data
traits <- read.table(file.path(data_dir, "PanTHERIA_1-0_WR05_Aug2008.txt"), row.names = NULL)
tree$tip.label <- tolower(tree$tip.label)
traits$traits_binom <- paste0(tolower(traits$MSW05_Species), "_", traits$MSW05_Binomial)
trait_to_tip <- match(traits$traits_binom, tree$tip.label)
traits <- traits[which(!is.na(trait_to_tip)), ]

add_NA <- function(vec) ifelse(as.numeric(vec) == -999, NA, as.numeric(vec))
data <- data.frame(
  ln_metabolism = log(add_NA(traits[, "X18.1_BasalMetRate_mLO2hr"])),
  ln_range      = log(add_NA(traits[, "X22.1_HomeRange_km2"])),
  ln_size       = log(add_NA(traits[, "X5.1_AdultBodyMass_g"]))
)
rownames(data) <- traits$traits_binom

# Prune tree to complete cases only (all 3 traits)
complete_rows <- complete.cases(data)
data_complete <- data[complete_rows, , drop = FALSE]
tree_complete <- keep.tip(tree, rownames(data_complete))

cat("Complete cases:", nrow(data_complete), "\n")

# Fit OU models using complete cases
plm1 <- phylolm(ln_metabolism ~ ln_size, data = data_complete,
                phy = tree_complete, model = "OUrandomRoot")
plm2 <- phylolm(ln_range ~ ln_size, data = data_complete,
                phy = tree_complete, model = "OUrandomRoot")
plm3 <- phylolm(ln_size ~ 1, data = data_complete,
                phy = tree_complete, model = "OUrandomRoot")

# Collect estimates
params <- c(
  "b1",               "b2",
  "xbar_metabolism",  "xbar_range",  "xbar_size",
  "theta_metabolism", "theta_range", "theta_size",
  "sigma2_metabolism","sigma2_range","sigma2_size"
)
estimates <- c(
  as.numeric(plm1$coefficients["ln_size"]),
  as.numeric(plm2$coefficients["ln_size"]),
  as.numeric(plm1$coefficients["(Intercept)"]),
  as.numeric(plm2$coefficients["(Intercept)"]),
  as.numeric(plm3$coefficients["(Intercept)"]),
  plm1$optpar,
  plm2$optpar,
  plm3$optpar,
  plm1$sigma2,
  plm2$sigma2,
  plm3$sigma2
)

results <- data.frame(parameter = params, estimate = estimates)
cat("\nReference estimates (phylolm OUrandomRoot, complete cases):\n")
print(results)

write.csv(results, file.path(out_dir, "reference_estimates.csv"), row.names = FALSE)
cat("\nSaved to reference_estimates.csv\n")
