library(ape)

phylo <- read.nexus("phylo_slopes.nex")
A <- vcv.phylo(phylo)
A <- A / max(A)
L <- t(chol(A))

# Save L with species names as headers
write.csv(L, "L_slopes.csv", row.names = FALSE)
cat("Cholesky factor saved to L_slopes.csv\n")
