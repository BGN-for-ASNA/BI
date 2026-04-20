full_data <- readRDS('c:/Users/Sosa/Documents/BI/Test/Capture-Recapture (WIP)/cr-in-stan/case-studies/data/fleayi-stan-data.rds')
tau <- full_data$tau[1:20, 1]
print("TAU SUMMARY:")
print(summary(tau))
y_raw <- full_data$y[1, 1:407, 1:21, ]
y_mat <- apply(y_raw, c(1, 2), function(x) {
    found <- which(x == 1)
    if (length(found) == 0) return(0L)
    as.integer(found[1])
})
print("Y_MAT SUMMARY:")
print(summary(as.vector(y_mat)))
print("Y_MAT UNIQUE:")
print(unique(as.vector(y_mat)))
