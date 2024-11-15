source(normalizePath('wsl.localhost\Ubuntu-22.04\home\sosa\work\BI\NBDA simulation\NBDA code 1.2.15.R'))

simulateNBDA<-function(m, s = 5, BNoise = 0.1 , baseRate=1/100,asocialLP=rep(1,N)){
  # Generate social transmission coefficients with noise from a normal distribution
  BVect <- exp(rnorm(N, log(2), sd = BNoise))
  N = nrow(m)
  # Initialize vectors for acquisition status, order, and time for each individual
  z = orderAcq = timeAcq = rep(0, N)
  
  # Set initial running time to zero
  runningTime <- 0
  
  # Loop through each individual to simulate the transmission process
  for (i in 1:N) {
    # Calculate the rate of acquisition for each individual, taking asocial learning, social influence, and acquisition status into account
    rate <- baseRate * (exp(asocialLP) + s * z %*% t(t(m) * BVect)) * (1 - z)
    
    # Generate times to the next acquisition event using an exponential distribution
    times <- rexp(N, rate)
    
    # Replace NaN values in `times` (where rate might be zero) with Inf, so they are ignored in finding the minimum time
    times[is.nan(times)] <- Inf
    
    # Find the individual with the shortest time to the next acquisition event and update acquisition order
    orderAcq[i] <- which(times == min(times))[1]
    
    # Add the minimum time to the running time to track cumulative acquisition time
    runningTime <- runningTime + min(times)
    
    # Record the cumulative time of acquisition for this individual
    timeAcq[i] <- runningTime
    
    # Update the acquisition status of the individual who acquired the trait
    z[which(times == min(times))[1]] <- 1
  }
  
  # Return a list containing acquisition times and order of acquisition
  return(list(timeAcq = timeAcq, orderAcq = orderAcq))
}
N = 10
network = matrix(runif(N*N,0.7,1)*rbinom(N*N,1,0.3), nrow=N)
resultR = simulateNBDA(network, s = 5)


## Testing simulation with NBDA library

library(NBDA)
adj.array=array(dim=c(nrow(network), ncol(network), 1))
adj.array[,,1]= network
diffdat=nbdaData("try1", 
                 assMatrix=adj.array,
                 orderAcq=resultR$orderAcq, 
                 timeAcq=resultR$timeAcq)

oa.fit_social=oadaFit(diffdat, type="social")

oa.fit_social=oadaFit(diffdat, type="social")
oa.fit_social@outputPar
oa.fit_social@aic
data.frame(Variable=oa.fit_social@varNames,MLE=oa.fit_social@outputPar,SE=oa.fit_social@se)

ta.fit_social=tadaFit(diffdat, type="social")
#ta.fit_social@outputPar
data.frame(Variable=ta.fit_social@varNames,MLE=round(ta.fit_social@outputPar,3),SE=round(ta.fit_social@se,3))

