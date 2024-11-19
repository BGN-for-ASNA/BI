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
oa.fit_social@outputPar
oa.fit_social@aic
data.frame(Variable=oa.fit_social@varNames,MLE=oa.fit_social@outputPar,SE=oa.fit_social@se)

ta.fit_social=tadaFit(diffdat, type="social")
#ta.fit_social@outputPar
data.frame(Variable=ta.fit_social@varNames,MLE=round(ta.fit_social@outputPar,3),SE=round(ta.fit_social@se,3))

# Computing s metric--------------------
slot_names <- slotNames(diffdat)
s4_elements <- setNames(
  lapply(slot_names, function(x) slot(diffdat, x)),
  slot_names
)
list2env(s4_elements, envir = .GlobalEnv)
demons=NULL

{
  
  if(is.na(timeAcq[1])){
    
    #put the time varying association matrix into an object called assMatrixTV
    if(length(dim(assMatrix))==3){ assMatrixTV<- array(data=assMatrix,dim=c(dim(assMatrix),1))}else{assMatrixTV<-assMatrix}
    
    # create default numeric id vector for individuals if no names are provided, if it is, convert to a factor
    if(is.null(idname)) idname <- (1:dim(assMatrix)[1]);
    # if there are no ties, make a vector of zeroes
    if(is.null(ties)) ties <- rep(0,length(orderAcq));
    
    
    
    # each asoc_ vector should be a vector of character strings of matrices whose names correspond to the names of the individual-level variables (ILVs). the rows of each matrix should correspond to individuals and the columns to times at which the value of the ILV changes
    nAcq <- ifelse(any(is.na(orderAcq)), 0, length(orderAcq)) # number of acquisition events EXCLUDING demonstrators.
    time1 <- vector(); # time period index: time start
    time2 <- vector(); # time period index: time end
    event.id.temp <- vector(); # vector to indicate the event number (this is essentially indexing the naive individuals before each event)
    
    #If no asoc variable is provided, set a single column of zeroes
    if(asoc_ilv[1]=="ILVabsent"|int_ilv[1]=="ILVabsent"|multi_ilv[1]=="ILVabsent"){
      ILVabsent <-matrix(data = rep(0, dim(assMatrix)[1]), nrow=dim(assMatrix)[1], byrow=F)
    }
    if(random_effects[1]=="REabsent"){
      REabsent <-matrix(data = rep(0, dim(assMatrix)[1]), nrow=dim(assMatrix)[1], byrow=F)
    }
    
    totalMetric <- vector() # total association of the individual that DID learn at an acquisition event, with all other individuals
    learnMetric <- vector(); # total associations of the individual that DID learn at an acquisition event, with all other individuals that have already learned
    
    status <- presentInDiffusion <-vector(); # set up the status vector and presentInDiffusion vector
    
    # If there is just one asocial variable matrix for all events and times, then you will have a column matrix for each ILV, the length of the number of individuals
    # If there are more than one asocial variable matrices (i.e. time-varying covariates), then you will have a matrix for each ILV, with rows equal to the number of individuals, and columns equal to the number of acquisition events (because in OADA we are constraining this to be the case: ILVs can only change at the same time as acquisition events occur otherwise you can't obtain a marginal likelihood, Will says, only a partial likelihood)
    asoc_ilv.dim <- dim(eval(as.name(asoc_ilv[1])))[1] # specify the dimensions of assoc.array
    int_ilv.dim <- dim(eval(as.name(int_ilv[1])))[1] # specify the dimensions of assoc.array
    multi_ilv.dim <- dim(eval(as.name(multi_ilv[1])))[1] # specify the dimensions of assoc.array
    random_effects.dim <- dim(eval(as.name(random_effects[1])))[1] # specify the dimensions of assoc.array
    
    
    # create asoc.array to hold the asocial variables. depending on the treatment required: "timevarying" or "constant", create a one-matrix array or a multi-matrix array
    if (asocialTreatment=="constant"){
      asoc_ilv.array <- array(dim=c(asoc_ilv.dim, 1, length(asoc_ilv)))
      dimnames(asoc_ilv.array) <- list(NULL, NULL, asoc_ilv)
      int_ilv.array <- array(dim=c(int_ilv.dim, 1, length(int_ilv)))
      dimnames(int_ilv.array) <- list(NULL, NULL, int_ilv)
      multi_ilv.array <- array(dim=c(multi_ilv.dim, 1, length(multi_ilv)))
      dimnames(multi_ilv.array) <- list(NULL, NULL, multi_ilv)
      random_effects.array <- array(dim=c(random_effects.dim, 1, length(random_effects)))
      dimnames(random_effects.array) <- list(NULL, NULL, random_effects)
      
    } else {
      if (asocialTreatment=="timevarying"){
        asoc_ilv.array <- array(dim=c(asoc_ilv.dim,nAcq,length(asoc_ilv)))
        dimnames(asoc_ilv.array) <- list(NULL, c(paste("time",c(1:nAcq),sep="")), asoc_ilv)
        int_ilv.array <- array(dim=c(int_ilv.dim,nAcq,length(int_ilv)))
        dimnames(int_ilv.array) <- list(NULL, c(paste("time",c(1:nAcq),sep="")), int_ilv)
        multi_ilv.array <- array(dim=c(multi_ilv.dim,nAcq,length(multi_ilv)))
        dimnames(multi_ilv.array) <- list(NULL, c(paste("time",c(1:nAcq),sep="")), multi_ilv)
        random_effects.array <- array(dim=c(random_effects.dim,nAcq,length(random_effects)))
        dimnames(random_effects.array) <- list(NULL, c(paste("time",c(1:nAcq),sep="")), random_effects)
      }
    }
    
    # generate a matrix that contains the status of each individual at each acquisition event
    statusMatrix <- matrix(0, nrow=dim(assMatrix)[2], ncol=1+nAcq)  # a matrix with as many rows as indivs and as many columns as acquisition events PLUS one for the demonstrators
    # create a list vector to hold the index of naive individuals after each acquisition event
    naive.id <-naive.id.names<- vector(mode="list", length=nAcq)
    
    # if there are seeded demonstrators add the vector (which should have length dim(assMatrix)[1]) to the first column of the statusMatrix to show which individuals set out as skilled (status of 1)
    if(is.null(demons)){
      statusMatrix[,1] <- rep(0,dim(assMatrix)[1])
    } else {
      for(i in 1:(1+nAcq)){
        statusMatrix[,i] <- demons
      }
    }
    
    availabilityMatrix <- statusMatrix # if there are ties the statusMatrix and the availabilityMatrix will differ (the latter gives who is available to be learned *from*). we create it as identical and modify it accordingly below
    
    #presenceMatrix gives who was present in the diffusion for each event- set to 1s by default
    if(is.null(presenceMatrix)){
      presenceMatrix<-statusMatrix;
      presenceMatrix[,]<-1;
    }else{
      #Add a column to the start of the presenceMatrix so the dimensions match statusMatrix
      presenceMatrix<-cbind(presenceMatrix[,1], presenceMatrix)
    }
    
    
    asocILVdata.naive <-intILVdata.naive<-multiILVdata.naive<-randomEffectdata.naive<-vector() # this will hold the individual level variables for the naive individuals
    
    # YOU WILL HAVE TO MAKE THIS WORK EVEN WHEN THERE ARE NO ASOCIAL VARIABLES... THINK ABOUT HOW YOU MIGHT DO THIS 20120822 (Theoni comment)
    # Will: I just put a dummy asoc variable in at the start with all 0s, when the model is fitted using oadaFit the constraints and offsets vector are
    # automatically modified to ignore the dummy variable (a zero is appended to the end of each, or 2 zeroes if type=unconstrained is specified)
    
    ############# to prevent errors when nAcq is zero
    if(nAcq==0){
      learnAsoc <-learnInt<-multiInt<- naive.id <- time1 <- time2 <- stMetric <- NA
      id <- id # this will be NA by default
    } else {
      #############
      
      # calculate the asocial learning variables for the learning individual at each step (at each acquisition event)
      learnAsoc <- matrix(nrow=length(asoc_ilv), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnAsoc) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
      learnInt <- matrix(nrow=length(int_ilv), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnInt) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
      learnMulti<- matrix(nrow=length(multi_ilv), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnMulti) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
      learnRE<- matrix(nrow=length(random_effects), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnRE) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
    } # closes else
    
    for(a in 1:length(asoc_ilv)){ # Loop through asocial variables - a loop
      asoc_ilv.array[,,a] <- eval(as.name(asoc_ilv[a])) # evaluate each one in turn
    }
    for(a in 1:length(int_ilv)){ # Loop through asocial variables - a loop
      int_ilv.array[,,a] <- eval(as.name(int_ilv[a])) # evaluate each one in turn
    }
    for(a in 1:length(multi_ilv)){ # Loop through asocial variables - a loop
      multi_ilv.array[,,a] <- eval(as.name(multi_ilv[a])) # evaluate each one in turn
    }
    for(a in 1:length(random_effects)){ # Loop through asocial variables - a loop
      random_effects.array[,,a] <- eval(as.name(random_effects[a])) # evaluate each one in turn
    }
    
    if(nAcq!=0){
      for (i in 1:nAcq){ # Loop through acquisition events - i loop
        
        k <- ifelse(asocialTreatment=="constant", 1, i) # this makes sure you are treating ILVs correctly if they are constant and if they are time-varying
        
        for(a in 1:length(asoc_ilv)){ # Loop through asocial variables - a loop
          learnAsoc[a,i] <- asoc_ilv.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        for(a in 1:length(int_ilv)){ # Loop through asocial variables - a loop
          learnInt[a,i] <- int_ilv.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        for(a in 1:length(multi_ilv)){ # Loop through asocial variables - a loop
          learnMulti[a,i] <- multi_ilv.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        for(a in 1:length(random_effects)){ # Loop through asocial variables - a loop
          learnRE[a,i] <- random_effects.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        
        statusMatrix[orderAcq[i],c((i+1):(nAcq+1))] <- 1 # give the individuals that acquired the trait a status of 1 and carry skilled status (1) through to all following acquisition events
        
        #WH this section was wrong- I  correct it below
        # correct the status of the individuals that can be learned from if there are ties, because in that case they will not be the same as the skilled individuals
        #               if (ties[i]==0){
        #                 availabilityMatrix[orderAcq[i],] <- statusMatrix[orderAcq[i],]
        #               } else {
        #                 availabilityMatrix[orderAcq[i],] <- ifelse(length(orderAcq[i-1]), availabilityMatrix[orderAcq[i-1],], statusMatrix[orderAcq[i],])
        #               } # closes ties if statement
        
        # if the event is recorded as tied with the previous event (ties[i]==1), it means that whoever learned in the previous event cannot be learned from for this event
        # therefore if a tie is present for event i, we do not update the availabilityMatrix to match the statusMatrix until the ties is ended
        if (ties[i]==0){
          availabilityMatrix[,i] <- statusMatrix[,i]
        } else {
          availabilityMatrix[,i] <- availabilityMatrix[,i-1]
        } # closes ties if statement
        
        
        #Now correct the availabilityMatrix such that individuals who are not present for an event cannot be learned from
        availabilityMatrix<-availabilityMatrix*presenceMatrix
        
        naive.id[[i]] <- which(statusMatrix[,i]==0) # index for naive individuals before the ith acquisition event
        
      } # closes the i loop - nAcq (k is i or 1)
      availabilityMatrix[,nAcq+1] <- statusMatrix[,nAcq+1]
    } # closes the if statement for nAcq!=0
    
    
    
    if(is.na(id[1])) {id <- paste(label,c(unlist(naive.id)), sep="_")} # id of naive individuals before each acquisition event, including demonstrators
    
    naive <- dim(assMatrix)[1]-apply(statusMatrix, 2, sum) # number of naive individuals remaining after each acq event
    
    
    # work out the number of association matrices provided and set up stMetric matrix accordingly
    stMetric <- matrix(data=0, nrow=length(id), ncol=dim(assMatrix)[3])
    dimnames(stMetric) <- list(NULL, paste("stMetric",c(1:dim(assMatrix)[3]),sep=""))
    
    #############################################################
    # Loop through acquisition events - learner loop
    # learnMetric is the sum of network connections of individuals that learned at each acquisition events, to other informed individuals.
    # This will always be 0 for the first animal that learned unless there were demonstrators
    
    # time1 and time2 index the time period or "event period" corresponding to acquisitions
    if(nAcq!=0){
      for(event in 1:nAcq){
        # it's a shame to have two identical loops but I need time1 and time2 to be ready for use when I come to calculate the social transmission metrics below
        time1 <- c(time1, rep(event-1, naive[event]))
        time2 <- c(time2, rep(event, naive[event]))
        if(is.na(event.id[1])){
          event.id.temp <- c(event.id.temp, rep(event, each=length(naive.id[[event]])))
        }
      } # closes for loop through events
      
      if(is.na(event.id[1])) {event.id <- paste(label, event.id.temp, sep="_")}
      
      for(event in 1:nAcq){ # event indexes the number of the acquisition event
        
        #Take the appropriate association matrix from the (weighted) time varying association matrix,
        #as determined for that event by the assMatrixIndex vector
        assMatrix<-array(assMatrixTV[,,, assMatrixIndex[event]],dim=dim(assMatrixTV)[1:3])
        
        learner <- orderAcq[event] # learner is individual id of the animal that learned AT an event
        nonlearners <- naive.id[[event]] # nonlearners are individual id of the animals that were naive BEFORE an event
        status <- c(status, statusMatrix[unlist(naive.id[[event]]), event+1])
        presentInDiffusion<-c(presentInDiffusion,presenceMatrix[unlist(naive.id[[event]]), event+1])
        
        temp.stMetric <- vector() # reset this before the metrics for each event are calculated
        
        for (nonlearner in nonlearners){
          
          # stMetric is the total assoc of the individuals that had NOT learned prior to that acquisition event, with all other already-informed individuals
          m1 <- matrix(data=assMatrix[nonlearner,,], nrow=dim(assMatrix)[3], byrow=T) # matrix1
          m2 <- (weights*availabilityMatrix[,event])*t(m1) # matrix2
          v1 <- apply(X=m2, MARGIN=2, FUN=sum) # vector1 of rowsums
          temp.stMetric <- rbind(temp.stMetric, v1)
          
        } # closes nonlearner loop for MATRIX stMetric
        
        stMetric[time2==event,] <- temp.stMetric
        
        if(asoc_ilv[1]=="ILVabsent"){
          ilv1 <-cbind("ILVabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            ilv1 <- matrix(asoc_ilv.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            ilv1 <- matrix(asoc_ilv.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        if(int_ilv[1]=="ILVabsent"){
          intilv1 <-cbind("ILVabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            intilv1 <- matrix(int_ilv.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            intilv1 <- matrix(int_ilv.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        if(multi_ilv[1]=="ILVabsent"){
          multiilv1 <-cbind("ILVabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            multiilv1 <- matrix(multi_ilv.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            multiilv1 <- matrix(multi_ilv.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        if(random_effects[1]=="REabsent"){
          randomeffect1 <-cbind("REabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            randomeffect1 <- matrix(random_effects.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            randomeffect1 <- matrix(random_effects.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        
        
        asocILVdata.naive <- rbind(asocILVdata.naive, ilv1)
        if(asoc_ilv[1]=="ILVabsent"){
          attr(asocILVdata.naive, "dimnames") <- list(NULL,"ILVabsent")
        }else{
          attr(asocILVdata.naive, "dimnames") <- list(NULL,asoc_ilv)
        }
        
        intILVdata.naive <- rbind(intILVdata.naive, intilv1)
        if(int_ilv[1]=="ILVabsent"){
          attr(intILVdata.naive, "dimnames") <- list(NULL,"ILVabsent")
        }else{
          attr(intILVdata.naive, "dimnames") <- list(NULL,int_ilv)
        }
        
        multiILVdata.naive <- rbind(multiILVdata.naive, multiilv1)
        if(multi_ilv[1]=="ILVabsent"){
          attr(multiILVdata.naive, "dimnames") <- list(NULL,"ILVabsent")
        }else{
          attr(multiILVdata.naive, "dimnames") <- list(NULL,multi_ilv)
        }
        randomEffectdata.naive <- rbind(randomEffectdata.naive, randomeffect1)
        if(random_effects[1]=="REabsent"){
          attr(randomEffectdata.naive, "dimnames") <- list(NULL,"REabsent")
        }else{
          attr(randomEffectdata.naive, "dimnames") <- list(NULL,random_effects)
        }
        
      } # closes event loop
    } else { # closes if(nAcq!=0) statement
      asocILVdata.naive <-matrix(data=rep(0,length(asoc_ilv)), nrow=1, ncol=length(asoc_ilv))
      intILVdata.naive<-matrix(data=rep(0,length(int_ilv)), nrow=1, ncol=length(int_ilv))
      multiILVdata.naive<-matrix(data=rep(0,length(multi_ilv)), nrow=1, ncol=length(multi_ilv))
      randomEffectdata.naive<-matrix(data=rep(0,length(random_effects)), nrow=1, ncol=length(random_effects))
      
      attr(asocILVdata.naive, "dimnames") <- list(NULL,asoc_ilv)
      attr(multiILVdata.naive, "dimnames") <- list(NULL,int_ilv)
      attr(intILVdata.naive, "dimnames") <- list(NULL,multi_ilv)
      attr(randomEffectdata.naive, "dimnames") <- list(NULL,random_effects)
    } # closes else
    
    
    #############################################################
    label <- rep(label, length.out=length(id))
    
    if(is.null(demons)) demons <- NA;
    
    #Subtract the first column from presenceMatrix (added previously) so it again gives the presence of each individual for each event
    presenceMatrix<-presenceMatrix[,-1]
    
    if(is.null(offsetCorrection)) offsetCorrection <- cbind(rep(0,dim(asocILVdata.naive)[1]),rep(0,dim(asocILVdata.naive)[1]),rep(0,dim(asocILVdata.naive)[1]),rep(0,dim(asocILVdata.naive)[1]));
    dimnames(offsetCorrection)[2]<-list(c("SocialOffsetCorr","AsocialILVOffsetCorr","InteractionOffsetCorr","MultiplicativeILVOffsetCorr"))
    
    callNextMethod(.Object, label=label, idname=idname, assMatrix=assMatrixTV, asoc_ilv=asoc_ilv, int_ilv=int_ilv,multi_ilv=multi_ilv,random_effects=random_effects, orderAcq=orderAcq, timeAcq=timeAcq, endTime=endTime,updateTimes=NA, ties=ties, trueTies=trueTies, demons=demons, weights=weights, statusMatrix=statusMatrix, availabilityMatrix=availabilityMatrix, event.id=event.id, id=id, time1=time1, time2=time2, status=status, presentInDiffusion= presentInDiffusion, presenceMatrix = presenceMatrix ,asocialTreatment=asocialTreatment, stMetric=stMetric, asocILVdata=asocILVdata.naive, intILVdata=intILVdata.naive, multiILVdata=multiILVdata.naive,randomEffectdata=randomEffectdata.naive,offsetCorrection=offsetCorrection,assMatrixIndex=assMatrixIndex)
  }else{
    #TADA version to be inserted here
    
    #put the time varying association matrix into an object called assMatrixTV
    if(length(dim(assMatrix))==3){ assMatrixTV<- array(data=assMatrix,dim=c(dim(assMatrix),1))}else{assMatrixTV<-assMatrix}
    
    # create default numeric id vector for individuals if no names are provided, if it is, convert to a factor
    if(is.null(idname)) idname <- (1:dim(assMatrix)[1]);
    # if there are no ties, make a vector of zeroes
    if(is.null(ties)) ties <- rep(0,length(orderAcq));
    
    
    
    # each asoc_ vector should be a vector of character strings of matrices whose names correspond to the names of the individual-level variables (ILVs). the rows of each matrix should correspond to individuals and the columns to times at which the value of the ILV changes
    nAcq <- ifelse(any(is.na(orderAcq)), 0, length(orderAcq)) # number of acquisition events EXCLUDING demonstrators.
    time1 <- vector(); # time period index: time start
    time2 <- vector(); # time period index: time end
    event.id.temp <- vector(); # vector to indicate the event number (this is essentially indexing the naive individuals before each event)
    
    #If no asoc variable is provided, set a single column of zeroes
    if(asoc_ilv[1]=="ILVabsent"|int_ilv[1]=="ILVabsent"|multi_ilv[1]=="ILVabsent"){
      ILVabsent <-matrix(data = rep(0, dim(assMatrix)[1]), nrow=dim(assMatrix)[1], byrow=F)
    }
    if(random_effects[1]=="REabsent"){
      REabsent <-matrix(data = rep(0, dim(assMatrix)[1]), nrow=dim(assMatrix)[1], byrow=F)
    }
    
    totalMetric <- vector() # total association of the individual that DID learn at an acquisition event, with all other individuals
    learnMetric <- vector(); # total associations of the individual that DID learn at an acquisition event, with all other individuals that have already learned
    
    status <- presentInDiffusion <-vector(); # set up the status vector and presentInDiffusion vector
    
    # If there is just one asocial variable matrix for all events and times, then you will have a column matrix for each ILV, the length of the number of individuals
    # If there are more than one asocial variable matrices (i.e. time-varying covariates), then you will have a matrix for each ILV, with rows equal to the number of individuals, and columns equal to the number of acquisition events (because in OADA we are constraining this to be the case: ILVs can only change at the same time as acquisition events occur otherwise you can't obtain a marginal likelihood, Will says, only a partial likelihood)
    asoc_ilv.dim <- dim(eval(as.name(asoc_ilv[1])))[1] # specify the dimensions of assoc.array
    int_ilv.dim <- dim(eval(as.name(int_ilv[1])))[1] # specify the dimensions of assoc.array
    multi_ilv.dim <- dim(eval(as.name(multi_ilv[1])))[1] # specify the dimensions of assoc.array
    random_effects.dim <- dim(eval(as.name(random_effects[1])))[1] # specify the dimensions of assoc.array
    
    
    # create asoc.array to hold the asocial variables. depending on the treatment required: "timevarying" or "constant", create a one-matrix array or a multi-matrix array
    if (asocialTreatment=="constant"){
      asoc_ilv.array <- array(dim=c(asoc_ilv.dim, 1, length(asoc_ilv)))
      dimnames(asoc_ilv.array) <- list(NULL, NULL, asoc_ilv)
      int_ilv.array <- array(dim=c(int_ilv.dim, 1, length(int_ilv)))
      dimnames(int_ilv.array) <- list(NULL, NULL, int_ilv)
      multi_ilv.array <- array(dim=c(multi_ilv.dim, 1, length(multi_ilv)))
      dimnames(multi_ilv.array) <- list(NULL, NULL, multi_ilv)
      random_effects.array <- array(dim=c(random_effects.dim, 1, length(random_effects)))
      dimnames(random_effects.array) <- list(NULL, NULL, random_effects)
      
    } else {
      if (asocialTreatment=="timevarying"){
        asoc_ilv.array <- array(dim=c(asoc_ilv.dim,(nAcq+1),length(asoc_ilv)))
        dimnames(asoc_ilv.array) <- list(NULL, c(paste("time",c(1:(nAcq+1)),sep="")), asoc_ilv)
        int_ilv.array <- array(dim=c(int_ilv.dim,(nAcq+1),length(int_ilv)))
        dimnames(int_ilv.array) <- list(NULL, c(paste("time",c(1:(nAcq+1)),sep="")), int_ilv)
        multi_ilv.array <- array(dim=c(multi_ilv.dim,(nAcq+1),length(multi_ilv)))
        dimnames(multi_ilv.array) <- list(NULL, c(paste("time",c(1:(nAcq+1)),sep="")), multi_ilv)
        random_effects.array <- array(dim=c(random_effects.dim,(nAcq+1),length(random_effects)))
        dimnames(random_effects.array) <- list(NULL, c(paste("time",c(1:(nAcq+1)),sep="")), random_effects)
      }
    }
    
    # generate a matrix that contains the status of each individual at each acquisition event
    statusMatrix <- matrix(0, nrow=dim(assMatrix)[2], ncol=1+nAcq)  # a matrix with as many rows as indivs and as many columns as acquisition events PLUS one for the demonstrators
    # create a list vector to hold the index of naive individuals after each acquisition event
    naive.id <- vector(mode="list", length=nAcq+1)
    
    # if there are seeded demonstrators add the vector (which should have length dim(assMatrix)[1]) to the first column of the statusMatrix to show which individuals set out as skilled (status of 1)
    if(is.null(demons)){
      statusMatrix[,1] <- rep(0,dim(assMatrix)[1])
    } else {
      for(i in 1:(1+nAcq)){
        statusMatrix[,i] <- demons
      }
    }
    
    availabilityMatrix <- statusMatrix # if there are ties the statusMatrix and the availabilityMatrix will differ (the latter gives who is available to be learned *from*). we create it as identical and modify it accordingly below
    
    #presenceMatrix gives who was present in the diffusion for each event- set to 1s by default
    if(is.null(presenceMatrix)){
      presenceMatrix<-statusMatrix;
      presenceMatrix[,]<-1;
    }else{
      #Add a column to the start of the presenceMatrix so the dimensions match statusMatrix
      presenceMatrix<-cbind(presenceMatrix[,1], presenceMatrix)
    }
    
    
    asocILVdata.naive <-intILVdata.naive<-multiILVdata.naive<-randomEffectdata.naive<-vector() # this will hold the individual level variables for the naive individuals
    
    # YOU WILL HAVE TO MAKE THIS WORK EVEN WHEN THERE ARE NO ASOCIAL VARIABLES... THINK ABOUT HOW YOU MIGHT DO THIS 20120822 (Theoni comment)
    # Will: I just put a dummy asoc variable in at the start with all 0s, when the model is fitted using oadaFit the constraints and offsets vector are
    # automatically modified to ignore the dummy variable (a zero is appended to the end of each, or 2 zeroes if type=unconstrained is specified)
    
    ############# to prevent errors when nAcq is zero
    if(nAcq==0){
      learnAsoc <-learnInt<-multiInt<- NA
      id <- id # this will be NA by default
    } else {
      #############
      
      # calculate the asocial learning variables for the learning individual at each step (at each acquisition event)
      learnAsoc <- matrix(nrow=length(asoc_ilv), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnAsoc) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
      learnInt <- matrix(nrow=length(int_ilv), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnInt) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
      learnMulti<- matrix(nrow=length(multi_ilv), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnMulti) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
      learnRE<- matrix(nrow=length(random_effects), ncol=nAcq) # a matrix with as many rows as individuals and as many columns as acquisition events
      dimnames(learnRE) <- list(NULL, c(paste("id",c(orderAcq),"time",c(1:nAcq),sep="")))
      
    } # closes else
    
    if(dim(eval(as.name(asoc_ilv[1])))[2]==(nAcq+1)|dim(eval(as.name(asoc_ilv[1])))[2]==1){
      for(a in 1:length(asoc_ilv)){ # Loop through asocial variables - a loop
        asoc_ilv.array[,,a] <- eval(as.name(asoc_ilv[a])) # evaluate each one in turn
      }
    }else{
      # If no values are provided for the final period to endTime, the values are assumed to be the same as for the final event
      for(a in 1:length(asoc_ilv)){ # Loop through asocial variables - a loop
        asoc_ilv.array[,1:nAcq,a] <- eval(as.name(asoc_ilv[a])) # evaluate each one in turn
        asoc_ilv.array[,nAcq+1,a] <- asoc_ilv.array[,nAcq,a]
      }
    }
    
    
    if(dim(eval(as.name(int_ilv[1])))[2]==(nAcq+1)|dim(eval(as.name(int_ilv[1])))[2]==1){
      for(a in 1:length(int_ilv)){ # Loop through asocial variables - a loop
        int_ilv.array[,,a] <- eval(as.name(int_ilv[a])) # evaluate each one in turn
      }
    }else{
      # If no values are provided for the final period to endTime, the values are assumed to be the same as for the final event
      for(a in 1:length(int_ilv)){ # Loop through asocial variables - a loop
        int_ilv.array[,1:nAcq,a] <- eval(as.name(int_ilv[a])) # evaluate each one in turn
        int_ilv.array[,nAcq+1,a] <- int_ilv.array[,nAcq,a]
      }
    }
    
    if(dim(eval(as.name(multi_ilv[1])))[2]==(nAcq+1)|dim(eval(as.name(multi_ilv[1])))[2]==1){
      for(a in 1:length(multi_ilv)){ # Loop through asocial variables - a loop
        multi_ilv.array[,,a] <- eval(as.name(multi_ilv[a])) # evaluate each one in turn
      }
    }else{
      # If no values are provided for the final period to endTime, the values are assumed to be the same as for the final event
      for(a in 1:length(multi_ilv)){ # Loop through asocial variables - a loop
        multi_ilv.array[,1:nAcq,a] <- eval(as.name(multi_ilv[a])) # evaluate each one in turn
        multi_ilv.array[,nAcq+1,a] <- multi_ilv.array[,nAcq,a]
      }
    }
    
    if(dim(eval(as.name(random_effects[1])))[2]==(nAcq+1)|dim(eval(as.name(random_effects[1])))[2]==1){
      for(a in 1:length(random_effects)){ # Loop through asocial variables - a loop
        random_effects.array[,,a] <- eval(as.name(random_effects)) # evaluate each one in turn
      }
    }else{
      # If no values are provided for the final period to endTime, the values are assumed to be the same as for the final event
      for(a in 1:length(random_effects)){ # Loop through asocial variables - a loop
        random_effects.array[,1:nAcq,a] <- eval(as.name(random_effects[a])) # evaluate each one in turn
        random_effects.array[,nAcq+1,a] <- random_effects.array[,nAcq,a]
      }
    }
    
    
    
    #  if(nAcq!=0){
    for (i in 1:(nAcq+1)){ # Loop through acquisition events - i loop
      
      if(i<=nAcq){
        #exclude final period where no one learned
        k <- ifelse(asocialTreatment=="constant", 1, i) # this makes sure you are treating ILVs correctly if they are constant and if they are time-varying
        
        for(a in 1:length(asoc_ilv)){ # Loop through asocial variables - a loop
          learnAsoc[a,i] <- asoc_ilv.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        for(a in 1:length(int_ilv)){ # Loop through asocial variables - a loop
          learnInt[a,i] <- int_ilv.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        for(a in 1:length(multi_ilv)){ # Loop through asocial variables - a loop
          learnMulti[a,i] <- multi_ilv.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        for(a in 1:length(random_effects)){ # Loop through asocial variables - a loop
          learnRE[a,i] <- random_effects.array[orderAcq[i],k,a] # fill the matrix with the rows of the asoc matrix that correspond to the individual that learned at each acquisition event. each column is an individual, each row is a type of variable
        }
        
        statusMatrix[orderAcq[i],c((i+1):(nAcq+1))] <- 1 # give the individuals that acquired the trait a status of 1 and carry skilled status (1) through to all following acquisition events
        
        
        # if the event is recorded as tied with the previous event (ties[i]==1), it means that whoever learned in the previous event cannot be learned from for this event
        # therefore if a tie is present for event i, we do not update the availabilityMatrix to match the statusMatrix until the ties is ended
        if (ties[i]==0){
          availabilityMatrix[,i] <- statusMatrix[,i]
        } else {
          availabilityMatrix[,i] <- availabilityMatrix[,i-1]
        } # closes ties if statement
      }
      
      
      #Now correct the availabilityMatrix such that individuals who are not present for an event cannot be learned from
      availabilityMatrix<-availabilityMatrix*presenceMatrix
      
      naive.id[[i]] <- which(statusMatrix[,i]==0) # index for naive individuals before the ith acquisition event
      
      
    } # closes the i loop - nAcq (k is i or 1)
    availabilityMatrix[,nAcq+1] <- statusMatrix[,nAcq+1]
    #  } # closes the if statement for nAcq!=0
    
    
    if(is.na(id[1])) {id <- paste(label,c(unlist(naive.id)), sep="_")} # id of naive individuals before each acquisition event, including demonstrators
    
    naive <- dim(assMatrix)[1]-apply(statusMatrix, 2, sum) # number of naive individuals remaining after each acq event (last event will be the end of the diffusion for TADA data with incomplete diffusion)
    
    
    # work out the number of association matrices provided and set up stMetric matrix accordingly
    stMetric <- matrix(data=0, nrow=length(id), ncol=dim(assMatrix)[3])
    dimnames(stMetric) <- list(NULL, paste("stMetric",c(1:dim(assMatrix)[3]),sep=""))
    
    #############################################################
    # Loop through acquisition events - learner loop
    # learnMetric is the sum of network connections of individuals that learned at each acquisition events, to other informed individuals.
    # This will always be 0 for the first animal that learned unless there were demonstrators
    
    # time1 and time2 index the time period or "event period" corresponding to acquisitions
    #if(nAcq!=0){ I cut this since it should work without now I have changed nAcq to nAcq+1 for TADA
    for(event in 1:(nAcq+1)){
      # it's a shame to have two identical loops but I need time1 and time2 to be ready for use when I come to calculate the social transmission metrics below
      time1 <- c(time1, rep(event-1, naive[event]))
      time2 <- c(time2, rep(event, naive[event]))
      if(is.na(event.id[1])){
        event.id.temp <- c(event.id.temp, rep(event, each=length(naive.id[[event]])))
      }
    } # closes for loop through events
    
    timeAcqNew<-c(0,timeAcq,endTime)
    TADAtime1<-timeAcqNew[time1+1]
    TADAtime2<-timeAcqNew[time2+1]
    
    
    if(is.na(event.id[1])) {event.id <- paste(label, event.id.temp, sep="_")}
    
    for(event in 1:(nAcq+1)){ # event indexes the number of the acquisition event- increased by 1 for TADA to allow for the endTime period
      
      #Take the appropriate association matrix from the (weighted) time varying association matrix,
      #as determined for that event by the assMatrixIndex vector
      if((length(assMatrixIndex)==nAcq)&(event==(nAcq+1))){
        #If no separate assMatrix is specified for the end period it is assumed to be the same as for the final acquisition event
        assMatrix<-array(assMatrixTV[,,, assMatrixIndex[nAcq]],dim=dim(assMatrixTV)[1:3])
      }else{
        assMatrix<-array(assMatrixTV[,,, assMatrixIndex[event]],dim=dim(assMatrixTV)[1:3])
      }
      
      learner <- orderAcq[event] # learner is individual id of the animal that learned AT an event
      nonlearners <- naive.id[[event]] # nonlearners are individual id of the animals that were naive BEFORE an event
      
      if(length(nonlearners)>0){
        #If everyone has learned by the final period of a TADA (i.e. up to endTime) the next section triggers errors- and we do not need a final period
        
        status <- c(status, statusMatrix[unlist(naive.id[[event]]), min(nAcq+1,event+1)])
        presentInDiffusion<-c(presentInDiffusion,presenceMatrix[unlist(naive.id[[event]]), min(nAcq+1,event+1)])
        
        temp.stMetric <- vector() # reset this before the metrics for each event are calculated
        
        for (nonlearner in nonlearners){
          
          # stMetric is the total assoc of the individuals that did NOT learn
          # by that acquisition event, with all other already-informed individuals
          
          m1 <- matrix(data=assMatrix[nonlearner,,], nrow=dim(assMatrix)[3], byrow=T) # rowMatrix 
          m2 <- (weights*availabilityMatrix[,event])*t(m1) # matrix2
          v1 <- apply(X=m2, MARGIN=2, FUN=sum) # vector1 of rowsums
          temp.stMetric <- rbind(temp.stMetric, v1)
          
        } # closes nonlearner loop for MATRIX stMetric
        
        stMetric[time2==event,] <- temp.stMetric
        
        
        if(asoc_ilv[1]=="ILVabsent"){
          ilv1 <-cbind("ILVabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            ilv1 <- matrix(asoc_ilv.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            ilv1 <- matrix(asoc_ilv.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        if(int_ilv[1]=="ILVabsent"){
          intilv1 <-cbind("ILVabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            intilv1 <- matrix(int_ilv.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            intilv1 <- matrix(int_ilv.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        if(multi_ilv[1]=="ILVabsent"){
          multiilv1 <-cbind("ILVabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            multiilv1 <- matrix(multi_ilv.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            multiilv1 <- matrix(multi_ilv.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
        
        if(random_effects[1]=="REabsent"){
          randomeffect1 <-cbind("REabsent"=rep(0,length(nonlearners)))
        }else{
          if(asocialTreatment=="constant"){
            randomeffect1 <- matrix(random_effects.array[nonlearners, 1,],nrow=length(nonlearners))
          }else{
            randomeffect1 <- matrix(random_effects.array[nonlearners, event,],nrow=length(nonlearners))
          }
        }# this makes sure the right column out of the asoc.array is used
      
        
        
        
        asocILVdata.naive <- rbind(asocILVdata.naive, ilv1)
        if(asoc_ilv[1]=="ILVabsent"){
          attr(asocILVdata.naive, "dimnames") <- list(NULL,"ILVabsent")
        }else{
          attr(asocILVdata.naive, "dimnames") <- list(NULL,asoc_ilv)
        }
        
        intILVdata.naive <- rbind(intILVdata.naive, intilv1)
        if(int_ilv[1]=="ILVabsent"){
          attr(intILVdata.naive, "dimnames") <- list(NULL,"ILVabsent")
        }else{
          attr(intILVdata.naive, "dimnames") <- list(NULL,int_ilv)
        }
        
        multiILVdata.naive <- rbind(multiILVdata.naive, multiilv1)
        if(multi_ilv[1]=="ILVabsent"){
          attr(multiILVdata.naive, "dimnames") <- list(NULL,"ILVabsent")
        }else{
          attr(multiILVdata.naive, "dimnames") <- list(NULL,multi_ilv)
        }
        randomEffectdata.naive <- rbind(randomEffectdata.naive, randomeffect1)
        if(random_effects[1]=="REabsent"){
          attr(randomEffectdata.naive, "dimnames") <- list(NULL,"REabsent")
        }else{
          attr(randomEffectdata.naive, "dimnames") <- list(NULL,random_effects)
        }
      }#closes if(length(nonlearners)>0) loop
      
    } # closes event loop
    
    #############################################################
    label <- rep(label, length.out=length(id))
    
    if(is.null(demons)) demons <- NA;
    
    #Subtract the first column from presenceMatrix (added previously) so it again gives the presence of each individual for each event
    presenceMatrix<-presenceMatrix[,-1]
    
    if(is.null(offsetCorrection)) offsetCorrection <- cbind(rep(0,dim(asocILVdata.naive)[1]),rep(0,dim(asocILVdata.naive)[1]),rep(0,dim(asocILVdata.naive)[1]),rep(0,dim(asocILVdata.naive)[1]));
    dimnames(offsetCorrection)[2]<-list(c("SocialOffsetCorr","AsocialILVOffsetCorr","InteractionOffsetCorr","MultiplicativeILVOffsetCorr"))
    
    callNextMethod(.Object, label=label, idname=idname, assMatrix=assMatrixTV, asoc_ilv=asoc_ilv, int_ilv=int_ilv,multi_ilv=multi_ilv,random_effects=random_effects, orderAcq=orderAcq, timeAcq=timeAcq, endTime=endTime,updateTimes=NA, ties=ties, trueTies=trueTies, demons=demons, weights=weights, statusMatrix=statusMatrix, availabilityMatrix=availabilityMatrix, event.id=event.id, id=id, time1=time1, time2=time2,TADAtime1=TADAtime1, TADAtime2=TADAtime2, status=status, presentInDiffusion= presentInDiffusion, presenceMatrix = presenceMatrix ,asocialTreatment=asocialTreatment, stMetric=stMetric, asocILVdata=asocILVdata.naive, intILVdata=intILVdata.naive, multiILVdata=multiILVdata.naive,randomEffectdata=randomEffectdata.naive,offsetCorrection=offsetCorrection,assMatrixIndex=assMatrixIndex)
    
  }
} # end function

# generate a matrix that contains the status of each individual at each acquisition event
statusMatrix <- matrix(0, nrow=dim(assMatrix)[2], ncol=1+nAcq)  # a matrix with as many rows as indivs and as many columns as acquisition events PLUS one for the demonstrators
# create a list vector to hold the index of naive individuals after each acquisition event
naive.id <-naive.id.names<- vector(mode="list", length=nAcq)

# Bayesian version-------------------------------------------------------------------------
## Setup data, if no rf and no fixed effect then only S metric as matrix of its values stored as a matrix wehre rows = time and columns individuals-----------------------------
library(bayesNBDA)
nbdadata = oa.fit_social@nbdadata
i<-1
tempNBDAdata<-nbdadata[[i]]

noSParams<-dim(tempNBDAdata@stMetric)[2]
dataTemplate<-matrix(0,ncol=length(unique(tempNBDAdata@id)),nrow=length(unique(tempNBDAdata@event.id)))
dimnames(dataTemplate)[[1]]<-unique(tempNBDAdata@event.id)
dimnames(dataTemplate)[[2]]<-unique(tempNBDAdata@id)
availabilityToLearn<-status<-dataTemplate
stMetric<-array(0,dim=c(dim(dataTemplate),dim(tempNBDAdata@stMetric)[2]))
asocILVdata<-array(0,dim=c(dim(dataTemplate),dim(tempNBDAdata@asocILVdata)[2]))
intILVdata<-array(0,dim=c(dim(dataTemplate),dim(tempNBDAdata@intILVdata)[2]))
multiILVdata<-array(0,dim=c(dim(dataTemplate),dim(tempNBDAdata@multiILVdata)[2]))
randomEffectdata<-array(0,dim=c(dim(dataTemplate),dim(tempNBDAdata@randomEffectdata)[2]))
offsetMatrix<-array(0,dim=c(dim(dataTemplate),4))

for(j in 1: length(tempNBDAdata@id)){
  index1<-which(tempNBDAdata@event.id[j]==unique(tempNBDAdata@event.id))
  index2<-which(tempNBDAdata@id[j]==unique(tempNBDAdata@id))
  
  stMetric[index1,index2,]<-tempNBDAdata@stMetric[j,]
  asocILVdata[index1,index2,]<-tempNBDAdata@asocILVdata[j,]
  intILVdata[index1,index2,]<-tempNBDAdata@intILVdata[j,]
  multiILVdata[index1,index2,]<-tempNBDAdata@multiILVdata[j,]
  randomEffectdata[index1,index2,]<-tempNBDAdata@randomEffectdata[j,]
  availabilityToLearn[index1,index2]<-1
  status[index1,index2]<-tempNBDAdata@status[j]
  offsetMatrix[index1,index2,]<-tempNBDAdata@offsetCorrection[j,]
}

dataLength<-maxNoInd<-0
for(i in 1:length(nbdadata)){
  dataLength<-dataLength+length(unique(nbdadata[[i]]@event.id))
  maxNoInd<-max(maxNoInd,length(unique(nbdadata[[i]]@id)))
}

stMetric_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd,dim(stMetric)[3]))
asocILVdata_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd,dim(asocILVdata)[3]))
intILVdata_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd,dim(intILVdata)[3]))
multiILVdata_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd,dim(multiILVdata)[3]))
randomEffectdata_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd,dim(randomEffectdata)[3]))
availabilityToLearn_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd))
status_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd))
offsetMatrix_allDiffusions<-array(NA,dim=c(dataLength,maxNoInd,4))

index3<-1
index4<-dim(status)[1]

stMetric_allDiffusions[index3:index4,1:dim(stMetric)[2],]<-stMetric
asocILVdata_allDiffusions[index3:index4,1:dim(stMetric)[2],]<-asocILVdata
intILVdata_allDiffusions[index3:index4,1:dim(stMetric)[2],]<-intILVdata
multiILVdata_allDiffusions[index3:index4,1:dim(stMetric)[2],]<-multiILVdata
randomEffectdata_allDiffusions[index3:index4,1:dim(stMetric)[2],]<-randomEffectdata
availabilityToLearn_allDiffusions[index3:index4,1:dim(stMetric)[2]]<-availabilityToLearn
status_allDiffusions[index3:index4,1:dim(stMetric)[2]]<-status
offsetMatrix_allDiffusions[index3:index4,1:dim(stMetric)[2],]<-offsetMatrix

randomEffectsLevels<-max(randomEffectdata_allDiffusions,na.rm=T)
noEvents<-dim(status_allDiffusions)[1]

#If there are unequal numbers of individuals in each diffusion then there will be NAs appearing in
#the phantom slots for non-existent individuals. We need to replace all these with zeros to avoid errors
for(i in 1:dim(stMetric_allDiffusions)[3]) stMetric_allDiffusions[,,i][is.na(availabilityToLearn_allDiffusions)]<-0
for(i in 1:dim(asocILVdata_allDiffusions)[3]) asocILVdata_allDiffusions[,,i][is.na(availabilityToLearn_allDiffusions)]<-0
for(i in 1:dim(intILVdata_allDiffusions)[3]) intILVdata_allDiffusions[,,i][is.na(availabilityToLearn_allDiffusions)]<-0
for(i in 1:dim(multiILVdata_allDiffusions)[3]) multiILVdata_allDiffusions[,,i][is.na(availabilityToLearn_allDiffusions)]<-0
for(i in 1:dim(randomEffectdata_allDiffusions)[3]) randomEffectdata_allDiffusions[,,i][is.na(availabilityToLearn_allDiffusions)]<-0
for(i in 1:dim(offsetMatrix_allDiffusions)[3]) offsetMatrix_allDiffusions[,,i][is.na(availabilityToLearn_allDiffusions)]<-0
status_allDiffusions[is.na(availabilityToLearn_allDiffusions)]<-0
availabilityToLearn_allDiffusions[is.na(availabilityToLearn_allDiffusions)]<-0
#Since the availabilityToLearn=0 for all these slots, the data is ignored when fitting the model

imputedAsocILVs<-which(apply(is.na(asocILVdata_allDiffusions),3,sum)>0)
if(length(imputedAsocILVs)>0){
  imputationAsocILVs<-matrix(0,nrow=randomEffectsLevels+1,ncol=length(imputedAsocILVs))
  for(i in 2:(randomEffectsLevels+1)){
    for(j in 1:length(imputedAsocILVs)){
      imputationAsocILVs[i,j]<-asocILVdata_allDiffusions[1,i,imputedAsocILVs[j]]
    }
  }
  dimnames(imputationAsocILVs)[[2]]<-nbdadata[[1]]@int_ilv[imputedAsocILVs]
}else{imputationAsocILVs<-NULL}


imputedIntILVs<-which(apply(is.na(intILVdata_allDiffusions),3,sum)>0)
if(length(imputedIntILVs)>0){
  imputationIntILVs<-matrix(0,nrow=randomEffectsLevels+1,ncol=length(imputedIntILVs))
  for(i in 2:(randomEffectsLevels+1)){
    for(j in 1:length(imputedIntILVs)){
      imputationIntILVs[i,j]<-intILVdata_allDiffusions[1,i,imputedIntILVs[j]]
    }
  }
  dimnames(imputationIntILVs)[[2]]<-nbdadata[[1]]@int_ilv[imputedIntILVs]
}else{imputationIntILVs<-NULL}

imputedMultiILVs<-which(apply(is.na(multiILVdata_allDiffusions),3,sum)>0)
if(length(imputedMultiILVs)>0){
  imputationMultiILVs<-matrix(0,nrow=randomEffectsLevels+1,ncol=length(imputedMultiILVs))
  for(i in 2:(randomEffectsLevels+1)){
    for(j in 1:length(imputedMultiILVs)){
      imputationMultiILVs[i,j]<-multiILVdata_allDiffusions[1,i,imputedMultiILVs[j]]
    }
  }
  dimnames(imputationMultiILVs)[[2]]<-nbdadata[[1]]@int_ilv[imputedMultiILVs]
}else{imputationMultiILVs<-NULL}

imputationILVs<-cbind(imputationAsocILVs,imputationIntILVs,imputationMultiILVs)

oada_jagsData<-list(
  noSParams=noSParams,
  noAsocParams=noAsocParams,
  noIntParams=noIntParams,
  noMultiParams=noMultiParams,
  noRandomEffects=noRandomEffects,
  randomEffectsLevels=randomEffectsLevels,
  noEvents=noEvents,
  maxNoInd=maxNoInd,
  
  stMetric=stMetric_allDiffusions,
  asocILVdata=asocILVdata_allDiffusions,
  intILVdata=intILVdata_allDiffusions,
  multiILVdata=multiILVdata_allDiffusions,
  randomEffectdata=randomEffectdata_allDiffusions,
  availabilityToLearn=availabilityToLearn_allDiffusions,
  status=status_allDiffusions,
  offsetMatrix=offsetMatrix_allDiffusions)


# JAGSoadaModel: generate JAGS file --------------------------------------------------------
JAGSoadaDataIn = oada_jagsData
randomModel=T
upperS=1000
asocPriorVar=1000
intPriorVar=10000
multiPriorVar=10000
REhyperPriorUpper=10

noSParams<-JAGSoadaDataIn$noSParams
noAsocParams<-JAGSoadaDataIn$noAsocParams
noIntParams<-JAGSoadaDataIn$noIntParams
noMultiParams<-JAGSoadaDataIn$noMultiParams
noRandomEffects<-JAGSoadaDataIn$noRandomEffects
randomEffectsLevels<-JAGSoadaDataIn$randomEffectsLevels
noEvents<-JAGSoadaDataIn$noEvents
maxNoInd<-JAGSoadaDataIn$maxNoInd

sPriors<-paste("\n\ts[",1:noSParams,"]~dunif(0,",upperS,")",sep="")
unscaledST<-paste("+s[",1:noSParams,"]*stMetric[j,k,",1:noSParams,"]",sep="",collapse = "")
if(noAsocParams==0){
  asocPriors<-NULL
  asocialLP<-"+0"
}else{
  asocPriors<-paste("\n\tbetaAsoc[",1:noAsocParams,"]~dnorm(0,",1/asocPriorVar,")",sep="")
  asocialLP<-paste("+betaAsoc[",1:noAsocParams,"]*asocILVdata[j,k,",1:noAsocParams,"]",sep="",collapse = "")
}
if(noIntParams==0){
  intPriors<-NULL
  intLP<-"+0"
}else{
  intPriors<-paste("\n\tbetaInt[",1:noIntParams,"]~dnorm(0,",1/intPriorVar,")",sep="")
  intLP<-paste0("+betaInt[",1:noIntParams,"]*intILVdata[j,k,",1:noIntParams,"]",sep="",collapse = "")
}
if(noMultiParams==0){
  multiPriors<-NULL
  multiLP<-NULL
}else{
  multiPriors<-paste("\n\tbetaMulti[",1:noMultiParams,"]~dnorm(0,",1/multiPriorVar,")",sep="")
  multiLP<-paste("+betaMulti[",1:noMultiParams,"]*asocILVdata[j,k,",1:noMultiParams,"]",sep="",collapse = "")
}
if(noRandomEffects==0|!randomModel){
  REPriors<-NULL
  sampleRandomEffectsFirst<-NULL
  sampleRandomEffects<-paste("\n\tre[1,j]<-0",sep="")
  multiLP_RE<-NULL
  #This just fills in a couple of entries for the random effects which is not used in the model anyway
  RElevelNumber="2"
}else{
  REPriors<-paste(paste("\n\tsigma[",1:noRandomEffects,"]~dunif(0,",REhyperPriorUpper,")",sep=""),
                  paste("\n\ttau[",1:noRandomEffects,"]<-1/(sigma[",1:noRandomEffects,"]*sigma[",1:noRandomEffects,"])",sep=""))
  sampleRandomEffectsFirst<-paste("\n\tre[",1:noRandomEffects,",1]<-0",sep="")
  sampleRandomEffects<-paste("\n\tre[",1:noRandomEffects,",j]~dnorm(0,tau[",1:noRandomEffects,"])",sep="")
  multiLP_RE<-paste("+re[",1:noRandomEffects,",(randomEffectdata[j,k,",1:noRandomEffects,"]+1)]",sep="",collapse = "")
  RElevelNumber="(randomEffectsLevels+1)"
}
if(noMultiParams==0&noRandomEffects==0){
  multiLP<-"+0"
}else{
  multiLP<-paste(multiLP,multiLP_RE,sep="",collapse = "")
}

#Specify the model in JAGS format (saves as a text file)
modelFileName = 'nbdaBayesian'
sink(modelFileName)
cat("
model{
    #1. Priors

    #Uniform prior for S parameters",
    sPriors,
    "

    #Normal priors for ILV parameters
    #Effect of ILVs on asocial learning",asocPriors,"
    #Effect of ILVs on social learning",intPriors,"
    #Multiplicative ILVs (asocial effect = social effect)",multiPriors,"

    #Random effects", REPriors,"
    ", sampleRandomEffectsFirst,
    "
    for(j in 2:",RElevelNumber,"){",sampleRandomEffects,"
    }


    for(j in 1:noEvents){
      for(k in 1:maxNoInd){
        #Get the linear predictor for ILV effects on asocial and social learning
        asocialLP[j,k]<-offsetMatrix[j,k,2]",asocialLP,"
        intLP[j,k]<-offsetMatrix[j,k,3]",intLP,"
        multiLP[j,k]<-offsetMatrix[j,k,4]",multiLP,"

        #Get the unscaled social transmission rate across all networks
        unscaledST[j,k]<-offsetMatrix[j,k,1]",unscaledST,"

        #Get the relative rate of learning

      relativeRate[j,k]<-(exp(asocialLP[j,k]+multiLP[j,k])+exp(intLP[j,k]+multiLP[j,k])*unscaledST[j,k])*availabilityToLearn[j,k]
      }

    }

    for(j in 1:noEvents){
      for(k in 1:maxNoInd){
        probs[j,k]<-relativeRate[j,k]/sum(relativeRate[j,1:maxNoInd])
        #A node used to get WAIC
        pYgivenThetaTemp[j,k]<-probs[j,k]*status[j,k]
      }
      status[j,1:maxNoInd]~ dmulti(probs[j,1:maxNoInd],1)
      #A node used to get WAIC (lppd and effective parameters)
      pYgivenTheta[j]<-sum(pYgivenThetaTemp[j,1:maxNoInd])
    }

    #Calculate propST
    #The log transformations are used here for numerical stability when rates are high, notably when getting the prior for propST

    for(j in 1:noEvents){
      for(k in 1:maxNoInd){
        learnersRateTemp[j,k]<-relativeRate[j,k]*status[j,k]
      }
      learnersRate[j]<-sum(learnersRateTemp[j,1:maxNoInd])
      for(l in 1:noSParams){
          for(k in 1:maxNoInd){
             logLearnerSocialRateTemp[j,k,l]<-(log(s[l])+log(stMetric[j,k,l])+(intLP[j,k]+multiLP[j,k]))*status[j,k]
          }
          loglearnerSocialRate[j,l]<-sum(logLearnerSocialRateTemp[j,1:maxNoInd,l])
	  logProbST[j,l]<-loglearnerSocialRate[j,l]- log(learnersRate[j])
          probST[j,l]<-exp(logProbST[j,l])
      }
    }
    for(l in 1:noSParams){
      propST[l]<-sum(probST[1:noEvents,l])/noEvents
    }
  }
",fill=FALSE)
sink()
# Model looks like r code-----------------------------
#1. Priors

#Uniform prior for S parameters 
s = dunif(1, 0,1000) 
re = matrix(0, 2,2)
for(j in 2: 2 ){ 
  re[1,j]<-0 
}

list2env(JAGSoadaDataIn, envir = .GlobalEnv)
asocialLP = intLP = multiLP = unscaledST = relativeRate = matrix(0, noEvents,maxNoInd)

for(j in 1:noEvents){
  for(k in 1:maxNoInd){
    #Get the linear predictor for ILV effects on asocial and social learning
    asocialLP[j,k]<-offsetMatrix[j,k,2] +0 
    intLP[j,k]<-offsetMatrix[j,k,3] +0 
    multiLP[j,k]<-offsetMatrix[j,k,4] +0 
    
    #Get the unscaled social transmission rate across all networks
    unscaledST[j,k]<- s*stMetric[j,k,1] #+ offsetMatrix[j,k,1] 
    
    #Get the relative rate of learning
    
    relativeRate[j,k]<-(exp(asocialLP[j,k]+multiLP[j,k])+
                          exp(intLP[j,k]+multiLP[j,k])*unscaledST[j,k])*availabilityToLearn[j,k]
  }
  
}

for(j in 1:noEvents){
  for(k in 1:maxNoInd){
    probs[j,k]<-relativeRate[j,k]/sum(relativeRate[j,1:maxNoInd])
    #A node used to get WAIC
    pYgivenThetaTemp[j,k]<-probs[j,k]*status[j,k]
  }
  status[j,1:maxNoInd]~ dmulti(probs[j,1:maxNoInd],1)
  #A node used to get WAIC (lppd and effective parameters)
  pYgivenTheta[j]<-sum(pYgivenThetaTemp[j,1:maxNoInd])
}

#Calculate propST
#The log transformations are used here for numerical stability when rates are high, notably when getting the prior for propST

for(j in 1:noEvents){
  for(k in 1:maxNoInd){
    learnersRateTemp[j,k]<-relativeRate[j,k]*status[j,k]
  }
  learnersRate[j]<-sum(learnersRateTemp[j,1:maxNoInd])
  for(l in 1:noSParams){
    for(k in 1:maxNoInd){
      logLearnerSocialRateTemp[j,k,l]<-(log(s[l])+log(stMetric[j,k,l])+(intLP[j,k]+multiLP[j,k]))*status[j,k]
    }
    loglearnerSocialRate[j,l]<-sum(logLearnerSocialRateTemp[j,1:maxNoInd,l])
    logProbST[j,l]<-loglearnerSocialRate[j,l]- log(learnersRate[j])
    probST[j,l]<-exp(logProbST[j,l])
  }
}
for(l in 1:noSParams){
  propST[l]<-sum(probST[1:noEvents,l])/noEvents
}
