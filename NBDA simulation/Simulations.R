
LogLik<-function(par,oa=orderAcq,sn=dynNet){
	LogLik<-0
	for(i in 1:100) LogLik<-LogLik-log(par*dynNet[i,orderAcq[i]]+1)+log(sum((par*dynNet[i,]+1)*(1-zRec[i,])))
	return(LogLik)
}


propST<-function(par,oa=orderAcq,sn=dynNet){
	prob<-rep(NA,100)
	for(i in 1:100) prob[i]<-par*dynNet[i,orderAcq[i]]/(par*dynNet[i,orderAcq[i]]+1)
	return(mean(prob))
}

profLikDiff<-function(par,modelLik,oa=orderAcq,sn=dynNet){
	abs(LogLik(par)-modelLik-1.92)
}



noSims<-noSim<-10000

pLearnVect<-c(0,0.05,0.1,0.15,0.2)
pRec<-sRec<-propSTRec<-realSTRec <-upperCI<-lowerCI<-upperProp<-lowerProp<-matrix(NA,noSims,length(pLearnVect))
B<-1
pObs<-matrix(0.15,nrow=100,ncol=100)

for(l in 1:length(pLearnVect)){
pLearn<-pLearnVect[l]
		
for(k in 1:noSims){
z<-rep(0,100)
orderAcq<-NULL
noObs<-rep(0,100)
dynNet<-zRec<-NULL
asocialLearns<-socialLearns<-0


while(sum(z)<100){
learnRate<-rep(0.2,100)*(1-z)
AsocialLearnTimes<-rexp(100,learnRate)
AsocialLearnTimes[is.nan(AsocialLearnTimes)]<-Inf
performRate<-B*z
PerformTimes<-rexp(100,performRate)
PerformTimes[is.nan(PerformTimes)]<-Inf

if(min(AsocialLearnTimes)<min(PerformTimes)){
	zRec<-rbind(zRec,z)
	z[which(AsocialLearnTimes==min(AsocialLearnTimes))]<-1
	orderAcq<-c(orderAcq,which(AsocialLearnTimes==min(AsocialLearnTimes)))
	dynNet<-rbind(dynNet,noObs)
	asocialLearns<-asocialLearns+1
}else{
	obsProbs<-pObs[,which(PerformTimes ==min(PerformTimes))]*(1-z)
	obs<-rbinom(100,1, obsProbs)
	noObs<-obs+noObs
	pSTrans<-obs*pLearn
	learned<-rbinom(100,1, pSTrans)

	orderAcq<-c(orderAcq,which(learned==1))
	if(sum(learned)>0){for (j in 1:length(which(learned==1))){
			dynNet<-rbind(dynNet,noObs)
			zRec<-rbind(zRec,z)
	}}
		z<-z+learned
	socialLearns<-socialLearns+length(which(learned==1))
}
}


model<-optimise(f=LogLik,interval=c(0,99999999))
LogLik(0)
pRec[k,l]<-pchisq(2*(LogLik(0)-model$objective),1,lower.tail=F)
sRec[k,l]<-model$minimum
propSTRec[k,l]<-propST(sRec[k,l])
realSTRec[k,l]<-socialLearns/100
if(pRec[k,l]>0.05){lowerCI[k,l]<-0}else{lowerCI[k,l]<-optimise(profLikDiff,interval=c(0,model$minimum), modelLik=model$objective)$minimum}
upperCI[k,l]<-optimise(profLikDiff,interval=c(model$minimum,99999999), modelLik=model$objective)$minimum
upperProp[k,l]<-propST(upperCI[k,l])
lowerProp[k,l]<-propST(lowerCI[k,l])

}
mean(pRec<0.05)

plot(propSTRec[,l], realSTRec[,l])
mean(propSTRec[,l]-realSTRec[,l])
mean(abs(propSTRec[,l]-realSTRec[,l]))
}

par(mfrow=c(2,2))
for(l in 2:length(pLearnVect)){
	#plot(realSTRec[,l],propSTRec[,l],xlim=c(0,1),ylim=c(0,1))
	mean(propSTRec[,l]-realSTRec[,l])
	mean(abs(propSTRec[,l]-realSTRec[,l]))
	#abline(a=0,b=1,col=2)
}

apply(pRec<0.05,2,mean)

apply(realSTRec,2,mean)
apply(propSTRec,2,mean)
apply(upperProp,2,mean)
apply(propSTRec-realSTRec,2,mean)

apply((realSTRec> lowerProp)&(realSTRec<upperProp),2,sum)/noSim
apply((realSTRec> upperProp),2,sum)/noSim
apply((realSTRec< lowerProp),2,sum)/noSim

plot(realSTRec,propSTRec,xlim=c(0,1),ylim=c(0,1))
cor(as.vector(realSTRec),as.vector(propSTRec))
#[1] 0.833727
lm(as.vector(propSTRec)~as.vector(realSTRec))

plot(realSTRec[pRec<0.05],propSTRec[pRec<0.05],xlim=c(0,1),ylim=c(0,1))
cor(as.vector(realSTRec[pRec<0.05]),as.vector(propSTRec[pRec<0.05]))
#[1] 0.833727
lm(as.vector(propSTRec[pRec<0.05])~as.vector(realSTRec[pRec<0.05]))

save(pRec,sRec,propSTRec,realSTRec ,upperCI,lowerCI,upperProp,lowerProp,file="sims1")
load("sims1")


summarySignificant<-summaryRealSTRec<-summaryPropSTRec<-summaryUpperProp<-summaryDiff<-InCI<-overCI<-underCI<-matrix(NA,nrow=length(pobsRobsVect),ncol=2)

for(m in 1:length(pobsRobsVect)){
#for(m in 1:6){
apply(pRec[,,m,1]<0.05,2,mean)-> summarySignificant[m,]
apply(realSTRec[,,m,1],2,mean)-> summaryRealSTRec[m,]
apply(propSTRec[,,m,1],2,mean)-> summaryPropSTRec[m,]
apply(upperProp[,,m,1],2,mean)-> summaryUpperProp[m,]
apply((propSTRec-realSTRec)[,,m,1],2,mean)-> summaryDiff[m,]

apply((realSTRec[,,m,1]> lowerProp[,,m,1])&(realSTRec[,,m,1]<upperProp[,,m,1]),2,sum)/noSims-> InCI[m,]
apply((realSTRec[,,m,1]> upperProp[,,m,1]),2,sum)/noSims-> overCI[m,]
apply((realSTRec[,,m,1]< lowerProp[,,m,1]),2,sum)/noSims-> underCI[m,]

#plot(realSTRec,propSTRec,xlim=c(0,1),ylim=c(0,1))
#cor(as.vector(realSTRec),as.vector(propSTRec))
#[1] 0.833727
#lm(as.vector(propSTRec)~as.vector(realSTRec))

#plot(realSTRec[pRec<0.05],propSTRec[pRec<0.05],xlim=c(0,1),ylim=c(0,1))
#cor(as.vector(realSTRec[pRec<0.05]),as.vector(propSTRec[pRec<0.05]))
#[1] 0.833727
#lm(as.vector(propSTRec[pRec<0.05])~as.vector(realSTRec[pRec<0.05]))
}




noSims<-10000

pobsRobsVect<-c(1,0.75,0.5)
pobsNobsVect<-c(0,0.125,0.25,0.5)

pLearnVect<-c(0,0.2)
pRec<-sRec<-propSTRec<-realSTRec <-upperCI<-lowerCI<-upperProp<-lowerProp<-array(NA,dim=c(noSims,length(pLearnVect),length(pRobsobsVect),length(pRobsnoobsVect)))
B<-1
pObs<-matrix(0.15,nrow=100,ncol=100)

for(m in 1:length(pRobsobsVect)){
pobsRobs<-pobsRobsVect[m]
for(n in 1:length(pRobsnoobsVect)){
pobsNobs<-pobsNobsVect[n]

for(l in 1:length(pLearnVect)){
pLearn<-pLearnVect[l]
		
for(k in 1:noSims){
z<-rep(0,100)
orderAcq<-NULL
noObs<-rep(0,100)
dynNet<-zRec<-NULL
asocialLearns<-socialLearns<-0


while(sum(z)<100){
learnRate<-rep(0.2,100)*(1-z)
AsocialLearnTimes<-rexp(100,learnRate)
AsocialLearnTimes[is.nan(AsocialLearnTimes)]<-Inf
performRate<-B*z
PerformTimes<-rexp(100,performRate)
PerformTimes[is.nan(PerformTimes)]<-Inf

if(min(AsocialLearnTimes)<min(PerformTimes)){
	zRec<-rbind(zRec,z)
	z[which(AsocialLearnTimes==min(AsocialLearnTimes))]<-1
	orderAcq<-c(orderAcq,which(AsocialLearnTimes==min(AsocialLearnTimes)))
	dynNet<-rbind(dynNet,noObs)
	asocialLearns<-asocialLearns+1
}else{
	obsProbs<-pObs[,which(PerformTimes ==min(PerformTimes))]*(1-z)
	Robs<-rbinom(100,1, obsProbs)
	FalseNeg<-rbinom(100,1,pobsNobs)*(1-z)
	TruePos<-rbinom(100,1,pobsRobs)*(1-z)
	obs<-Robs
	obs[Robs==0]<-FalseNeg[Robs==0]
	obs[Robs==1]<-TruePos[Robs==1]	
	noObs<-Robs+noObs
	pSTrans<-obs*pLearn
	learned<-rbinom(100,1, pSTrans)

	orderAcq<-c(orderAcq,which(learned==1))
	if(sum(learned)>0){for (j in 1:length(which(learned==1))){
			dynNet<-rbind(dynNet,noObs)
			zRec<-rbind(zRec,z)
	}}
		z<-z+learned
	socialLearns<-socialLearns+length(which(learned==1))
}
}


model<-optimise(f=LogLik,interval=c(0,99999999))
LogLik(0)
pRec[k,l,m,n]<-pchisq(2*(LogLik(0)-model$objective),1,lower.tail=F)
sRec[k,l,m,n]<-model$minimum
propSTRec[k,l,m,n]<-propST(sRec[k,l,m,n])
realSTRec[k,l,m,n]<-socialLearns/100
if(pRec[k,l,m,n]>0.05){lowerCI[k,l,m,n]<-0}else{lowerCI[k,l,m,n]<-optimise(profLikDiff,interval=c(0,model$minimum), modelLik=model$objective)$minimum}
upperCI[k,l,m,n]<-optimise(profLikDiff,interval=c(model$minimum,99999999), modelLik=model$objective)$minimum
upperProp[k,l,m,n]<-propST(upperCI[k,l,m,n])
lowerProp[k,l,m,n]<-propST(lowerCI[k,l,m,n])

}
}
}
}


save(pRec,sRec,propSTRec,realSTRec ,upperCI,lowerCI,upperProp,lowerProp,file="sims2")
load("sims2")





#Now with constrained pObs


noSims<-10000

pobsRobsVect<-c(1,0.95,0.9,0.85,0.8,0.6,0.4,0.2,0.15)


pLearnVect<-c(0,0.2)
pRec<-sRec<-propSTRec<-realSTRec <-upperCI<-lowerCI<-upperProp<-lowerProp<-array(NA,dim=c(noSims,length(pLearnVect),length(pobsRobsVect),1))
B<-1
pObs<-matrix(0.15,nrow=100,ncol=100)
pobsNobsVect<-(pObs[1,1]-pObs[1,1]*pobsRobsVect)/(1-pObs[1,1])
n<-1

for(m in 1:length(pobsRobsVect)){
pobsRobs<-pobsRobsVect[m]
pobsNobs<-pobsNobsVect[m]
for(l in 1:length(pLearnVect)){
pLearn<-pLearnVect[l]
		
for(k in 1:noSims){
z<-rep(0,100)
orderAcq<-NULL
noObs<-rep(0,100)
dynNet<-zRec<-NULL
asocialLearns<-socialLearns<-0


while(sum(z)<100){
learnRate<-rep(0.2,100)*(1-z)
AsocialLearnTimes<-rexp(100,learnRate)
AsocialLearnTimes[is.nan(AsocialLearnTimes)]<-Inf
performRate<-B*z
PerformTimes<-rexp(100,performRate)
PerformTimes[is.nan(PerformTimes)]<-Inf

if(min(AsocialLearnTimes)<min(PerformTimes)){
	zRec<-rbind(zRec,z)
	z[which(AsocialLearnTimes==min(AsocialLearnTimes))]<-1
	orderAcq<-c(orderAcq,which(AsocialLearnTimes==min(AsocialLearnTimes)))
	dynNet<-rbind(dynNet,noObs)
	asocialLearns<-asocialLearns+1
}else{
	obsProbs<-pObs[,which(PerformTimes ==min(PerformTimes))]*(1-z)
	Robs<-rbinom(100,1, obsProbs)
	FalseNeg<-rbinom(100,1,pobsNobs)*(1-z)
	TruePos<-rbinom(100,1,pobsRobs)*(1-z)
	obs<-Robs
	obs[Robs==0]<-FalseNeg[Robs==0]
	obs[Robs==1]<-TruePos[Robs==1]	
	noObs<-Robs+noObs
	pSTrans<-obs*pLearn
	learned<-rbinom(100,1, pSTrans)

	orderAcq<-c(orderAcq,which(learned==1))
	if(sum(learned)>0){for (j in 1:length(which(learned==1))){
			dynNet<-rbind(dynNet,noObs)
			zRec<-rbind(zRec,z)
	}}
		z<-z+learned
	socialLearns<-socialLearns+length(which(learned==1))
}
}


model<-optimise(f=LogLik,interval=c(0,99999999))
LogLik(0)
pRec[k,l,m,n]<-pchisq(2*(LogLik(0)-model$objective),1,lower.tail=F)
sRec[k,l,m,n]<-model$minimum
propSTRec[k,l,m,n]<-propST(sRec[k,l,m,n])
realSTRec[k,l,m,n]<-socialLearns/100
if(pRec[k,l,m,n]>0.05){lowerCI[k,l,m,n]<-0}else{lowerCI[k,l,m,n]<-optimise(profLikDiff,interval=c(0,model$minimum), modelLik=model$objective)$minimum}
upperCI[k,l,m,n]<-optimise(profLikDiff,interval=c(model$minimum,99999999), modelLik=model$objective)$minimum
upperProp[k,l,m,n]<-propST(upperCI[k,l,m,n])
lowerProp[k,l,m,n]<-propST(lowerCI[k,l,m,n])

}
}
}



save(pRec,sRec,propSTRec,realSTRec ,upperCI,lowerCI,upperProp,lowerProp,file="sims3")
load("sims3")

summarySignificant<-summaryRealSTRec<-summaryPropSTRec<-summaryUpperProp<-summaryDiff<-InCI<-overCI<-underCI<-matrix(NA,nrow=length(pobsRobsVect),ncol=2)

for(m in 1:length(pobsRobsVect)){
#for(m in 1:6){
apply(pRec[,,m,1]<0.05,2,mean)-> summarySignificant[m,]
apply(realSTRec[,,m,1],2,mean)-> summaryRealSTRec[m,]
apply(propSTRec[,,m,1],2,mean)-> summaryPropSTRec[m,]
apply(upperProp[,,m,1],2,mean)-> summaryUpperProp[m,]
apply((propSTRec-realSTRec)[,,m,1],2,mean)-> summaryDiff[m,]

apply((realSTRec[,,m,1]> lowerProp[,,m,1])&(realSTRec[,,m,1]<upperProp[,,m,1]),2,sum)/noSims-> InCI[m,]
apply((realSTRec[,,m,1]> upperProp[,,m,1]),2,sum)/noSims-> overCI[m,]
apply((realSTRec[,,m,1]< lowerProp[,,m,1]),2,sum)/noSims-> underCI[m,]

#plot(realSTRec,propSTRec,xlim=c(0,1),ylim=c(0,1))
#cor(as.vector(realSTRec),as.vector(propSTRec))
#[1] 0.833727
#lm(as.vector(propSTRec)~as.vector(realSTRec))

#plot(realSTRec[pRec<0.05],propSTRec[pRec<0.05],xlim=c(0,1),ylim=c(0,1))
#cor(as.vector(realSTRec[pRec<0.05]),as.vector(propSTRec[pRec<0.05]))
#[1] 0.833727
#lm(as.vector(propSTRec[pRec<0.05])~as.vector(realSTRec[pRec<0.05]))
}




