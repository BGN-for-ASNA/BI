

noSims<-1000
noInd<-100
sVect<-seq(0,5,1)
BNoiseVect<-c(0,0.1,0.25,0.5,1)
asocialLP<-rep(1,noInd)
asoc<-cbind(rep(0,noInd))
baseRate<-1/100

sRecord<-aicRecord<-pRecord<-sRecordW<-aicRecordW<-pRecordW<-TsRecord<-TaicRecord<-TpRecord<-TsRecordW<-TaicRecordW<-TpRecordW<-profLikForS <-profLikForSW <-TprofLikForS <-TprofLikForSW <-array(NA,dim=c(noSims,length(sVect),length(BNoiseVect)))

for(l in 1:length(BNoiseVect)){
	BNoise<-BNoiseVect[l]
for(k in 1:length(sVect)){
	s<-sVect[k]
	for(j in 1:noSims){

		#socialNet<-matrix(runif(noInd*noInd,0.7,1)*rbinom(noInd*noInd,1,0.3), nrow=noInd)

		socialNet<-matrix(0, ncol=noInd, nrow=noInd)
		for(i in 0:9){socialNet[i*10+(1:10),i*10+(1:10)]<-runif(100,0.5,1)}
		
		
		BVect<-exp(rnorm(noInd,log(2),sd=BNoise))
		z<-rep(0,noInd)
		orderAcq<-timeAcq <-rep(NA,noInd)
		runningTime<-0

		for(i in 1:noInd){
			rate<-baseRate*(exp(asocialLP)+s*z%*%t(t(socialNet)*BVect))*(1-z)
			times<-rexp(noInd,rate)
			times[is.nan(times)]<-Inf
			orderAcq [i]<-which(times==min(times))[1]
			runningTime<-runningTime+min(times)
			timeAcq[i]<-runningTime
			z[which(times==min(times))[1]]<-1
		}

		oaDataObject<-oaData(assMatrix= socialNet, taskid="1", groupid="1", asoc=asoc, orderAcq= orderAcq)
		oaDataObjectW<-oaData(assMatrix= socialNet, taskid="1", groupid="1", asoc=asoc, orderAcq= orderAcq,weights= BVect)
		taDataObject<-taData(timeAcq,max(timeAcq)+1,oadata= oaDataObject)
		taDataObjectW<-taData(timeAcq,max(timeAcq)+1,oadata= oaDataObjectW)
		model<-addFit(oaDataObject,interval=c(0,9999))
		modelW<-addFit(oaDataObjectW,interval=c(0,9999))
		Tmodel<-tadaFit(taDataObject)
		TmodelW<-tadaFit(taDataObjectW)
		sRecord[j,k,l]<-model@optimisation$minimum
		aicRecord[j,k,l]<-model@aicc
		pRecord[j,k,l]<-model@ LRTsocTransPV
		profLikForS[j,k,l]<-profileLikelihoodAddOADA(s*2,1, model, oaDataObject)-model@optimisation$objective
		sRecordW[j,k,l]<-modelW@optimisation$minimum
		aicRecordW[j,k,l]<-modelW@aicc
		pRecordW[j,k,l]<-modelW@ LRTsocTransPV
		profLikForSW[j,k,l]<-profileLikelihoodAddOADA(s,1, modelW, oaDataObjectW)-modelW@optimisation$objective
		TsRecord[j,k,l]<-Tmodel@optimisation$par[1]
		TaicRecord[j,k,l]<-Tmodel@aicc
		TpRecord[j,k,l]<-Tmodel@ LRTsocTransPV
		TprofLikForS[j,k,l]<-profileLikelihoodTADA(s*2,1, Tmodel, taDataObject)-Tmodel@optimisation$objective
		TsRecordW[j,k,l]<-TmodelW@optimisation$par[1]
		TaicRecordW[j,k,l]<-TmodelW@aicc
		TpRecordW[j,k,l]<-TmodelW@ LRTsocTransPV
		TprofLikForSW[j,k,l]<-profileLikelihoodTADA(s,1, TmodelW, taDataObjectW)-TmodelW@optimisation$objective

				save(sRecord,aicRecord,pRecord,sRecordW,aicRecordW,pRecordW,TsRecord,TaicRecord,TpRecord,TsRecordW, TaicRecordW,TpRecordW,profLikForS ,profLikForSW ,TprofLikForS ,TprofLikForSW, file="NBDANoiseinBAverageRate2.dat")

	}
}
}

