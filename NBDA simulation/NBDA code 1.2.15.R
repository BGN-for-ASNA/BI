#A collection of functions for detecting social transmission from order of acquisition and time of acquisition data
#Please report any bugs or requests for added functionality to Will Hoppitt at w.j.e.hoppitt@leeds.ac.uk

#UPDATE HISTORY

#DATE 5/10/2017
#Now confidence intervals functions work for multiple diffusions for MultiCoxFit

#DATE 6/8/2015
#Changed search interval for optimise in tadaFit, for fitting null model so the limits are more appropriate when not all individuals in the population learn

#DATE 17/3/2014
#plotAssociations function added- this gives a graphical representation of the extent to which the diffusion follows the network


#DATE 27/2/2014
#Profile likelihood functions now work with with MultiCoxFit for a single s parameter only

#DATE: 28/12/2013
#Modified aicTable function so s bounds can be set by the user

#DATE: 26/11/2013
#Bug needs fixing in fitted values for multiCoxFit- currently disabled


#DATE: 09/10/2013
#Attempted some updates to profile likelihood functions for TADA to improve convergence- 
#using scaling of 1/Rate parameter using parscale argument in control for optim

#DATE: 27/9/2013
#Extension of functions for plotting profile likelihoods and calculating profile likeihood confidence intervals
#to gammatadaFit and additive OADA (addFit function). multiCoxFit version to follow.

#DATE: 29/8/2013
#Version 1.2.3
#Inclusion of functions for plotting profile likelihoods and calculating profile likeihood confidence intervals
#Currently for tadaFit models only: gammatadaFit and OADA versions to follow

#DATE: 29/8/2013
#Version 1.2.2
#Bug fixed that prevented aicTable working with a single oaData object.

#DATE: 10/4/2013
#Version 1.2.1

#BUGS FIXED FOR 1.2.1
#The total association with demonstrators (stMetric) was not being updated for the final time period (from last solve to endTime) when creating taData objects
#This should now be fixed
#Previously if there were no individual level variables one had to specify a dummy set (one variable with zeroes for all individuals). Now this is done automatically
#when creating oaData objects, if nothing is specified for the asoc argument.

#NEW FOR VERSION 1.2
#1. Inclusion of functions for fitting TADA with a baseline rate that corresponds to a gamma distribution
#of latencies under asocial conditions. 
#See ESM in Hoppitt, Kandler, Kendal & Laland (2010) Learning & Behaviour 38(3):243-251

#2. Corrected mistake in null AIC when no ILVs are included in the model

#3. Inclusion of a model selection function that fits all possible combinations of the specified variables, with
#both constant and gamma-type baseline functions, and additive and multiplicative models and returns an ordered
#table based on AIC or AICc. This can be used to implement the model selection strategy suggested by Hoppitt &
#Laland (Submitted) American Journal of Primatology

#4. Now allows diffusions with seeded demonstrators, using demons argument to taData and oaData.
#discreteTadaFit also takes individuals with an acquisition time of 0 to be trained demonstrators.

#5. Can now fit a pure social learning model with no asocial learning as used by Franz & Nunn (2009)
#This only makes sense for seeded diffusions

#6. Can now specify "update times", i.e. allow that an individual acquires at one time, but starts transmitting the
#trait later. Details to follow in User Guide.

#7. Can now extract parameter estimates for a null model, using the function nullSummary

#8. Can now fit a version of OADA that assumes a common baseline function within each stratum where a
#stratum is a specified set of diffusions. This is done by treating each stratum as a single diffusion
#but with zero connections between individuals in different actual diffusions
#Times are needed since we need the order within each stratum
#This can be seen as a compromise between the assumptions of the origninal OADA and TADA
#This currently only works when all diffusions have a common end time

#9. Can now add weights indicating each individuals effectiveness as a demonstrator, using the weights
#argument to oaData and taData. For example, one could
#weight by the rate at which an individual performs the trait after having acquired it. I anticipate that
#this could improve statistical power by providing a more realistic model.
#There is no reason why a weighted model cannot be compared to an unweighted model using AIC, in terms of
#explanatory power, but the values of s will not necessarily be comparable.

#Version 1.3 will include MCMC techniques to allow random effects, marginal credible intervals,
#allow the researcher to account for uncertainty in the social network and acquisition times
#and model selection by RJMCMC

#For fitting of Cox models
require(survival)
require(combinat)


setClass("oaData",representation(idname="vector",assMatrix="matrix",asoc="matrix",orderAcq="vector",updateTimes="vector",ties="vector",trueTies="list",demons="vector",weights="vector",groupid="character",taskid="character",groupvect="vector", taskvect="vector",diffvect="vector",coxdata="data.frame",mldata="data.frame"));

setMethod("initialize",
    signature(.Object = "oaData"),
    function (.Object, idname=NULL, assMatrix,asoc,orderAcq,ties=NULL,trueTies=list(NULL),demons=NULL,groupid="1",taskid="1",updateTimes=NULL,groupvect=NA, taskvect=NA,diffvect=NA,weights=rep(1, dim(assMatrix)[1]),...) 
    {
    	

    	#Create default names vector if none is provided, if it is, convert to a factor
		if(is.null(idname)) idname<-(1:dim(assMatrix)[1]);
		
		time1<-vector();
		time2<-vector();
		status<-vector();
		


		id<-vector();
		identity<-vector();
		stMetric<-vector();
		totalMetric<-vector()
		learnMetric<-vector();
		totalAsoc<-vector();
		group<-task<-vector();
	
		#If asoc is null, create dummy data with all zeroes
		if(is.null(asoc)){asoc<-cbind(rep(0,dim(assMatrix)[1]))}
	
		#Make sure the asoc is in matrix form
		asoc<-as.matrix(asoc);
	
		#Calculate the asocial learning variables for the learning individual at each step
		learnAsoc<-matrix(nrow=dim(asoc)[2],ncol=length(orderAcq))
		for(i in 1:length(orderAcq)) learnAsoc[,i]<-asoc[orderAcq[i]]

		#If there are no ties vector of zeroes
		if(is.null(ties)) ties<-rep(0,length(orderAcq));

		#Define functions for calculating total st Metric
		newFunc<-function(x) x*(1-statusTracker)
		newFunc2<-function(x) x*statusTracker2*weights

		
		#Generate a variable for tracking the status of individuals in the group in question
		if(is.null(demons)){
			statusTracker<-rep(0,dim(assMatrix)[1]);
			statusTracker2<-rep(0,dim(assMatrix)[1]);
		}else{
			statusTracker<-statusTracker2<-demons;
		}
		
		#Loop through acquisition events
		for(i in 1:length(orderAcq)){

			if (ties[i]==0)statusTracker2<-statusTracker;
	
			#Loop through each individual
			for(j in 1:dim(assMatrix)[1]){

				#Only naive individuals considered
				if(statusTracker[j]==0){

					#Record variables for Cox model
					time1<-c(time1,i-1);
					time2<-c(time2,i);
					identity<-c(identity,j);
					stMetric<-c(stMetric,sum(assMatrix[j,]*statusTracker2*weights));
					id<-c(id,idname[j]);
					if(!is.na(groupvect[1]))group<-c(group,groupvect[j])
					if(!is.na(taskvect[1]))task<-c(task,taskvect[j])
					
					#Record status as one if individual acquires trait, zero otherwise for Cox model
					if(j==orderAcq[i]){
						status<-c(status,1);
					
						#Record the social transmission metric and asocial learning variables for the learning individual (ML model)
						learnMetric<-c(learnMetric,sum(assMatrix[j,]*statusTracker2*weights));

					}else{
						status<-c(status,0);

					}
				}
			}

			#Calculate total st metric over all individuals in the group for each acquisition event (ML model)
			newMatrix<-apply(assMatrix,2,newFunc)
			if(is.na(diffvect[1])){
				totalMetric<-c(totalMetric,sum(apply(newMatrix,1,newFunc2)));
			}else{
				
				 totalMetric<-rbind(totalMetric,as.vector(tapply(apply(apply(newMatrix,1,newFunc2),2,sum),diffvect,sum)))
			}
		
			#Set statusTracker to one if individual acquires trait
			statusTracker[orderAcq[i]]<-1;
		}
	
#Record individual variables for each line of data in the Cox model
indVar<-matrix(asoc[identity,],ncol=ncol(asoc),dimnames=dimnames(asoc));
if(is.na(groupvect[1])) group<-rep(groupid,length(time1));
if(is.na(taskvect[1])) task<-rep(taskid,length(time1));
coxdata<-data.frame(id=as.factor(id),time1,time2,status,identity,stMetric,indVar,group,task);
#And for the ML models
if(is.na(groupvect[1])) {group<-rep(groupid,length(orderAcq))}else{group<-rep("NA",length(orderAcq))};
if(is.na(taskvect[1])) {task<-rep(taskid,length(orderAcq))}else{task<-rep("NA",length(orderAcq))};
mldata<-data.frame(group,task,orderAcq,learnMetric,totalMetric,asoc[orderAcq,]);

if(is.null(demons)) demons<-NA;
if(is.na(groupvect[1])) {groupvect<-groupid};
if(is.na(taskvect[1])) {taskvect<-taskid};

callNextMethod(.Object,idname=idname,assMatrix=assMatrix, asoc=asoc, orderAcq=orderAcq, updateTimes=NA, ties=ties, trueTies=trueTies, demons=demons, weights=weights, groupid=groupid,taskid=taskid, groupvect=groupvect, taskvect=taskvect, diffvect=diffvect,coxdata=coxdata, mldata=mldata)

}
)

oaData<-function(idname=NULL, assMatrix,asoc=NULL,orderAcq,ties=NULL,trueTies=list(NULL),groupid="1",taskid="1", groupvect=NA, taskvect=NA, diffvect=NA,updateTimes=NULL,demons=NULL,weights=rep(1, dim(assMatrix)[1])){
	
	
	new("oaData",idname=idname,assMatrix=assMatrix,asoc=asoc,orderAcq=orderAcq,ties=ties,trueTies=trueTies,groupid=groupid,taskid=taskid,demons=demons,groupvect=groupvect,taskvect=taskvect,diffvect=diffvect,weights=weights);
	
}


#Gets likelihood for a multiplicative model using a transformed cox model with specified value of s
multiCoxLikelihood<-function(s,oadata,formula=NULL,bounded=FALSE, stratify="both"){
	#Fit transformed cox model

	#Calulate the appropriate value of s for each line of data
	if(is.null(oadata@coxdata$sParamIndex)){
	
		sVect<-rep(s,dim(oadata@coxdata)[1])
	
	}else{
		
		stemp<-1*(oadata@coxdata$sParamIndex>0)
		stemp[which(oadata@coxdata$sParamIndex>0)]<-s[oadata@coxdata$sParamIndex[oadata@coxdata$sParamIndex>0]]
		sVect<-stemp;
		
	}
	
	if(bounded==T){
			if(stratify=="both")model<-coxph(Surv(time1,time2,status)~strata(group,task)+offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);
			if(stratify=="group")model<-coxph(Surv(time1,time2,status)~strata(group)+offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);
			if(stratify=="task")model<-coxph(Surv(time1,time2,status)~strata(task)+offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);
			if(stratify=="none")model<-coxph(Surv(time1,time2,status)~offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);

		}else	
		{
			if(stratify=="both")model<-coxph(Surv(time1,time2,status)~strata(group,task)+offset(log(sVect*stMetric+1)),data=oadata@coxdata);
			if(stratify=="group")model<-coxph(Surv(time1,time2,status)~strata(group)+offset(log(sVect*stMetric+1)),data=oadata@coxdata);
			if(stratify=="task")model<-coxph(Surv(time1,time2,status)~strata(task)+offset(log(sVect*stMetric+1)),data=oadata@coxdata);
			if(stratify=="none")model<-coxph(Surv(time1,time2,status)~offset(log(sVect*stMetric+1)),data=oadata@coxdata);

		}
	if(is.null(formula)){
		model1<-model
		return(-model1$loglik[1])
	}else{
		model1<-update(model,formula);
	}
	-model1$loglik[2];

}



#Combine the coxdata from two or more oaData objects, into an oaData object with NA in all other slots
combineOaCoxData<-function(oaNames,sParam=NULL){
	
	#Set the default sParam (same s for all diffusions) if it is not defined
	if(is.null(sParam)) sParam<-rep(1,length(oaNames));

	newOaObject<-eval(as.name(oaNames[1]));
	if(is.na(newOaObject@diffvect[1])){
		sParamIndex<-rep(sParam[1],dim(newOaObject@coxdata)[1])
		diffcount<-1
	}else{
		sParamIndex<-sParam[newOaObject@diffvect[newOaObject@coxdata$id]]
		diffcount<-max(newOaObject@diffvect)
	}
	
	if(length(oaNames)>1){
	for(i in 2:length(oaNames)){

			newOaObject@coxdata<-rbind(newOaObject@coxdata,eval(as.name(oaNames[i]))@coxdata);

			if(is.na(newOaObject@diffvect[1])){
				sParamIndex<-c(sParamIndex,rep(sParam[i],dim(eval(as.name(oaNames[i]))@coxdata)[1]));
				diffcount<-diffcount+1
			}else{
				sParamIndex<-c(sParamIndex,sParam[diffcount+newOaObject@diffvect[newOaObject@coxdata$id]]);
				diffcount<-diffcount+ max(newOaObject@diffvect)
			}

	}
	}
	newOaObject@idname<-NA;
	newOaObject@assMatrix<-matrix(NA);
	newOaObject@asoc<-matrix(NA);
	newOaObject@orderAcq<-NA;
	newOaObject@groupid<-"NA";
	newOaObject@taskid<-"NA";
	newOaObject@mldata<-data.frame(NA);
	
	newOaObject@coxdata<-cbind(newOaObject@coxdata,sParamIndex)

	return(newOaObject);
}


#Define class of object for the fitted multiplicative model
setClass("multiCoxFit",representation(optimisation="list",sParam="numeric",bounded="logical",formula="formula",coxcall="call",coxdf="numeric",coef="matrix",conf.int="matrix",loglik="numeric",aic="numeric", aicc="numeric",nulllik="numeric",aicNull="numeric",aiccNull="numeric",LRTsocTransTS="numeric",LRTsocTransPV="numeric",fitted="numeric",time="numeric",status="numeric",task="factor",group="factor"));

#Method for initializing object- including model fitting
setMethod("initialize",
    signature(.Object = "multiCoxFit"),
    function (.Object, oadata,sParam=NULL,formula,startValue=startValue,bounded=bounded,interval=interval,method=method,stratify=stratify,...) 
    {

		#If the oadata if a character vector containing multiple oaData objects, combine into a single object first
		if(is.character(oadata)){		
			#Set the default sParam (same s for all diffusions) if it is not defined
			if(is.null(sParam)) sParam<-rep(1,length(oadata));
			oadata<-combineOaCoxData(oadata,sParam=sParam);
		}else{
			
			sParam<-1;	
		}

		sampleSize<-sum(oadata@coxdata$status);
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		#Set staring values if not specified by the user
		if(is.null(startValue)) 		startValue<-rep(0, noSParam);

		
		if(length(startValue)>1) method="nlminb";

		#Optimise for s
		#If bounded==T the social learning parameter is constrained between 0 and 1, otherwise between 0 and Inf
		if(method=="nlminb"){	
			if(bounded==T){
				fit1<-nlminb(start=startValue,objective=multiCoxLikelihood,lower=0,upper=1,oadata=oadata,formula=formula,bounded=T,stratify=stratify);
			}else{			
				fit1<-nlminb(start=startValue,objective=multiCoxLikelihood,lower=0,upper=Inf,oadata=oadata,formula=formula,bounded=F,stratify=stratify);
				}
			stFitted<-fit1$par;
		}
		if(method=="optimise"){	
			if(bounded==T){
				fit1<-optimise(f=multiCoxLikelihood,interval=c(0,1),oadata=oadata,formula=formula,bounded=T,stratify=stratify);
			}else{			
				fit1<-optimise(f=multiCoxLikelihood,interval=interval,oadata=oadata,formula=formula,bounded=F,stratify=stratify);
				}
			stFitted<-fit1$minimum;
		}
		

		#Calulate the appropriate value of s for each line of data
		if(is.null(oadata@coxdata$sParamIndex)){
	
			sVect<-rep(stFitted,dim(oadata@coxdata)[1])
	
		}else{
		
			stemp<-1*(oadata@coxdata$sParamIndex>0)
			stemp[which(oadata@coxdata$sParamIndex>0)]<-stFitted[oadata@coxdata$sParamIndex[oadata@coxdata$sParamIndex>0]]
			sVect<-stemp;
		
		}
			
			
			
		#Feed back into the cox model and record appropriate output



		if(bounded==F) {
			if(stratify=="both") model<-coxph(Surv(time1,time2,status)~strata(group,task)+offset(log(sVect*stMetric+1)),data=oadata@coxdata);
			if(stratify=="group") model<-coxph(Surv(time1,time2,status)~strata(group)+offset(log(sVect*stMetric+1)),data=oadata@coxdata);
			if(stratify=="task") model<-coxph(Surv(time1,time2,status)~strata(task)+offset(log(sVect*stMetric+1)),data=oadata@coxdata);
			if(stratify=="none") model<-coxph(Surv(time1,time2,status)~offset(log(sVect*stMetric+1)),data=oadata@coxdata);
		}
		if(bounded==T){
			if(stratify=="both") model<-coxph(Surv(time1,time2,status)~strata(group,task)+offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);
			if(stratify=="group") model<-coxph(Surv(time1,time2,status)~strata(group)+offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);
			if(stratify=="task") model<-coxph(Surv(time1,time2,status)~strata(task)+offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);
			if(stratify=="none") model<-coxph(Surv(time1,time2,status)~offset(log(sVect*stMetric+(1-sVect))),data=oadata@coxdata);		}
		
		if(is.null(formula)){
			model1<-model
		}else{
			model1<-update(model,formula);
		}

		if(is.null(formula)){
			
			coef<-matrix(nrow=1,ncol=1);
			conf.int<-matrix(nrow=1,ncol=1);
		}else{
			coef<-summary(model1)$coef;
			conf.int<-summary(model1)$conf.int
		}

		#For a frailty model the summary does not output an object, the stats are printed to the screen, therefore we cannot store them in the object
		if(is.null(coef)) {
			coef<-matrix(nrow=1,ncol=1);
		}
		if(is.null(conf.int)) conf.int<-matrix(nrow=1,ncol=1);

		loglik<-model1$loglik;
		coxcall<-model1$call;
		if(stratify=="both")nullmodel<-coxph(Surv(time1,time2,status)~strata(group,task),data=oadata@coxdata);
		if(stratify=="group")nullmodel<-coxph(Surv(time1,time2,status)~strata(group),data=oadata@coxdata);
		if(stratify=="task")nullmodel<-coxph(Surv(time1,time2,status)~strata(task),data=oadata@coxdata);
		if(stratify=="none")nullmodel<-coxph(Surv(time1,time2,status)~1,data=oadata@coxdata);
		
		if(is.null(formula)){
			nullmodel<-nullmodel
			nulllik<-nullmodel$loglik[1];		
		}else{
			nullmodel<-update(nullmodel,formula);
			nulllik<-nullmodel$loglik[2];
		}

		#Store appropriate formula if none is given
		if(is.null(formula)) {
			formula<-~.
			loglik<-rep(loglik,2)
		}

		#Record the total df for the cox model
		coxdf<-round(sum(model1$df));
		
		LRTsocTransTS<-2*(-nulllik+loglik[2])
		LRTsocTransPV<-1-pchisq(LRTsocTransTS,df= noSParam);


		#Record aic of fitted and null model. Also small sample AIC
        if(is.na(coef[1])){
        	aic<-2*(length(stFitted)+coxdf)-2*loglik[2];
        	aicc<-2*(length(stFitted)+coxdf)*(sampleSize/(sampleSize-length(stFitted)-coxdf-1))-2*loglik[2];
	        aicNull<-2*coxdf-2*nulllik;
	        aiccNull<-2*(coxdf)*(sampleSize/(sampleSize-coxdf-1))-2*nulllik;
	    }else{
	        aic<-2*(dim(coef)[1]+length(stFitted))-2*loglik[2];
        	aicc<-2*(dim(coef)[1]+length(stFitted))*(sampleSize/(sampleSize-(dim(coef)[1]+length(stFitted))-1))-2*loglik[2];
	        aicNull<-2*(dim(coef)[1])-2*nulllik;
	        aiccNull<-2*(dim(coef)[1])*(sampleSize/(sampleSize-(dim(coef)[1])-1))-2*nulllik;
     	}

		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)){}else{
			if(aicc<aic) aicc<-NaN;
		}
		if(is.nan(aicNull)|is.nan(aiccNull)){}else{
			if(aiccNull<aicNull) aiccNull<-NaN;
		}
		        
        #Record fitted values
################
#TO BE CORRECTED

#        fitted<-exp(predict(model1));
 		fitted<--999
 
        time<-oadata@coxdata$time2;
        status<-oadata@coxdata$status;
        task<-oadata@coxdata$task;
        group<-oadata@coxdata$group;       

 
 
        callNextMethod(.Object,bounded=bounded,optimisation=fit1,sParam=sParam,formula=formula,coxcall=coxcall,coxdf=coxdf,coef=coef,conf.int=conf.int,loglik=loglik,aic=aic,aicc=aicc,nulllik=nulllik,aicNull=aicNull,aiccNull=aiccNull,LRTsocTransTS=LRTsocTransTS,LRTsocTransPV=LRTsocTransPV,fitted=fitted,time=time,status=status,task=task,group=group,...)
        

       
    }
)

#Function for implementing the initialization
multiCoxFit<-function(oadata,sParam=NULL,formula=NULL,startValue=NULL,bounded=FALSE,interval=c(0,999),method="optimise",stratify="both"){
		
	new("multiCoxFit",oadata=oadata, sParam=sParam, formula=formula,startValue=startValue,bounded=bounded,interval=interval,method=method,stratify=stratify)	
}


setMethod("summary",
    signature(object = "multiCoxFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])));
		cat("Summary of Multiplicative Social Transmission Model\nOrder of acquisition data\n");


		if(is.na(object@coef[1])){

			if(is.null(object@optimisation$par)){

			if(object@bounded==T) {
				cat("Bounded parameterisation.");
				sumtable<-data.frame(Estimate=object@optimisation$minimum,Unbounded=object@optimisation$minimum/(1-object@optimisation$minimum), row.names="Social Transmission");
			}
			if(object@bounded==F) {
				cat("Unbounded parameterisation.");
				sumtable<-data.frame(Estimate=object@optimisation$minimum,Bounded=object@optimisation$minimum/(1+object@optimisation$minimum), row.names="Social Transmission");
			}
				
		}else{

			if(object@bounded==T) {
				cat("Bounded parameterisation.");
				sumtable<-data.frame(Estimate=c(object@optimisation$par),Unbounded=c(object@optimisation$par/(1-object@optimisation$par)), row.names=c(paste("Social transmission",1:noSParam)));			}
			if(object@bounded==F) {
				cat("Unbounded parameterisation.");
				sumtable<-data.frame(Estimate=c(object@optimisation$par),Bounded=c(object@optimisation$par/(1+object@optimisation$par)),row.names=c(paste("Social transmission",1:noSParam)));
			}

		}
			
		}else{

			if(is.null(object@optimisation$par)){

			if(object@bounded==T) {
				cat("Bounded parameterisation.");
				sumtable<-data.frame(Estimate=c(object@optimisation$minimum,object@coef[,1]),Unbounded=c(object@optimisation$minimum/(1-object@optimisation$minimum),rep(NA,length(object@coef[,1]))),se=c(rep(NA,noSParam),object@coef[,3]),z=c(rep(NA,noSParam),object@coef[,4]),p=c(rep(NA,noSParam),object@coef[,5]), row.names=c(paste("Social transmission",1:noSParam),row.names(object@coef)));			}
			if(object@bounded==F) {
				cat("Unbounded parameterisation.");
				sumtable<-data.frame(Estimate=c(object@optimisation$minimum,object@coef[,1]),Bounded=c(object@optimisation$minimum/(1+object@optimisation$minimum),rep(NA,length(object@coef[,1]))),se=c(rep(NA,noSParam),object@coef[,3]),z=c(rep(NA,noSParam),object@coef[,4]),p=c(rep(NA,noSParam),object@coef[,5]), row.names=c(paste("Social transmission",1:noSParam),row.names(object@coef)));
				
			}
			}else{
	
			if(object@bounded==T) {
				cat("Bounded parameterisation.");
				sumtable<-data.frame(Estimate=c(object@optimisation$par,object@coef[,1]),Unbounded=c(object@optimisation$par/(1-object@optimisation$par),rep(NA,length(object@coef[,1]))),se=c(rep(NA,noSParam),object@coef[,3]),z=c(rep(NA,noSParam),object@coef[,4]),p=c(rep(NA,noSParam),object@coef[,5]), row.names=c(paste("Social transmission",1:noSParam),row.names(object@coef)));			}
			if(object@bounded==F) {
				cat("Unbounded parameterisation.");
				sumtable<-data.frame(Estimate=c(object@optimisation$par,object@coef[,1]),Bounded=c(object@optimisation$par/(1+object@optimisation$par),rep(NA,length(object@coef[,1]))),se=c(rep(NA,noSParam),object@coef[,3]),z=c(rep(NA,noSParam),object@coef[,4]),p=c(rep(NA,noSParam),object@coef[,5]), row.names=c(paste("Social transmission",1:noSParam),row.names(object@coef)));
			}
		}
	}	
		cat("\n\nCoefficients:\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
       
    }
)


setMethod("anova",
    signature(object = "multiCoxFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.na(object@coef[1])){

			table<-data.frame(Df=c(noSParam+object@coxdf,0+object@coxdf),LogLik=c(-object@loglik[2],-object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));			

		}else{

			table<-data.frame(Df=c((dim(object@coef)[1]+ noSParam),(dim(object@coef)[1])),LogLik=c(-object@loglik[2],-object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));

		}

		atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine multiplicatively\n", class = c("anova", "data.frame"));
		
		
		return(atable);

    }
)


nbdaProfile<-function(data=NULL,model=NULL,range=NULL,range2=NULL, additive=TRUE, bounded=F,resolution=1000,confInt=c(0.95,0.99),ylim=NULL,param=1,otherParam=NULL,progress=F, stepLength=NULL){

	if(class(model)=="multiCoxFit"){
		additive<-FALSE
		formula<-model@formula;
		#If the oadata if a character vector containing multiple oaData objects, combine into a single object first
		if(is.character(data)){		
			if(additive==FALSE) data<-combineOaCoxData(data,sParam=model@sParam)
		}
	}
	
	if(class(model)=="tadaFit"|class(model)=="gammaTadaFit"){
		additive<-model@additive
		bounded<-F
		#If the oadata if a character vector containing multiple oaData objects, combine into a single object first
		if(is.character(data)){		
			data<-combineTadaData(data,sParam=model@sParam)
		}
	}	

	if(class(model)=="discreteTadaFit"){
		additive<-model@additive;
		bounded=F;
		
	}

	if(is.null(model)){
		fitted.value<-NULL;
	}else{
		if(is.null(model@optimisation$minimum)){ 
			fitted.value<-model@optimisation$par;
		}else{
			fitted.value<-model@optimisation$minimum;
		}

		#Convert parameters in model to bounded/unbounded as specified
		if(bounded==T){
			if(model@bounded==F) {
				if(class(model)=="multiCoxFit"){ 
					fitted.value<-fitted.value/(1+fitted.value);
				}else{
					if(is.null(model@asocialVar)){
						fitted.value<-fitted.value/(1+fitted.value);
					}else{
						fitted.value[1:(length(fitted.value)-length(model@asocialVar))]<-fitted.value[1:(length(fitted.value)-length(model@asocialVar))]/(1+fitted.value[1:(length(fitted.value)-length(model@asocialVar))]);	
					}	
				}
			}
		}
		if(bounded==F){
			if(model@bounded==T) {
				if(class(model)=="multiCoxFit"){ 
					fitted.value<-fitted.value/(1-fitted.value)
				}else{
					if(is.null(model@asocialVar)){
						fitted.value<-fitted.value/(1-fitted.value)
					}else{
						fitted.value[1:(length(fitted.value)-length(model@asocialVar))]<-fitted.value[1:(length(fitted.value)-length(model@asocialVar))]/(1+fitted.value[1:(length(fitted.value)-length(model@asocialVar))]);	
					}	
				}
			}
		}
	}


	#If values are specified for the other parameters, replace the fitted values with these
	if(is.null(otherParam)){}else{
		
		fitted.value[-param]<-otherParam
		
	}	
	
	#Set ranges of plot to 0,1 if bounded, user specified if unbounded with  default of 0,100, unless defined by user
	
	if(is.null(range)){
		if(bounded==T){
			range<-c(0,1);
			if(fitted.value[param[1]]>1) range<-c(0,2*fitted.value[param[1]]);
		}else{
			range<-c(0,2*fitted.value[param[1]]);
		}
	}

	if(is.null(range2)){
		if(bounded==T){
			range2<-c(0,1);
			if (is.na(fitted.value[param[2]])==FALSE){
				if(fitted.value[param[2]]>1) range2<-c(0,2*fitted.value[param[2]]);
			}
		}else{
			range2<-c(0,100);		
			if (is.na(fitted.value[param[2]])==FALSE){
				if(fitted.value[param[2]]>0) range2<-c(0,2*fitted.value[param[2]]);
				if(fitted.value[param[2]]<0) range2<-c(2*fitted.value[param[2]],0);
			}
		}
	}

	#If the user specifies only one parameter to be plotted, plot in 1D with other parameters set at fitted values, unless the user specifes values for the other parameters

	if(length(param)==1){

		#Create vector of x values
		xValues<-seq(range[1],range[2],length.out=resolution);
		#Cut out the last value (likelihood is 0) if bounded
		if(bounded==T) xValues<-xValues[-resolution];
		#Initialise y values
		yValues<-vector(mode="numeric",length=length(xValues));

		#If there is more than one parameter set the others to their fitted value
		if(is.null(model@optimisation$minimum)){
	
			newXValues<-t(matrix(rep(fitted.value,length(xValues)),nrow=length(fitted.value)));
			newXValues[,param]<-xValues;
		
		}else{

			newXValues<-as.matrix(xValues);
		
		
		}

		if(class(model)=="multiCoxFit") {
			asocialVar<-NULL;
		}else{
			asocialVar<-model@asocialVar;
			if(asocialVar=="NaN") asocialVar<-NULL;
		}

		#Calculate likelihood for each x value	
		for(i in 1:dim(newXValues)[1]){
			if(class(model)=="multiCoxFit") {
				yValues[i]<-multiCoxLikelihood(newXValues[i,],oadata=data,formula=model@formula,bounded=bounded);
				labels<-paste("Social transmission parameter", param);
			}else{
					
				if(class(model)=="tadaFit"){
					yValues[i]<-tadaLikelihood(l=newXValues[i,],tadata=data,asocialVar=asocialVar,bounded=bounded,additive=additive, task=model@task, group=model@group,sParam=model@sParam)
					labels<-model@varNames[param];

				}else{
					
					if(class(model)=="discreteTadaFit"){
						
						yValues[i]<-discreteTadaLikelihood(l=newXValues[i,],tadata=data,asocialVar=asocialVar,bounded=bounded,additive=additive, task=model@task, group=model@group, stepLength=stepLength,sParam=model@sParam);
						labels<-model@varNames[param];
												
						
						
					}else{
					
						yValues[i]<-addLikelihood(newXValues[i,],data=data,asocialVar=asocialVar,bounded=bounded);
						labels<-model@varNames[param];
					}
				}
			}
		}

	
		#Plot
		plot(xValues,yValues,type="l",xlab=labels[1],ylab="log Likelihood",ylim=ylim);

	
		if(is.null(fitted.value)){}else{

			if(class(model)=="multiCoxFit") {
				confIntValue<-(multiCoxLikelihood(fitted.value,oadata=data,formula=formula,bounded=bounded)+qchisq(confInt,1)/2);			}else{
					
				if(class(model)=="tadaFit"){
					confIntValue<-(tadaLikelihood(fitted.value,tadata=data,asocialVar=asocialVar,bounded=bounded, task=model@task, group=model@group, sParam=model@sParam)+qchisq(confInt,1)/2);
				}else{

					if(class(model)=="discreteTadaFit"){
					
						confIntValue<-(discreteTadaLikelihood(fitted.value,tadata=data,asocialVar=asocialVar,bounded=bounded, task=model@task, group=model@group, sParam=model@sParam, stepLength=stepLength))+(qchisq(confInt,1)/2);
						
					}else{

						confIntValue<-(addLikelihood(fitted.value,data=data,asocialVar=asocialVar,bounded=bounded, sParam=model@sParam)+qchisq(confInt,1)/2);
					}
				}
			}

			abline(h=confIntValue,lty=1+1:length(confInt));

			#Find confidence intervals. Replace with NA if on boundary (unless boundary is zero for ST paramter)
			STlCI<-vector();
			STuCI<-vector();
			for (i in 1:length(confInt)){
				STlCI<-c(STlCI,min(xValues[yValues<(confIntValue[i])]));
				STuCI<-c(STuCI,max(xValues[yValues<(confIntValue[i])]));
				if(STlCI[i]==0) {}else{
					if(STlCI[i]==min(xValues)) STlCI[i]<-NA;
				}
				if(STuCI[i]==max(xValues)) STuCI[i]<-NA;

			}
		}
		ciTable<-cbind(confInt,STlCI,STuCI);
		if(class(model)=="multiCoxFit") {
			dimnames(ciTable)[2]<-list(c("Confidence level",paste("Social Transmission",paste(rep(param,each=2),c("lower","upper")))));
		}else{
			dimnames(ciTable)[2]<-list(c("Confidence level",paste(rep(model@varNames[param],each=2),c("lower","upper"))));
		}
		return(ciTable)


	}else{
	#If the user specifies two parameters to be plotted, plot contours in 2D with other parameters set at fitted values, unless the user specifes values for the other parameters		
		
		#Create vectors of x and y values
		xValues<-seq(range[1],range[2],length.out=resolution);
		#Cut out the last value (likelihood is 0) if bounded
		if(bounded==T) xValues<-xValues[-resolution];
		yValues<-seq(range2[1],range2[2],length.out=resolution);
		#Cut out the last value (likelihood is 0) if bounded
		if(bounded==T) yValues<-yValues[-resolution];


		x<-matrix(nrow=length(xValues),ncol=length(yValues));
		y<-matrix(nrow=length(xValues),ncol=length(yValues));
		z<-matrix(nrow=length(xValues),ncol=length(yValues));


		asocialVar<-model@asocialVar;
		if(asocialVar=="NaN") asocialVar<-NULL;	

		#Generate grid of x y and z values
		for(i in 1:length(xValues)){
	
			for(j in 1:length(yValues)){

				if(progress==T){
					cat(i," ",j,"\n");
					flush.console();
				}
			
				x[i,j]<-xValues[i];
				y[i,j]<-yValues[j];
			
				input<-fitted.value;
				input[param[1]]<-xValues[i];
				input[param[2]]<-yValues[j];

				if(class(model)=="multiCoxFit") {
					z[i,j]<-multiCoxLikelihood(s=input,oadata=data,formula=model@formula,bounded=bounded);					labels<-paste("Social transmission parameter", param);
				}else{
					
					if(class(model)=="tadaFit"){
						z[i,j]<-tadaLikelihood(l=input,tadata=data,asocialVar=asocialVar,bounded=bounded,additive=additive,task=model@task, group=model@group,sParam=model@sParam);
						labels<-model@varNames[param];					}else{

						if(class(model)=="discreteTadaFit"){
					
							z[i,j]<-discreteTadaLikelihood(l=input,tadata=data,asocialVar=asocialVar,bounded=bounded,additive=additive,task=model@task, group=model@group,sParam=model@sParam,stepLength=stepLength);							labels<-model@varNames[param];					
					
						}else{
					
							if(class(model)=="gammaTadaFit"){
								z[i,j]<-gammaTadaLikelihood(l=input,tadata=data,asocialVar=asocialVar,bounded=bounded,additive=additive,task=model@task, group=model@group,sParam=model@sParam);
								labels<-model@varNames[param];
							}else{

								z[i,j]<-addLikelihood(l=input,data=data,asocialVar=asocialVar,bounded=bounded,sParam=model@sParam);
								labels<-model@varNames[param];
							}
						}
					}

				}
			}
		}

		if(is.null(model)){
		
			filled.contour(xValues,yValues,z,xlab=labels[1],ylab=labels[2]);

		}else{
			confIntValue<-model@optimisation$objective+c(0,(qchisq(confInt,1)/2));
			filled.contour(xValues,yValues,z,levels=confIntValue,xlab=labels[1],ylab=labels[2]);

			#Find confidence intervals. Replace with NA if on boundary (unless boundary is zero for ST paramter)
			STlCI<-vector();
			STuCI<-vector();
			ALlCI<-vector();
			ALuCI<-vector();
			for (i in 1:length(confInt)){
				STlCI<-c(STlCI,min(x[z<(confIntValue[i+1])]));
				STuCI<-c(STuCI,max(x[z<(confIntValue[i+1])]));
				if(STlCI[i]==0) {}else{
					if(STlCI[i]==min(x)) STlCI[i]<-NA;
				}
				if(STuCI[i]==max(x)) STuCI[i]<-NA;
				ALlCI<-c(ALlCI,min(y[z<(confIntValue[i+1])]));
				ALuCI<-c(ALuCI,max(y[z<(confIntValue[i+1])]));
				if(ALlCI[i]==min(y)) ALlCI[i]<-NA;
				if(ALuCI[i]==max(y)) ALuCI[i]<-NA;
			}
			ciTable<-cbind(confInt,STlCI,STuCI,ALlCI,ALuCI);
			if(class(model)=="multiCoxFit") {
				dimnames(ciTable)[2]<-list(c("Confidence level",paste("Social Transmission",paste(rep(param,each=2),c("lower","upper")))));
			}else{
				dimnames(ciTable)[2]<-list(c("Confidence level",paste(rep(model@varNames[param],each=2),c("lower","upper"))));
			}
			return(ciTable)
			
	
		}
	}
}


#Define class of object for the fitted nbda model
setClass("tadaFit",representation(optimisation="list",optimisationNull="list",additive="logical",sParam="numeric",bounded="logical",loglik="numeric",aic="numeric", aicc="numeric",nulllik="numeric",aicNull="numeric",aiccNull="numeric",LRTsocTransTS="numeric",LRTsocTransPV="numeric",asocialVar="numeric",varNames="character",task="logical",group="logical")); 


#Method for initializing object- including model fitting
setMethod("initialize",
    signature(.Object = "tadaFit"),
    function (.Object, tadata,additive,sParam,startValue,asocialVar,bounded,task,group,...) 
    {

		#If the tadata if a character vector containing multiple oaData objects, combine into a single object first
		#Also set the default sParam if nothing is specified
		if(is.character(tadata)){		
			if(is.null(sParam)) sParam<-rep(1,length(tadata));
				tadata<-combineTadaData(tadata,sParam=sParam);
			

		}else{
			sParam<-1;				
		}

		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		sampleSize<-sum(tadata@nbdaData$status);

		extraParam<-0;

		if(task) extraParam<-extraParam+length(levels(tadata@nbdaData$task))-1;
		if(group) extraParam<-extraParam+length(levels(tadata@nbdaData$group))-1;
		
		#Set staring values if not specified by the user
		if(is.null(startValue)){
			 		startValue<-c(rep(0,noSParam),1,rep(0,length(asocialVar)+extraParam));
					startInd<-1;
		}else{
		
			startInd<-0;
			
		}

if((length(asocialVar)+extraParam)==0){
	if(length(tadata@timeAcq)/dim(tadata@oadata@assMatrix)[1]<1){
		upperSearchLimit<--100*tadata@endTime/log(1-length(tadata@timeAcq)/dim(tadata@oadata@assMatrix)[1])
	}else{
		upperSearchLimit<-max(tadata@nbdaData$time2)+1000
			
	}
	null<-optimise(nullTadaLikelihood,interval=c(0, upperSearchLimit),asocialVar=asocialVar,tadata=tadata, additive=additive,sParam=sParam);
	if(startInd==1) startValue[2]<-null$minimum;
}else{
	null<-nlminb(start=startValue[-(1: noSParam)],nullTadaLikelihood,lower=c(0,rep(-Inf,length(asocialVar)+extraParam)),asocialVar=asocialVar,tadata=tadata,task=task,group=group,additive=additive,sParam=sParam);
	if(startInd==1) startValue[-(1: noSParam)]<-null$par;
}


		
		fit1<-nlminb(start=startValue,tadaLikelihood,lower=c(rep(0,noSParam+1),rep(-Inf,length(asocialVar)+extraParam)),sParam=sParam,asocialVar=asocialVar,tadata=tadata,bounded=bounded,task=task,group=group,additive=additive);

		#Perform LRT for social transmission
		loglik<-fit1$objective;
		nulllik<-null$objective;
		LRTsocTransTS<-2*(nulllik-loglik)
		LRTsocTransPV<-1-pchisq(LRTsocTransTS,df=noSParam);

		if(is.null(asocialVar)) asocialVar<-NaN;


		#Calculate aic and for model without social transmission
		k<-length(fit1$par);
		aic<-(2*k)+(2*loglik);
		aicc<-aic+(2*k*(k+1)/(sampleSize-k-1));

		if(is.null(null$par)){k<-1}else{k<-length(null$par)};
		aicNull<-(2*k)+(2*nulllik);
		aiccNull<-aicNull+(2*k*(k+1)/(sampleSize-k-1));

		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)){}else{
			if(aicc<aic) aicc<-NaN;
		}
		if(is.nan(aicNull)|is.nan(aiccNull)){}else{
			if(aiccNull<aicNull) aiccNull<-NaN;
		}
		
		#Extract names of variables
		if(is.nan(asocialVar[1])){
	
			varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling");			
			
		}else{
			varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling",unlist(dimnames(tadata@nbdaData)[2])[asocialVar+6]);
		}
		if(task) varNames<-c(varNames,paste("Task",levels(tadata@nbdaData$task)[-1]));
		if(group) varNames<-c(varNames,paste("Group",levels(tadata@nbdaData$group)[-1]));





        callNextMethod(.Object,optimisation=fit1,optimisationNull=null,additive=additive,sParam=sParam,bounded=bounded,loglik=loglik, aic=aic,aicc=aicc,nulllik=nulllik,aicNull=aicNull,aiccNull=aiccNull,LRTsocTransTS=LRTsocTransTS,LRTsocTransPV=LRTsocTransPV,asocialVar=asocialVar,varNames=varNames,task=task,group=group,...) 

       

       
    }
)


#Function for implementing the initialization
tadaFit<-function(tadata,sParam=NULL,asocialVar=NULL,startValue=NULL,bounded=FALSE,task=FALSE,group=FALSE,additive=TRUE){

	
	if(bounded==T) {
	
	cat("Bounded parameterisation not yet available for time of acquisition models");
	return("Bounded parameterisation not yet available for time of acquisition models");	
		
	}	
	
	new("tadaFit",tadata=tadata,sParam=sParam,asocialVar=asocialVar,startValue=startValue,bounded=bounded,task=task,group=group,additive=additive)	
}



#Gets likelihood for a multiplicative nbda model 
tadaLikelihood<-function(l,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE){




	if(is.null(sParam)){
		noSParam<-1
		
	}else{
		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
	}

#Found that paramterising by 1/rate works better for optimisation
l[noSParam+1]<-1/l[noSParam+1];

	if(is.null(asocialVar)){
		}else{
#		if(max(asocialVar)>dim(tadata@oadata@asoc)[2]){
		
#			return("Invalid asocial learning variables selected")
		
		
#		}
	}

	#Cut down to the asocial variables specified by asocialVar
	asoc<-as.matrix(tadata@nbdaData[,asocialVar+6]);

	if(task){
		
		taskNames<-paste("task",levels(tadata@nbdaData$task),sep="");
		taskMatrix<-matrix(nrow=dim(tadata@nbdaData)[1],ncol=length(taskNames)-1,dimnames=list(NULL,taskNames[-1]))
		for(i in 2:length(taskNames)){
			
			taskMatrix[,i-1]<-(tadata@nbdaData$task==levels(tadata@nbdaData$task)[i])*1
			
		}
		asoc<-cbind(asoc,taskMatrix);
		
	}
	
	if(group){
		
		groupNames<-paste("group",levels(tadata@nbdaData$group),sep="");
		groupMatrix<-matrix(nrow=dim(tadata@nbdaData)[1],ncol=length(groupNames)-1,dimnames=list(NULL,groupNames[-1]))
		for(i in 2:length(groupNames)){
			
			groupMatrix[,i-1]<-(tadata@nbdaData$group==levels(tadata@nbdaData$group)[i])*1
			
		}
		asoc<-cbind(asoc,groupMatrix);		
	}


	#Calulate the appropriate value of s for each line of data
	if(is.null(tadata@nbdaData$sParamIndex)){
	
		sVect<-rep(l[1],dim(tadata@nbdaData)[1])
	
	}else{
		
		stemp<-1*(tadata@nbdaData$sParamIndex>0)
		stemp[which(tadata@nbdaData$sParamIndex>0)]<-l[tadata@nbdaData$sParamIndex[tadata@nbdaData$sParamIndex>0]]
		sVect<-stemp;
		
	}

	
	#Check if there are any asocial variables, set rate to one if not
	if(dim(asoc)[2]==0){
		lpRateAsoc<-rep(0,dim(tadata@nbdaData)[1]);
	}else{
		lpRateAsoc<-apply(t(matrix(rep(l[-(1:(noSParam+1))],dim(tadata@nbdaData)[1]),nrow=dim(asoc)[2]))*asoc,1,sum);
	}

	lpSocTrans<-sVect*tadata@nbdaData[6];
		
	if(bounded==FALSE){

		if(additive==TRUE){
	

			likelihood<-tadata@nbdaData$status*(log(l[noSParam+1])+log(lpSocTrans+exp(lpRateAsoc))-(l[noSParam+1]*(lpSocTrans+exp(lpRateAsoc)))*tadata@nbdaData$time)+(1-tadata@nbdaData$status)*((-l[noSParam+1]*(lpSocTrans+exp(lpRateAsoc)))*tadata@nbdaData$time);
			
		}else{
			
			likelihood<-tadata@nbdaData$status*(log(l[noSParam+1])+log(lpSocTrans+1)+lpRateAsoc-l[noSParam+1]*(lpSocTrans+1)*exp(lpRateAsoc)*tadata@nbdaData$time)+(1-tadata@nbdaData$status)*(-l[noSParam+1]*(lpSocTrans+1)*exp(lpRateAsoc)*tadata@nbdaData$time);
		
		}

	
	}

	if(bounded==TRUE){

		return("Bounded parameterisation not yet available for time of acquisition models");

	}


	return(-sum(likelihood));

}

nullTadaLikelihood<-function(l,tadata,sParam=NULL,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE){
	
	if(is.null(sParam)){
		noSParam<-1
		
	}else{
		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
	}

	l<-c(rep(0,noSParam),l);
	tadaLikelihood(l,tadata,bounded=FALSE,asocialVar=asocialVar,task=task,group=group,additive=additive,sParam=sParam);

}

#Combine the coxdata from two or more oaData objects, into an oaData object with NA in all other slots
combineTadaData<-function(taNames,sParam=NULL){


	#Set the default sParam if nothing is specified
	if(is.null(sParam)) sParam<-rep(1,length(taNames));
	
	newTaObject<-eval(as.name(taNames[1]));
	sParamIndex<-rep(sParam[1],dim(newTaObject@nbdaData)[1])
	
	for(i in 2:length(taNames)){
	

			newTaObject@nbdaData<-rbind(newTaObject@nbdaData,eval(as.name(taNames[i]))@nbdaData);
			sParamIndex<-c(sParamIndex,rep(sParam[i],dim(eval(as.name(taNames[i]))@nbdaData)[1]));		
	}
	newTaObject@oadata@idname<-NA;
	newTaObject@oadata@assMatrix<-matrix(NA);
	newTaObject@oadata@asoc<-matrix(NA);
	newTaObject@oadata@orderAcq<-NA;
	newTaObject@oadata@groupid<-"NA";
	newTaObject@oadata@taskid<-"NA";
	newTaObject@oadata@mldata<-data.frame(NA);
	newTaObject@oadata@coxdata<-data.frame(NA);

	newTaObject@nbdaData<-cbind(newTaObject@nbdaData,sParamIndex)

	return(newTaObject);
}


setMethod("anova",
    signature(object = "tadaFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.null(object@optimisation$par[1])){
    
    		table<-data.frame(Df=c(1,0),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));

    
    	}else{   

   		   table<-data.frame(Df=c(length(object@optimisation$par),length(object@optimisation$par)-noSParam),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));
   		 
   		}

		if(object@additive){atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine additively\n", class = c("anova", "data.frame"))}else{
atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine multiplicatively\n", class = c("anova", "data.frame"))			
		}
		
		return(atable);

    }
)


setMethod("summary",
    signature(object = "tadaFit"),
    function (object,...) 
    {


		if(object@additive==T)cat("Summary of Additive Social Transmission Model\nTime of Acquisition Data\n");
		if(object@additive==F)cat("Summary of Multiplicative Social Transmission Model\nTime of Acquisition Data\n");

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

# 		if(length(object@optimisation$par)<3){

  			if(object@bounded==FALSE) {
  				cat("Unbounded parameterisation.");
	    	   	sumtable<-data.frame(Estimate=c(object@optimisation$par),Bounded=c(object@optimisation$par[1: noSParam]/(1+object@optimisation$par[1: noSParam]),rep("",length(object@optimisation$par)-noSParam)),row.names=object@varNames);   
  
    	}		
    	
    	
    	
 #   	}else{		

#    	}
		
		
		cat("\n\nCoefficients\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
       
    }
)

#Define class of object for the fitted additive model
setClass("addFit",representation(optimisation="list",optimisationNull="list",sParam="numeric",bounded="logical",loglik="numeric",aic="numeric",aicc="numeric",nulllik="numeric",aicNull="numeric",aiccNull="numeric",LRTsocTransTS="numeric",LRTsocTransPV="numeric",asocialVar="numeric",varNames="character"));


#Method for initializing addFit object- including model fitting
setMethod("initialize",
    signature(.Object = "addFit"),
    function (.Object, oadata,sParam=NULL, startValue=startValue,bounded=bounded,method=method,interval=interval,asocialVar=NULL,task=F,group=F,controlIn=control,...) 
    {

		#If sParam is NULL this means only one s parameter is required
		
			if(class(oadata)=="oaData") {
				subdata<-oadata;
				diffSum<-length(levels(as.factor(subdata@diffvect)))
				if(is.na(subdata@diffvect[1]))diffSum<-1
				if(is.null(sParam)) 	sParam<-rep(1,diffSum);
			}else{
				#Count the number of diffusions and build group and task vectors
				diffSum<-0;
				groupvect<-taskvect<-NULL;
				for(i in 1:length(oadata)){
					subdata<-eval(as.name(oadata[i]));
					if(is.na(subdata@diffvect[1])){diffSum<-diffSum+1}else{diffSum<-diffSum+length(levels(as.factor(subdata@diffvect)))}
					groupvect<-c(groupvect,subdata@groupvect);
					taskvect<-c(taskvect,subdata@taskvect);
				}
				if(is.null(sParam)) sParam<-rep(1,diffSum);
			}
	
		#Calculate the number of different s parameters
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		#Record asocialVar names before adding in group and task indicator variables
		if(is.null(asocialVar)){asocialVarNames<-NULL}else{asocialVarNames<-(unlist(dimnames(subdata@asoc)[2]))[asocialVar]};

		#Add indicator varaibles for group and task
		groupList<-taskList<-NULL
		if(group){
			noGroupVar<-length(levels(as.factor(groupvect))[-1]);
			groupList<-levels(as.factor(groupvect))[-1];
			asocialVar<-c(asocialVar,dim(subdata@asoc)[2]+(1:noGroupVar))
		}else{noGroupVar<-0}
		if(task){
			noTaskVar<-length(levels(as.factor(taskvect))[-1]);
			taskList<-levels(as.factor(taskvect))[-1];
			asocialVar<-c(asocialVar,dim(subdata@asoc)[2]+(1:noTaskVar)+noGroupVar)
		}else{noTaskVar<-0}

		
		#Set staring values if not specified by the user
		if(is.null(startValue)) 		startValue<-rep(0,length(asocialVar)+ noSParam);

		#Set vector of upper and lower values for each parameter. Unbounded for asocial learning variables
		lower<-c(rep(0,noSParam),rep(-Inf,length(asocialVar)));
		if (bounded==T)	upper<-c(rep(1,noSParam),rep(Inf,length(asocialVar)));
		if (bounded==F)	upper<-rep(Inf,length(asocialVar)+noSParam);
		
		if(is.null(interval)) interval<-c(0,999);
			
		#Optimise for s
		#If there are no asocial learning variables and a single st parameter we can use optimise. This is the default unless nlminb is specified. If there are asocial learning variables, nlminb is used
		if(length(startValue)==1){
			if(method=="nlminb"){	
				if(bounded==T){
					fit1<-nlminb(start=startValue,objective=addLikelihood,lower=0,upper=1,data=oadata,bounded=T,asocialVar=NULL,groupList=groupList, taskList=taskList,sParam=sParam, control=controlIn);

				}else{			
					fit1<-nlminb(start=startValue,objective=addLikelihood,lower=0,upper=Inf,data=oadata,bounded=F,asocialVar=NULL,groupList=groupList, taskList=taskList,sParam=sParam, control=controlIn);
					}
			}else{
				if(bounded==T){
					fit1<-optimise(f=addLikelihood,interval=c(0,1),data=oadata,bounded=T,asocialVar=NULL,groupList=groupList, taskList=taskList,sParam=sParam);
				}else{			
					fit1<-optimise(f=addLikelihood,interval=interval,data=oadata,bounded=F,asocialVar=NULL,groupList=groupList, taskList=taskList,sParam=sParam);
				}					
			}
			nulllik<-nulladdLikelihood(0,oadata,bounded=bounded);
			nullfit<-list(0)
			
		}else{
			if(bounded==T){
				fit1<-nlminb(start=startValue,objective=addLikelihood,lower=lower,upper=upper,data=oadata,bounded=T,asocialVar=asocialVar,sParam=sParam,groupList=groupList, taskList=taskList, control=controlIn);
				if(length(startValue)==noSParam){
					nulllik<-nulladdLikelihood(0,oadata,bounded=bounded);
					nullfit<-list(0);
				}else{
					nullfit<-nlminb(start=startValue[-(1:noSParam)],objective=nulladdLikelihood,data=oadata,bounded=T,asocialVar=asocialVar,groupList=groupList, taskList=taskList);
					nulllik<-nullfit$objective;
				}
			}else{			
				fit1<-nlminb(start=startValue,objective=addLikelihood,lower=lower,upper=upper,data=oadata,bounded=F,asocialVar=asocialVar,sParam=sParam,groupList=groupList, taskList=taskList, control=controlIn);
				if(length(startValue)==noSParam){
					nulllik<-nulladdLikelihood(0,oadata,bounded=bounded);
					nullfit<-list(0);
				}else{
					nullfit<-nlminb(start=startValue[-(1:noSParam)],objective=nulladdLikelihood,data=oadata,bounded=F,asocialVar=asocialVar,groupList=groupList, taskList=taskList);
					nulllik<-nullfit$objective;
				}
			}
			
		}

		
		#Perform LRT for social transmission
		loglik<-fit1$objective;
		LRTsocTransTS<-2*(nulllik-loglik)
		LRTsocTransPV<-1-pchisq(LRTsocTransTS,df=noSParam);

		if(is.null(asocialVar)) asocialVar<-NaN;

		sampleSize<-sampSizeExtract(oadata);

		#Calculate aic and for model without social transmission
		aic<-2*length(fit1$par)+2*loglik;
		aicc<-2*(length(fit1$par))*(sampleSize/(sampleSize-(length(fit1$par))-1))+2*loglik;

		if (is.nan(asocialVar[1])){
			aicNull<-2*nulllik;
			aiccNull<-aicNull;
			aic<-2*noSParam+2*loglik;
			aicc<-2*noSParam*(sampleSize/(sampleSize-2))+2*loglik;
		}else{
			aicNull<-2*length(asocialVar)+2*nulllik;
			aiccNull<-2*(length(asocialVar))*(sampleSize/(sampleSize-(length(asocialVar))-1))+2*nulllik;
		}	

		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)){}else{
			if(aicc<aic) aicc<-NaN;
		}
		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aicNull)|is.nan(aiccNull)){}else{
			if(aiccNull<aicNull) aiccNull<-NaN;
		}
		
		#Extract names of variables
		if(is.nan(asocialVar[1])){
	
			varNames<-c(paste("Social transmission",1:noSParam));			
			
		}else{
			

				groupNames<-taskNames<-NULL;
				if(group) groupNames<-paste("Group",(unlist(levels(as.factor(groupvect))))[-1],sep="");
				if(task) taskNames<-paste("Task",(unlist(levels(as.factor(taskvect))))[-1],sep="");
			
				varNames<-c(paste("Social transmission",1:noSParam), asocialVarNames,groupNames,taskNames);
		}


        callNextMethod(.Object,bounded=bounded, optimisation=fit1,optimisationNull=nullfit,sParam=sParam,loglik=loglik, aic=aic,aicc=aicc,nulllik=nulllik,aicNull=aicNull,aiccNull=aiccNull,LRTsocTransTS=LRTsocTransTS,LRTsocTransPV=LRTsocTransPV,asocialVar=asocialVar,varNames=varNames,...)
        

       
    }
)		
	
#Function for implementing the initialization
addFit<-function(oadata,sParam=NULL,startValue=NULL,bounded=FALSE,interval=c(0,999),method="optimise",asocialVar=NULL,group=F, task=F, control=NULL){
		
	new("addFit",oadata=oadata,sParam=sParam,startValue=startValue,bounded=bounded,interval=interval,method=method,asocialVar=asocialVar,group=group, task=task, control=control)	
}


#Calculates likelihood for the additive model. If true ties are given then the likelihood is adjusted for these
addLikelihood<-function(l,data,asocialVar=NULL,bounded=FALSE, sParam=NULL, groupList=NULL, taskList=NULL){

	#Run appropriate function depending on which parameterisation for s has been specified
	if(bounded==T){
		
		likelihood<-addLikelihoodBounded(l=l,data=data,asocialVar=asocialVar, sParam= sParam, groupList=groupList, taskList=taskList);
		
	} else {
	
		likelihood<-addLikelihoodNotBounded(l=l,data=data,asocialVar=asocialVar, sParam= sParam, groupList=groupList, taskList=taskList);
	
	}

if(is.character(data)){
	
		return(likelihood);
	
	}else{
		if(is.null(unlist(data@trueTies[1]))){
				return(likelihood);
		}else{
		
			for (i in 1:length(data@trueTies)){
			
				likelihood<-likelihood+addCorrectTrueTie(l=l,data=data,tiePosition=unlist(data@trueTies[i]),asocialVar=asocialVar,bounded=bounded);
			
			}
			return(likelihood);
		}
	}
}


#Calculates null likelihood for the additive model (s is fixed to zero, so only the other parameters are optimised by nlminb)
nulladdLikelihood<-function(l,data,asocialVar=NULL,bounded=FALSE,groupList=NULL, taskList=NULL){

	#Adds a zero to the parameter vector to correspond to no social transmission
	l<-c(0,l)

	likelihood<-addLikelihood(l=l,data=data,asocialVar=asocialVar,bounded=bounded,groupList= groupList, taskList= taskList)


	return(likelihood);

}


#Calculates likelihood for the additive model (single group) with s between 0 and 1
addLikelihoodBounded<-function(l,data,asocialVar=NULL, sParam=NULL,noSParam=NULL,groupList=NULL, taskList=NULL){

#Define required function
sumWithoutNA<-function(x) sum(na.omit(x))

#If the data if a character vector containing multiple oaData objects, make separate calls to addLikelihood and add together
if(is.character(data)){		

		#If sParam is NULL this means only one s parameter is required
		if(is.null(sParam)) {
			#Count the number of diffusions
			diffSum<-0;
			for(i in 1:length(data)){
				subdata<-eval(as.name(data[i]));
				if(is.na(subdata@diffvect[1])){diffSum<-diffSum+1}else{diffSum<-diffSum+length(levels(as.factor(subdata@diffvect)))}
			}
			sParam<-rep(1,diffSum);
		}
		#Calculate the number of different s parameters
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
			
		totalLikelihood<-0;
		diffcount<-0
			
		for(i in 1:length(data)){
			subdata<-eval(as.name(data[i]));
			if(is.na(subdata@diffvect[1])){noDiffs<-1}else{noDiffs<-length(levels(as.factor(subdata@diffvect)))}
			newSParam<-sParam[(diffcount+1):(diffcount+ noDiffs)]
		if(noDiffs==1){
			if (sParam[i]==0){
				
				newl<-l[-(1:noSParam)];
				totalLikelihood<- totalLikelihood + nulladdLikelihood(l=newl,data= subdata,asocialVar=asocialVar,bounded=T,groupList=groupList, taskList=taskList);
				
			}else{
			#Build the parameter vector for the diffusion in question by selecting the correct st parameter and adding the others on the end

				newl<-c(l[sParam[i]],l[-(1:noSParam)]);
				totalLikelihood<-totalLikelihood+addLikelihoodBounded(l=newl,data= subdata,asocialVar=asocialVar,groupList=groupList, taskList=taskList);
			}
				
		}else{
			#This bit is for if we have multiple diffusions because of stratified data			#Build the parameter vector for the diffusion in question by selecting the correct st parameter and adding the others on the end

			newNoSParam<-length(levels(as.factor(newSParam[newSParam>0])))
			totalLikelihood<-totalLikelihood+addLikelihoodBounded(l=l,data= subdata,asocialVar=asocialVar,sParam=newSParam,noSParam=newNoSParam,groupList=groupList, taskList=taskList);
			
		}
		diffcount<-diffcount+noDiffs;
		
		
		}
			
		return(totalLikelihood);
					
}else{


	#If sParam is NULL this means only one s parameter is required
	if(is.null(sParam)) sParam<-rep(1,length(data@diffvect));

	#Calculate the number of different s parameters
	if(is.null(noSParam)){ noSParam<-length(levels(as.factor(sParam[sParam>0])))}

	#Now we need to account for the sParam=0
	if(sum(sParam==0)>0){
		l<-c(0,l);
		sParam<-sParam+1;
		noSParam<-noSParam+1;
	}

	#Calculate indicator variables for task and group if necessary
	if(is.null(groupList)){groupMatrix<-NULL}else{
	groupMatrix<-matrix(ncol=length(groupList),nrow=length(data@groupvect));
	for(i in 1:length(groupList)){
		groupMatrix[,i]<-(data@groupvect==groupList[i])*1;
	}
	dimnames(groupMatrix)[2]<-list(levels(as.factor(data@groupvect))[-1])
	}
	if(is.null(taskList)){taskMatrix<-NULL}else{
	taskMatrix<-matrix(ncol=length(taskList),nrow=length(data@taskvect));
	for(i in 1:length(taskList)){
		taskMatrix[,i]<-(data@taskvect==taskList[i])*1;
	}
	dimnames(taskMatrix)[2]<-list(levels(as.factor(data@taskvect))[-1])
	}
	
	asoc<-cbind(data@asoc,groupMatrix,taskMatrix)

	#Cut down to the asocial variables specified by asocialVar and add in group and task indicators
	asoc<-as.matrix(asoc[,asocialVar]);
	
	#Check if there are any asocial variables, set rate to one if not
	if(is.null(asocialVar)){
		solverAsocialRate<-rep(1,length(data@orderAcq));
	}else{
		#Multiply asocial variable by coefficients
		multiCoef<-as.matrix(asoc[data@orderAcq,]*t(matrix(l[-(1:noSParam)],ncol=length(data@orderAcq),nrow=length(asocialVar))));
		#Add rows for the linear predictor for asocial learning for each solving individual
		solverLinearPred<-apply(multiCoef,1,sum);
		#Take exponentials to get rate of asocial learning
		solverAsocialRate<-exp(solverLinearPred);
	}
	
	#Now find the rate of social transmission by multiplying by the social transmission parameter
	if(is.na(data@diffvect[1])){	

		#Now find the rate of non-solver social transmission
		solverSocialTransRate<-data@mldata$learnMetric*l[1];
		#Total rate for the solver at each stage
		solverTotalRate<-solverAsocialRate*(1-l[1])+solverSocialTransRate;

	}else{
			
		solverSocialTransRate<-data@mldata$learnMetric*l[sParam[data@diffvect[data@mldata$orderAcq]]]
		#Total rate for the solver at each stage
		solverTotalRate<-solverAsocialRate*(1-l[sParam[data@diffvect[data@mldata$orderAcq]]])+solverSocialTransRate;
	}



	#Take logs and add across acquisition events
	lComp1<-sum(log(solverTotalRate));
	

	#Generate a linear predictor for the rate of asocial acquisition for each ind at each time step	
	if(is.na(data@demons[1])){
		statusTracker<-rep(0,dim(data@assMatrix)[1]);
	}else{
		statusTracker<-data@demons;	
	}


	nsAsocialRate<-vector(length=length(data@orderAcq));
	
	for(i in 1:length(data@orderAcq)){

		#If there are no asocial learning variables set everyone to zero
		if(is.null(asocialVar)){
			nsLinearPred<-rep(0,dim(data@assMatrix)[1])
		}else{
		
			#Multiply asocial learning variables by coefficients
			nsMultiCoef<-as.matrix(asoc*t(matrix(l[-(1:noSParam)],ncol=dim(data@assMatrix)[1],nrow=length(asocialVar))));
			#Add rows to get linear predictors for asocial learning
			nsLinearPred<-apply(nsMultiCoef,1,sum);

		}

		if(is.na(data@diffvect[1])){

			#Take exponentials and set =0 if for individuals who have already acquired the behaviour
			nsAsocialRate[i]<-sum((1-statusTracker)*exp(nsLinearPred)*(1-l[1]));

		}else{

			#Take exponentials and set =0 if for individuals who have already acquired the behaviour
			nsAsocialRate[i]<-sum((1-statusTracker)*exp(nsLinearPred)*(1-l[sParam[data@diffvect]]));
		}
		
		statusTracker[data@orderAcq[i]]<-1;
	}
	
	if(is.na(data@diffvect[1])){	
		#Now find the rate of non-solver social transmission
		nsSocialTransRate<-data@mldata$totalMetric*l[1]
	}else{
			
		tempSRs<-t(matrix(l[sParam],nrow=length(sParam),ncol=dim(data@mldata)[1]))*data@mldata[,4+(1:length(levels(as.factor(data@diffvect))))]
		nsSocialTransRate<-apply(tempSRs,1,sum)
	}
	
	#Add together
	nsTotalRate<-nsSocialTransRate+nsAsocialRate;
	

	#Sum the asocial and social rates, take logs and add across acquisition events
	lComp2<-sum(log(nsTotalRate));	
	#Return the likelihood for the additive model
	return(-lComp1+lComp2)
}
}

#Calculates likelihood for the additive model (single group) with s between 0 and 1
addLikelihoodNotBounded<-function(l,data,asocialVar=NULL, sParam=NULL,noSParam=NULL,groupList=NULL, taskList=NULL){

#Define required function
sumWithoutNA<-function(x) sum(na.omit(x))

#If the data if a character vector containing multiple oaData objects, make separate calls to addLikelihood and add together
if(is.character(data)){		

		#If sParam is NULL this means only one s parameter is required
		if(is.null(sParam)) {
			#Count the number of diffusions
			diffSum<-0;
			for(i in 1:length(data)){
				subdata<-eval(as.name(data[i]));
				if(is.na(subdata@diffvect[1])){diffSum<-diffSum+1}else{diffSum<-diffSum+length(levels(as.factor(subdata@diffvect)))}
			}
			sParam<-rep(1,diffSum);
		}
		#Calculate the number of different s parameters
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
			
		totalLikelihood<-0;
		diffcount<-0
			
		for(i in 1:length(data)){
			subdata<-eval(as.name(data[i]));
			if(is.na(subdata@diffvect[1])){noDiffs<-1}else{noDiffs<-length(levels(as.factor(subdata@diffvect)))}
			newSParam<-sParam[(diffcount+1):(diffcount+ noDiffs)]
		if(noDiffs==1){
			if (sParam[i]==0){
				
				newl<-l[-(1:noSParam)];
				totalLikelihood<- totalLikelihood + nulladdLikelihood(l=newl,data= subdata,asocialVar=asocialVar,bounded=T,groupList=groupList, taskList=taskList);
				
			}else{
			#Build the parameter vector for the diffusion in question by selecting the correct st parameter and adding the others on the end

				newl<-c(l[sParam[i]],l[-(1:noSParam)]);
				totalLikelihood<-totalLikelihood+addLikelihoodNotBounded(l=newl,data= subdata,asocialVar=asocialVar,groupList=groupList, taskList=taskList);
			}
				
		}else{
			#This bit is for if we have multiple diffusions because of stratified data			#Build the parameter vector for the diffusion in question by selecting the correct st parameter and adding the others on the end

			newNoSParam<-length(levels(as.factor(newSParam[newSParam>0])))
			totalLikelihood<-totalLikelihood+addLikelihoodNotBounded(l=l,data= subdata,asocialVar=asocialVar,sParam=newSParam,noSParam=newNoSParam,groupList=groupList, taskList=taskList);
			
		}
		diffcount<-diffcount+noDiffs;
		
		
		}
			
		return(totalLikelihood);
					
}else{


	#If sParam is NULL this means only one s parameter is required
	if(is.null(sParam)) sParam<-rep(1,length(data@diffvect));

	#Calculate the number of different s parameters
	if(is.null(noSParam)){ noSParam<-length(levels(as.factor(sParam[sParam>0])))}

	#Now we need to account for the sParam=0
	if(sum(sParam==0)>0){
		l<-c(0,l);
		sParam<-sParam+1;
		noSParam<-noSParam+1;
	}

	#Calculate indicator variables for task and group if necessary
	if(is.null(groupList)){groupMatrix<-NULL}else{
	groupMatrix<-matrix(ncol=length(groupList),nrow=length(data@groupvect));
	for(i in 1:length(groupList)){
		groupMatrix[,i]<-(data@groupvect==groupList[i])*1;
	}
	dimnames(groupMatrix)[2]<-list(levels(as.factor(data@groupvect))[-1])
	}
	if(is.null(taskList)){taskMatrix<-NULL}else{
	taskMatrix<-matrix(ncol=length(taskList),nrow=length(data@taskvect));
	for(i in 1:length(taskList)){
		taskMatrix[,i]<-(data@taskvect==taskList[i])*1;
	}
	dimnames(taskMatrix)[2]<-list(levels(as.factor(data@taskvect))[-1])
	}
	
	asoc<-cbind(data@asoc,groupMatrix,taskMatrix)

	#Cut down to the asocial variables specified by asocialVar and add in group and task indicators
	asoc<-as.matrix(asoc[,asocialVar]);
	
	#Check if there are any asocial variables, set rate to one if not
	if(is.null(asocialVar)){
		solverAsocialRate<-rep(1,length(data@orderAcq));
	}else{
		#Multiply asocial variable by coefficients
		multiCoef<-as.matrix(asoc[data@orderAcq,]*t(matrix(l[-(1:noSParam)],ncol=length(data@orderAcq),nrow=length(asocialVar))));
		#Add rows for the linear predictor for asocial learning for each solving individual
		solverLinearPred<-apply(multiCoef,1,sum);
		#Take exponentials to get rate of asocial learning
		solverAsocialRate<-exp(solverLinearPred);
	}
	
	#Now find the rate of social transmission by multiplying by the social transmission parameter
	if(is.na(data@diffvect[1])){	

		#Now find the rate of non-solver social transmission
		solverSocialTransRate<-data@mldata$learnMetric*l[1];
		#Total rate for the solver at each stage
		solverTotalRate<-solverAsocialRate+solverSocialTransRate;

	}else{
			
		solverSocialTransRate<-data@mldata$learnMetric*l[sParam[data@diffvect[data@mldata$orderAcq]]]
		#Total rate for the solver at each stage
		solverTotalRate<-solverAsocialRate+solverSocialTransRate;
	}



	#Take logs and add across acquisition events
	lComp1<-sum(log(solverTotalRate));
	

	#Generate a linear predictor for the rate of asocial acquisition for each ind at each time step	
	if(is.na(data@demons[1])){
		statusTracker<-rep(0,dim(data@assMatrix)[1]);
	}else{
		statusTracker<-data@demons;	
	}


	nsAsocialRate<-vector(length=length(data@orderAcq));
	
	for(i in 1:length(data@orderAcq)){

		#If there are no asocial learning variables set everyone to zero
		if(is.null(asocialVar)){
			nsLinearPred<-rep(0,dim(data@assMatrix)[1])
		}else{
		
			#Multiply asocial learning variables by coefficients
			nsMultiCoef<-as.matrix(asoc*t(matrix(l[-(1:noSParam)],ncol=dim(data@assMatrix)[1],nrow=length(asocialVar))));
			#Add rows to get linear predictors for asocial learning
			nsLinearPred<-apply(nsMultiCoef,1,sum);

		}

		if(is.na(data@diffvect[1])){

			#Take exponentials and set =0 if for individuals who have already acquired the behaviour
			nsAsocialRate[i]<-sum((1-statusTracker)*exp(nsLinearPred));

		}else{

			#Take exponentials and set =0 if for individuals who have already acquired the behaviour
			nsAsocialRate[i]<-sum((1-statusTracker)*exp(nsLinearPred));
		}
		
		statusTracker[data@orderAcq[i]]<-1;
	}
	
	if(is.na(data@diffvect[1])){	
		#Now find the rate of non-solver social transmission
		nsSocialTransRate<-data@mldata$totalMetric*l[1]
	}else{
			
		tempSRs<-t(matrix(l[sParam],nrow=length(sParam),ncol=dim(data@mldata)[1]))*data@mldata[,4+(1:length(levels(as.factor(data@diffvect))))]
		nsSocialTransRate<-apply(tempSRs,1,sum)
	}
	
	#Add together
	nsTotalRate<-nsSocialTransRate+nsAsocialRate;
	

	#Sum the asocial and social rates, take logs and add across acquisition events
	lComp2<-sum(log(nsTotalRate));	
	#Return the likelihood for the additive model
	return(-lComp1+lComp2)
}
}


setMethod("summary",
    signature(object = "addFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])));


		cat("Summary of Additive Social Transmission Model\nOrder of acquisition data\n");
    	if(is.null(object@optimisation$par[1])){
			if(object@bounded==T) {
				cat("Bounded parameterisation");
				sumtable<-data.frame(Estimate=object@optimisation$minimum,Unbounded=object@optimisation$minimum/(1-object@optimisation$minimum),row.names=object@varNames);
			}
			if(object@bounded==F) {
				cat("Unbounded parameterisation");
				sumtable<-data.frame(Estimate=object@optimisation$minimum,Bounded=object@optimisation$minimum/(1+object@optimisation$minimum),row.names=object@varNames);
			}

    	}else{
    		
  			if(object@bounded==T) {
  				cat("Bounded parameterisation");
	    	   	sumtable<-data.frame(Estimate=c(object@optimisation$par),Unbounded=c(object@optimisation$par[1:noSParam]/(1-object@optimisation$par[1:noSParam]),rep(NA,length(object@optimisation$par)-noSParam)),row.names=object@varNames);
	    	}
  			if(object@bounded==F) {
  				cat("Unbounded parameterisation");
	    	   	sumtable<-data.frame(Estimate=c(object@optimisation$par),Bounded=c(object@optimisation$par[1:noSParam]/(1+object@optimisation$par[1:noSParam]),rep(NA,length(object@optimisation$par)-noSParam)),row.names=object@varNames);
	    	}
    	 
    	
    	}


		if(object@bounded==T) 
		if(object@bounded==F) cat("Unbounded parameterisation");		
		
		cat("\n\nCoefficients\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
       
    }
)

setMethod("anova",
    signature(object = "addFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.null(object@optimisation$par[1])){
    
    		table<-data.frame(Df=c(1,0),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));

    
    	}else{   

   		   table<-data.frame(Df=c(length(object@optimisation$par),length(object@optimisation$par)-noSParam),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));
   		 
   		}


		atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine additively\n", class = c("anova", "data.frame"));

		
		return(atable);

    }
)


#Extracts the sample size for oaData in order for aicc to be calculated. As Burnham and Anderson (2002) point out, sample size is not always a straithforward issue. Here we take it to be the number of acquisition events
sampSizeExtract<-function(oadata){
	
	if(class(oadata)=="oaData"){
		
		return(length(oadata@orderAcq))
		
	}

	if(class(oadata)=="taData"){
		
		return(length(oadata@oadata@orderAcq))
		
	}	

	
	if(is.character(oadata)){		
			
		totalSS<-0;
			
		for(i in 1:length(oadata)){
			
			totalSS<-totalSS+sampSizeExtract(eval(as.name(oadata[i])));
				
		}
			
		return(totalSS);	
	}
	
}

	
#This function takes an oadata object and the specified position of a "true tie": where the order in which a specified subset of the diffusion chain is not known, and returns a correction to the likelihood calculated from the order given in the oadata object. If the number of tied individuals is too large there are too many permutations and the function will not work. This is the version for the additive model. There is not yet a version for the multiplicative model.
addCorrectTrueTie<-function(l,data,tiePosition,asocialVar=NULL,bounded=FALSE){
	
	status<-rep(0,length(data@orderAcq));
	if(min(tiePosition)!=1)status[data@orderAcq[1:(min(tiePosition)-1)]]<-1;
	
	#Cut down to the asocial variables specified by asocialVar
	asoc<-as.matrix(data@asoc[,asocialVar]);	
	
	#Get likelihood for the given order
	#Check if there are any asocial variables, set rate to one if not
	if(is.null(asocialVar)){
		solverAsocialRate<-rep(1,length(tiePosition));
	}else{
		#Multiply asocial variable by coefficients
		multiCoef<-as.matrix(asoc[data@orderAcq[tiePosition],]*t(matrix(l[-1],ncol=length(tiePosition),nrow=length(asocialVar))));
		#Add rows for the linear predictor for asocial learning for each solving individual
		solverLinearPred<-apply(multiCoef,1,sum);
		#Take exponentials to get rate of asocial learning
		solverAsocialRate<-exp(solverLinearPred);
	}
	
	#Now find the rate of social transmission by multiplying by the social transmission parameter
	solverSocialTransRate<-data@mldata$learnMetric[tiePosition]*l[1];
	
	#Total rate for the solver at each stage
	if (bounded==T )solverTotalRate<-solverAsocialRate*(1-l[1])+solverSocialTransRate;
	if (bounded==F )solverTotalRate<-solverAsocialRate+solverSocialTransRate;

	#Take logs and add across acquisition events
	lComp1<-sum(log(solverTotalRate));	
		#Generate a linear predictor for the rate of asocial acquisition for each ind at each time step	
	#Set status tracker to the start of the tied sequence
	statusTracker<-status;

	nsAsocialRate<-vector(length=length(tiePosition));
	
	for(i in tiePosition){

		#If there are no asocial learning variables set everyone to zero
		if(is.null(asocialVar)){
			nsLinearPred<-rep(0,dim(data@assMatrix)[1])
		}else{
		
			#Multiply asocial learning variables by coefficients
			nsMultiCoef<-as.matrix(asoc*t(matrix(l[-1],ncol=dim(data@assMatrix)[1],nrow=length(asocialVar))));
			#Add rows to get linear predictors for asocial learning
			nsLinearPred<-apply(nsMultiCoef,1,sum);

		}

		#Take exponentials and set =0 if for individuals who have already acquired the behaviour
		nsAsocialRate[i+1-tiePosition[1]]<-sum((1-statusTracker)*exp(nsLinearPred));
		
		statusTracker[data@orderAcq[i]]<-1;
	}
	
		
	#Now find the rate of non-solver social transmission
	nsSocialTransRate<-data@mldata$totalMetric[tiePosition]*l[1]
	#Add together
	if (bounded==T) nsTotalRate<-nsSocialTransRate+nsAsocialRate*(1-l[1]);
	if (bounded==F) nsTotalRate<-nsSocialTransRate+nsAsocialRate;	

	#Sum the asocial and social rates, take logs and add across acquisition events
	lComp2<-sum(log(nsTotalRate));	

	#Likelihood for the observed order of data within the tie
	givenOrderLogLik<--lComp1+lComp2
	
	#Now find the number of possible permutations within the tie
	
	perms<-permn(data@orderAcq[tiePosition]);
	
	
	#Now record the likelihood for each possible order within the tie
	
	likelihoodrecord<-vector(length=length(perms));
	
	for(i in 1:length(perms)){
		
		newOrderAcq<-data@orderAcq;
		newOrderAcq[tiePosition]<-unlist(perms[i]);
		
		#Get likelihood for each order
		#Check if there are any asocial variables, set rate to one if not
		if(is.null(asocialVar)){
			solverAsocialRate<-rep(1,length(tiePosition));
		}else{
			#Multiply asocial variable by coefficients
			multiCoef<-as.matrix(asoc[newOrderAcq[tiePosition],]*t(matrix(l[-1],ncol=length(tiePosition),nrow=length(asocialVar))));
			#Add rows for the linear predictor for asocial learning for each solving individual
			solverLinearPred<-apply(multiCoef,1,sum);
			#Take exponentials to get rate of asocial learning
			solverAsocialRate<-exp(solverLinearPred);
		}
	
		#Now find the rate of social transmission (for solver and total)

		#Define functions for calculating total st Metric
		newFunc<-function(x) x*(1-statusTracker)
		newFunc2<-function(x) x*statusTracker
		
		newLearnMetric<-vector();
		newtotalMetric<-vector();
		nsAsocialRate<-vector();

	
		#Set status tracker to the start of the tied sequence
		statusTracker<-status;

		for(j in tiePosition){
			
			newLearnMetric<-c(newLearnMetric,sum(data@assMatrix[newOrderAcq[j],]*statusTracker));
			#Calculate total st metric over all individuals in the group for each acquisition event (ML model)
			newMatrix<-apply(data@assMatrix,2,newFunc)
			newtotalMetric<-c(newtotalMetric,sum(apply(newMatrix,1,newFunc2)));

			#If there are no asocial learning variables set everyone to zero
			if(is.null(asocialVar)){
				nsLinearPred<-rep(0,dim(data@assMatrix)[1])
			}else{
		
				#Multiply asocial learning variables by coefficients
				nsMultiCoef<-as.matrix(asoc*t(matrix(l[-1],ncol=dim(data@assMatrix)[1],nrow=length(asocialVar))));
				#Add rows to get linear predictors for asocial learning
				nsLinearPred<-apply(nsMultiCoef,1,sum);

			}

			#Take exponentials and set =0 if for individuals who have already acquired the behaviour
			nsAsocialRate<-c(nsAsocialRate,sum((1-statusTracker)*exp(nsLinearPred)));

			
			statusTracker[newOrderAcq[j]]<-1;
			
		}

		solverSocialTransRate<-newLearnMetric*l[1];
	
		#Total rate for the solver at each stage
		if (bounded==T )solverTotalRate<-solverAsocialRate*(1-l[1])+solverSocialTransRate;
		if (bounded==F )solverTotalRate<-solverAsocialRate+solverSocialTransRate;

		#Take logs and add across acquisition events
		lComp1<-sum(log(solverTotalRate));	

		
		#Now find the rate of non-solver social transmission
		nsSocialTransRate<-newtotalMetric*l[1]
		#Add together
		if (bounded==T) nsTotalRate<-nsSocialTransRate+nsAsocialRate*(1-l[1]);
		if (bounded==F) nsTotalRate<-nsSocialTransRate+nsAsocialRate;	

		#Sum the asocial and social rates, take logs and add across acquisition events
		lComp2<-sum(log(nsTotalRate));	
		#Return the likelihood for the additive model
		likelihoodrecord[i]<--lComp1+lComp2		
	}
	
	#Calculate the total likelihood of the observed data within the tie
	logLikTie<--log(sum(exp(-likelihoodrecord)))
	
	#Calculate required adjustment by taking away the likelihood of the order given and adding the total likelihood of any order that results in the observed tie
	adjustment<--givenOrderLogLik+logLikTie;
	
	return(adjustment);
	
	
}


#A new class of data object for fitting time of acquisition models, similar to Franz & Nunn's NBDA

setClass("taData",representation(oadata="oaData",timeAcq="vector",endTime="numeric", nbdaData="data.frame"));

#Initialise method just takes an oaData object and an time of acquisition vector and makes a new taData object
setMethod("initialize",
    signature(.Object = "taData"),
    function (.Object, idname=NULL, oadata,timeAcq,endTime,updateTimes=NULL,demons=NULL,groupvect=groupvect, taskvect=taskvect, diffvect=diffvect,...) 
    {
 
 if(is.null(updateTimes)){
 
	timeVect<-c(0,timeAcq,endTime);
	 
 	nbdaData<-oadata@coxdata;
 	nbdaData$time1<-timeVect[oadata@coxdata$time1+1];
 	nbdaData$time2<-timeVect[oadata@coxdata$time1+2];
 
	
 	temp<-oadata@coxdata[(oadata@coxdata$time2==length(oadata@orderAcq))&(oadata@coxdata$status==0),];
	if(is.na(oadata@demons[1])){statusTracker<-rep(0,dim(oadata@assMatrix)[1])}else{statusTracker<-oadata@demons}
 	statusTracker[oadata@coxdata$identity[oadata@coxdata$status==1]]<-1	
 	temp$stMetric<-(apply(t(t(oadata@assMatrix)*statusTracker),1,sum))[statusTracker==0]
 	if(dim(temp)[1]>0){
 		temp$time1<-timeVect[length(timeVect)-1];
 		temp$time2<-timeVect[length(timeVect)];
 	
	 	nbdaData<-rbind(nbdaData,temp);
 		nbdaData$time<- nbdaData$time2-nbdaData$time1;
 	}
 	
 	nbdaData$time<-nbdaData$time2-nbdaData$time1;

	#If there are tied data cut down the data appropriately
	if( sum(nbdaData$time==0)>0){
		#Cut down tied data appropriately
		rem<-(nbdaData$time==0)&nbdaData$status==1;
		statUpgrade<-rep(FALSE,dim(nbdaData)[1]);
		for(i in 1: length(nbdaData[rem,]$identity)){
			statUpgrade[as.numeric(rownames(nbdaData[nbdaData$identity==nbdaData[rem,]$identity[i],])[dim(nbdaData[nbdaData$identity==nbdaData[rem,]$identity[i],])[1]-1])]<-TRUE;
		}
		nbdaData$status[statUpgrade]<-1;
		nbdaData<-nbdaData[nbdaData$time>0,]	
	}
}else{
	assMatrix<-oadata@assMatrix
	asoc<-oadata@asoc
	orderAcq<-oadata@orderAcq
	taskid<-oadata@taskid
	groupid<-oadata@groupid
	diffvect<-oadata@diffvect
	oadata@mldata<-upDateTimes(idname=idname,orderAcq,timeAcq,endTime,updateTimes,assMatrix=assMatrix,asoc,demons,taskid,groupid,return="mldata",diffvect=diffvect,groupvect=groupvect, taskvect=taskvect);
	oadata@coxdata<-upDateTimes(idname=idname,orderAcq,timeAcq,endTime,updateTimes,assMatrix,asoc,demons,taskid,groupid,return="coxdata",diffvect=diffvect,groupvect=groupvect, taskvect=taskvect);
	nbdaData<-upDateTimes(idname=idname,orderAcq,timeAcq,endTime,updateTimes,assMatrix,asoc,demons,taskid,groupid,return="nbdadata",diffvect=diffvect,groupvect=groupvect, taskvect=taskvect);
	oadata@updateTimes<-updateTimes;
	
	}


 
	callNextMethod(.Object,oadata=oadata,timeAcq=timeAcq,endTime=endTime,nbdaData=nbdaData)

}
)


#Creates a new taData object taking either an oaData object and time of acquisition vector, or all components separately
taData<-function( timeAcq, endTime, idname=NULL,oadata=NULL,assMatrix=NULL,asoc=NULL,orderAcq=NULL,groupid=NULL,taskid=NULL,updateTimes=NULL,demons=NULL,taskvect=NA,groupvect=NA,diffvect=NA,weights=NULL){
	
	if(is.null(oadata)){

		if(is.null(weights)) weights<-rep(1,dim(assMatrix)[1])

		if(is.null(taskid)) taskid<-"1";
		if(is.null(groupid)) groupid<-"1";
		oadata<-oaData(idname=idname, assMatrix=assMatrix,asoc=asoc,orderAcq=orderAcq,groupid=groupid,taskid=taskid,demons=demons,taskvect= taskvect,groupvect= groupvect, diffvect=diffvect,weights=weights);

		
	}
	

			
	new("taData",oadata=oadata,timeAcq=timeAcq, endTime=endTime,updateTimes=updateTimes,demons=demons,groupvect=groupvect,taskvect=taskvect)		

	
}


#Define class of object for the fitted discrete tada model
setClass("discreteTadaFit",representation(optimisation="list",optimisationNull="list",additive="logical",sParam="numeric",bounded="logical",loglik="numeric",aic="numeric", aicc="numeric",nulllik="numeric",aicNull="numeric",aiccNull="numeric",LRTsocTransTS="numeric",LRTsocTransPV="numeric",asocialVar="numeric",varNames="character",task="logical",group="logical")); 


#Method for initializing object- including model fitting
setMethod("initialize",
    signature(.Object = "discreteTadaFit"),
    function (.Object, tadata,additive,sParam,startValue,asocialVar,bounded,task,group,stepLength,...) 
    {

		if(is.character(tadata)){		

	
		#If sParam is NULL this means only one s parameter is required
		if(is.null(sParam)) sParam<-rep(1,length(tadata));
		#Calculate the number of different s parameters
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		#For the purposes of working out how many task and group levels there are only:
		combinedData<-combineTadaData(tadata,sParam=sParam);
		sampleSize<-sum(combinedData@nbdaData$status);
		maxRate<-max(combinedData@nbdaData$time2)+1000

		noTasks<-length(levels(combinedData@nbdaData$task));
		noGroups<-length(levels(combinedData@nbdaData$group));
		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		extraParam<-0;

		if(task) extraParam<-extraParam+length(levels(combinedData@nbdaData$task))-1;
		if(group) extraParam<-extraParam+length(levels(combinedData@nbdaData$group))-1;

		}else{
			
			sParam<-1;
			noSParam<-1;
			extraParam<-0;
			maxRate<-max(tadata@nbdaData$time2)+1000
			sampleSize<-sum(tadata@nbdaData$status);
			noTasks<-0;
			noGroups<-0;
		}
		
		#Set staring values if not specified by the user
		if(is.null(startValue)){
			 		startValue<-c(rep(0,noSParam),1,rep(0,length(asocialVar)+extraParam));
					startInd<-1;
		}else{
		
			startInd<-0;
			
		}

if((length(asocialVar)+extraParam)==0){
	null<-optimise(nullDiscreteTadaLikelihood,interval=c(0,maxRate),asocialVar=asocialVar,tadata=tadata, additive=additive,sParam=sParam,group=group,task=task, stepLength=stepLength);
	if(startInd==1) startValue[noSParam+1]<-null$minimum;
}else{
	null<-nlminb(start=startValue[-(1: noSParam)], nullDiscreteTadaLikelihood,lower=c(0,rep(-Inf,length(asocialVar)+extraParam)),asocialVar=asocialVar,tadata=tadata,additive=additive,sParam=sParam,group=group,task=task, stepLength=stepLength);
	if(startInd==1) startValue[-(1: noSParam)]<-null$par;
}


		
		fit1<-nlminb(start=startValue,discreteTadaLikelihood,lower=c(rep(0,noSParam+1),rep(-Inf,length(asocialVar)+extraParam)),sParam=sParam,asocialVar=asocialVar,tadata=tadata,task=task,group=group,additive=additive, stepLength=stepLength);

		#Perform LRT for social transmission
		loglik<-fit1$objective;
		nulllik<-null$objective;
		LRTsocTransTS<-2*(nulllik-loglik)
		LRTsocTransPV<-1-pchisq(LRTsocTransTS,df=noSParam);

		if(is.null(asocialVar)) asocialVar<-NaN;


		#Calculate aic and for model without social transmission
		k<-length(fit1$par);
		aic<-(2*k)+(2*loglik);
		aicc<-aic+(2*k*(k+1)/(sampleSize-k-1));

		if(is.null(null$par)){k<-1}else{k<-length(null$par)};
		aicNull<-(2*k)+(2*nulllik);
		aiccNull<-aicNull+(2*k*(k+1)/(sampleSize-k-1));

		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)|(aicc==Inf)){}else{
			if(aicc<aic) aicc<-NaN;
		}
		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aicNull)|is.nan(aiccNull)|(is.na(aiccNull))|(aiccNull==Inf)){}else{
			if(aiccNull<aicNull) aiccNull<-NaN;
		}
				
	#Extract names of variables
		if(is.nan(asocialVar[1])){
	
			varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling");			
			
		}else{
			varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling",unlist(dimnames(tadata@nbdaData)[2])[asocialVar+6]);
		}
		if(task) varNames<-c(varNames,paste("Task",levels(combinedData@nbdaData$task)[-1]));
		if(group) varNames<-c(varNames,paste("Group",levels(combinedData@nbdaData$group)[-1]));





        callNextMethod(.Object,optimisation=fit1,optimisationNull=null,additive=additive,sParam=sParam,bounded=bounded,loglik=loglik, aic=aic,aicc=aicc,nulllik=nulllik,aicNull=aicNull,aiccNull=aiccNull,LRTsocTransTS=LRTsocTransTS,LRTsocTransPV=LRTsocTransPV,asocialVar=asocialVar,varNames=varNames,task=task,group=group,...) 

       

       
    }
)


#Function for implementing the initialization
discreteTadaFit<-function(tadata,sParam=NULL,asocialVar=NULL,startValue=NULL,bounded=FALSE,task=FALSE,group=FALSE,additive=TRUE,stepLength=1){

	
	if(bounded==T) {
	
	cat("Bounded parameterisation not yet available for time of acquisition models");
	return("Bounded parameterisation not yet available for time of acquisition models");	
		
	}	
	
	new("discreteTadaFit",tadata=tadata,sParam=sParam,asocialVar=asocialVar,startValue=startValue,bounded=bounded,task=task,group=group,additive=additive,stepLength=stepLength)	
}


nullDiscreteTadaLikelihood<-function(l,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,stepLength=1){
	
		if(is.null(sParam)){noSParam<-1}else{noSParam<-length(levels(as.factor(sParam[sParam>0])))};
		newl<-c(rep(0,noSParam),l);
		return(discreteTadaLikelihood(newl,tadata,sParam=sParam,bounded=bounded,asocialVar=asocialVar,task=task,group=group, additive=additive,stepLength=stepLength));
}


#Gets likelihood for a multiplicative nbda model 
discreteTadaLikelihood<-function(l,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,stepLength=1){

#If the data is a character vector containing multiple oaData objects, make separate calls to addLikelihood and add together


if(is.character(tadata)){		

	
	#If sParam is NULL this means only one s parameter is required
	if(is.null(sParam)) sParam<-rep(1,length(tadata));
	#Calculate the number of different s parameters
	noSParam<-length(levels(as.factor(sParam[sParam>0])))

	#For the purposes of working out how many task and group levels there are only:
	combinedData<-combineTadaData(tadata,sParam=sParam);
	sampleSize<-sum(combinedData@nbdaData$status);

	noTasks<-length(levels(combinedData@nbdaData$task));
	noGroups<-length(levels(combinedData@nbdaData$group));
	
	sParamVect<-l[1:noSParam];
	rateScale<-l[(noSParam+1)];
	if(is.null(asocialVar)){
		asocVarVect<-NULL;
		asocLength<-0;
	}else{
		asocLength<-length(asocialVar);
		asocVarVect<-l[(noSParam+2):(noSParam+2+ asocLength-1)];
	}
	lengthTask<-0
	taskParam<-NULL
	if(task){
		lengthTask<-noTasks-1
		if(lengthTask>0){
			taskParam<-c(0,l[(noSParam+2+asocLength):(noSParam+2+asocLength+lengthTask)])
		}
	}
	lengthGroup<-0
	groupParam<-NULL
	if(group){
		lengthGroup<-noGroups-1
		if(lengthGroup>0){
			groupParam<-c(0,l[(noSParam+2+asocLength+lengthTask):(noSParam+2+asocLength+lengthTask+lengthGroup-1)])
		}
	}
			
	totalLikelihood<-0;
	

	


			
	for(i in 1:length(tadata)){

	
		if(length(stepLength)==1) indStepLength <-rep(stepLength,eval(as.name(tadata[i]))@endTime);

		if(dim(as.matrix(stepLength))[2]==1) {indStepLength<-as.matrix(stepLength)[,1]} else{indStepLength<-as.matrix(stepLength)[,i]}

		if (sParam[i]==0){
			#The st is contstrained to zero for this diffusion
			
			newl<-c(0,rateScale,asocVarVect, taskParam[which(eval(as.name(tadata[i]))@oadata@taskid==levels(combinedData@nbdaData$task))], groupParam[which(eval(as.name(tadata[i]))@oadata@groupid==levels(combinedData@nbdaData$group))])
			
			totalLikelihood<- totalLikelihood + discreteTadaLikelihood(l=newl,tadata=eval(as.name(tadata[i])),asocialVar=asocialVar, task=task, group=group, additive=additive, stepLength=stepLength);
				
		}else{
		#Build the parameter vector for the diffusion in question by selecting the correct st parameter and adding the others on the end
			newl<-c(l[sParam[i]],rateScale,asocVarVect, taskParam[which(eval(as.name(tadata[i]))@oadata@taskid==levels(combinedData@nbdaData$task))], groupParam[which(eval(as.name(tadata[i]))@oadata@groupid==levels(combinedData@nbdaData$group))])
			
			totalLikelihood<- totalLikelihood + discreteTadaLikelihood(l=newl,tadata=eval(as.name(tadata[i])),asocialVar=asocialVar, task=task, group=group, additive=additive, stepLength= indStepLength);
		}
				
	}
			
		return(totalLikelihood);
					
}else{

	
	#Cut down to the asocial variables specified by asocialVar
	asoc<-as.matrix(tadata@oadata@asoc[,asocialVar]);

	#If there are group and/or task effects, store them and take them off the end of the parameter vector
	if(group==TRUE) {
		groupEffect<-l[length(l)];
		l<-l[-length(l)];
	}else{groupEffect<-0}
	if(task==TRUE) {
		taskEffect<-l[length(l)];
		l<-l[-length(l)];
	}else{taskEffect<-0}

	
	#Check if there are any asocial variables, set rate to one if not
	if(is.null(asocialVar)){
		linearPred<-rep(0,dim(tadata@oadata@assMatrix)[1]);
	}else{
		#Multiply asocial variable by coefficients and sum for each individual
		linearPred<-apply(as.matrix(asoc*t(matrix(l[-(1:2)],ncol=dim(asoc)[1],nrow=length(asocialVar)))),1,sum);
	}
	#Take exponentials to get rate of asocial learning
	asocialRate<-exp(linearPred+groupEffect + taskEffect);

	timeSolve<-rep(tadata@endTime+1,length=dim(tadata@oadata@assMatrix)[1])

	for(i in 1:length(tadata@oadata@orderAcq)){
		
		timeSolve[tadata@oadata@orderAcq[i]]<-tadata@timeAcq[i]
		
	}


	if(length(stepLength)==1) stepLength<-rep(stepLength,tadata@endTime)

	logLikelihood<-0
	statusTracker<-rep(0,dim(tadata@oadata@assMatrix)[1])
	i<-0
	statusTracker[which(timeSolve==i)]<-1;	

	#Cycle through timeSteps, updating log likelihood and status of individuals
	for(i in 1:tadata@endTime){
		socialRate<-apply((tadata@oadata@assMatrix*statusTracker),2,sum)*l[1];
		if(additive){
			stepRate<-((1/l[2])*(socialRate+asocialRate)*stepLength[i]);
		}else{
			stepRate<-((1/l[2])*((socialRate+1)*asocialRate)*stepLength[i]);
		}
		logLikelihood<-logLikelihood +sum(((1-1*(timeSolve==i))*(stepRate)-log(1-exp(-stepRate))*(timeSolve==i))*(1-statusTracker));
		statusTracker[which(timeSolve==i)]<-1;
	}

	return(logLikelihood)
	
}
}


setMethod("anova",
    signature(object = "discreteTadaFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.null(object@optimisation$par[1])){
    
    		table<-data.frame(Df=c(1,0),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));

    
    	}else{   

   		   table<-data.frame(Df=c(length(object@optimisation$par),length(object@optimisation$par)-noSParam),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));
   		 
   		}

		if(object@additive){atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine additively\n", class = c("anova", "data.frame"))}else{
atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine multiplicatively\n", class = c("anova", "data.frame"))			
		}
		
		return(atable);

    }
)


setMethod("summary",
    signature(object = "discreteTadaFit"),
    function (object, ...) 
    {

		if(object@additive==T)cat("Summary of Additive Social Transmission Model\nDiscrete Time of Acquisition Data\n");
		if(object@additive==F)cat("Summary of Multiplicative Social Transmission Model\nDiscrete Time of Acquisition Data\n");


		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))



  			if(object@bounded==F) {
  				cat("Unbounded parameterisation.");
	    	   	sumtable<-data.frame(Estimate=c(object@optimisation$par),Bounded=c(object@optimisation$par[1: noSParam]/(1+object@optimisation$par[1: noSParam]),rep("",length(object@optimisation$par)-noSParam)),row.names=object@varNames);   
    	
    		}		
    	

		
		
		cat("\n\nCoefficients\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
       
    }
)


#Gets likelihood for a multiplicative nbda model 
gammaTadaLikelihood<-function(l,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE){




	if(is.null(sParam)){
		noSParam<-1
		
	}else{
		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
	}

#Found that paramterising by 1/rate works better for optimisation
rate<-l[noSParam+1]<-1/l[noSParam+1];
shape<-l[noSParam+2]

	#Cut down to the asocial variables specified by asocialVar
	asoc<-as.matrix(tadata@nbdaData[,asocialVar+6]);

	if(task){
		
		taskNames<-paste("task",levels(tadata@nbdaData$task),sep="");
		taskMatrix<-matrix(nrow=dim(tadata@nbdaData)[1],ncol=length(taskNames)-1,dimnames=list(NULL,taskNames[-1]))
		for(i in 2:length(taskNames)){
			
			taskMatrix[,i-1]<-(tadata@nbdaData$task==levels(tadata@nbdaData$task)[i])*1
			
		}
		asoc<-cbind(asoc,taskMatrix);
		
	}
	
	if(group){
		
		groupNames<-paste("group",levels(tadata@nbdaData$group),sep="");
		groupMatrix<-matrix(nrow=dim(tadata@nbdaData)[1],ncol=length(groupNames)-1,dimnames=list(NULL,groupNames[-1]))
		for(i in 2:length(groupNames)){
			
			groupMatrix[,i-1]<-(tadata@nbdaData$group==levels(tadata@nbdaData$group)[i])*1
			
		}
		asoc<-cbind(asoc,groupMatrix);		
	}


	#Calulate the appropriate value of s for each line of data
	if(is.null(tadata@nbdaData$sParamIndex)){
	
		sVect<-rep(l[1],dim(tadata@nbdaData)[1])
	
	}else{
		
		stemp<-1*(tadata@nbdaData$sParamIndex>0)
		stemp[which(tadata@nbdaData$sParamIndex>0)]<-l[tadata@nbdaData$sParamIndex[tadata@nbdaData$sParamIndex>0]]
		sVect<-stemp;
		
	}

	
	#Check if there are any asocial variables, set rate to one if not
	if(dim(asoc)[2]==0){
		lpRateAsoc<-rep(0,dim(tadata@nbdaData)[1]);
	}else{
		lpRateAsoc<-apply(t(matrix(rep(l[-(1:(noSParam+2))],dim(tadata@nbdaData)[1]),nrow=dim(asoc)[2]))*asoc,1,sum);
	}

	lpSocTrans<-sVect*tadata@nbdaData[6];

cHazTstart<--pgamma(tadata@nbdaData$time1,shape=shape,rate=rate, lower = FALSE, log = TRUE)
cHazTend<--pgamma(tadata@nbdaData$time2,shape=shape,rate=rate, lower = FALSE, log = TRUE)
HazTend<-dgamma(tadata@nbdaData$time2,shape=shape,rate=rate)/(pgamma(tadata@nbdaData$time2,shape=shape,rate=rate, lower = FALSE))
		
	if(bounded==FALSE){

		if(additive==TRUE){
	
			loglik1<--(lpSocTrans+exp(lpRateAsoc))*(cHazTstart-cHazTend)
			loglik2<--tadata@nbdaData$status*(log(lpSocTrans+exp(lpRateAsoc))+log(HazTend))

			likelihood<-loglik1+loglik2

			
		}else{

			loglik1<--(lpSocTrans+1)*exp(lpRateAsoc)*(cHazTstart-cHazTend)
			loglik2<--tadata@nbdaData$status*(log(lpSocTrans+1)+lpRateAsoc+log(HazTend))

			likelihood <-loglik1+loglik2


		}
	
	}

	if(bounded==TRUE){

		return("Bounded parameterisation not yet available for time of acquisition models");

	}


	return(sum(likelihood));

}

nullGammaTadaLikelihood<-function(l,tadata,sParam=NULL,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE){
	
	if(is.null(sParam)){
		noSParam<-1
		
	}else{
		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
	}

	l<-c(rep(0,noSParam),l);
	gammaTadaLikelihood(l,tadata,bounded=FALSE,asocialVar=asocialVar,task=task,group=group,additive=additive,sParam=sParam);

}



#Define class of object for the fitted nbda model
setClass("gammaTadaFit",representation(optimisation="list",optimisationNull="list",additive="logical",sParam="numeric",bounded="logical",loglik="numeric",aic="numeric", aicc="numeric",nulllik="numeric",aicNull="numeric",aiccNull="numeric",LRTsocTransTS="numeric",LRTsocTransPV="numeric",asocialVar="numeric",varNames="character",task="logical",group="logical")); 


#Method for initializing object- including model fitting
setMethod("initialize",
    signature(.Object = "gammaTadaFit"),
    function (.Object, tadata,additive,sParam,startValue,asocialVar,bounded,task,group,optim,lower,...) 
    {

		#If the tadata if a character vector containing multiple oaData objects, combine into a single object first
		#Also set the default sParam if nothing is specified
		if(is.character(tadata)){		
			if(is.null(sParam)) sParam<-rep(1,length(tadata));			tadata<-combineTadaData(tadata,sParam=sParam);

		}else{
			sParam<-1;				
		}

		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		sampleSize<-sum(tadata@nbdaData$status);

		extraParam<-0;

		if(task) extraParam<-extraParam+length(levels(tadata@nbdaData$task))-1;
		if(group) extraParam<-extraParam+length(levels(tadata@nbdaData$group))-1;
		
		#Set staring values if not specified by the user
		if(is.null(startValue)){
			 		startValue<-c(rep(0,noSParam),mean(tadata@timeAcq),1,rep(0,length(asocialVar)+extraParam));
					startInd<-1;
		}else{
		
			startInd<-0;
			
		}

		#Set lower bounds if not specified by the user
		if(is.null(lower)){
			lower=c(rep(0,noSParam+2),rep(-Inf,length(asocialVar)+extraParam))
		}


if(optim=="optim"){

null<-optim(par=startValue[-(1: noSParam)],nullGammaTadaLikelihood,method="L-BFGS-B",lower=lower[-(1:noSParam)],asocialVar=asocialVar,tadata=tadata,task=task,group=group,additive=additive,sParam=sParam);
if(startInd==1) startValue[-(1: noSParam)]<-null$par;


		
		fit1<-optim(par=startValue,gammaTadaLikelihood,method="L-BFGS-B",lower=lower,sParam=sParam,asocialVar=asocialVar,tadata=tadata,bounded=bounded,task=task,group=group,additive=additive);



		#For LRT for social transmission
		loglik<-fit1$value;
		nulllik<-null$value;


		#Get no. parameters
		k<-length(fit1$par);


}else{
null<-nlminb(start=startValue[-(1: noSParam)],nullGammaTadaLikelihood,lower=lower[-(1:noSParam)],asocialVar=asocialVar,tadata=tadata,task=task,group=group,additive=additive,sParam=sParam);
if(startInd==1) startValue[-(1: noSParam)]<-null$par;


		
		fit1<-nlminb(start=startValue,gammaTadaLikelihood,lower=lower,sParam=sParam,asocialVar=asocialVar,tadata=tadata,bounded=bounded,task=task,group=group,additive=additive);


		#For LRT for social transmission
		loglik<-fit1$objective;
		nulllik<-null$objective;

		#Get no. parameters
		k<-length(fit1$par);
}

		#Perform LRT for social transmission
		LRTsocTransTS<-2*(nulllik-loglik)
		LRTsocTransPV<-1-pchisq(LRTsocTransTS,df=noSParam);

		if(is.null(asocialVar)) asocialVar<-NaN;


		#Calculate aic and for model without social transmission

		aic<-(2*k)+(2*loglik);
		aicc<-aic+(2*k*(k+1)/(sampleSize-k-1));

		k<-length(null$par);
		aicNull<-(2*k)+(2*nulllik);
		aiccNull<-aicNull+(2*k*(k+1)/(sampleSize-k-1));

		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)){}else{
			if(aicc<aic) aicc<-NaN;
		}
		if(is.nan(aicNull)|is.nan(aiccNull)){}else{
			if(aiccNull<aicNull) aiccNull<-NaN;
		}
		
		#Extract names of variables
		if(is.nan(asocialVar[1])){
	
			varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling","Shape");			
			
		}else{
			varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling","Shape",unlist(dimnames(tadata@nbdaData)[2])[asocialVar+6]);
		}
		if(task) varNames<-c(varNames,paste("Task",levels(tadata@nbdaData$task)[-1]));
		if(group) varNames<-c(varNames,paste("Group",levels(tadata@nbdaData$group)[-1]));





        callNextMethod(.Object,optimisation=fit1,optimisationNull=null,additive=additive,sParam=sParam,bounded=bounded,loglik=loglik, aic=aic,aicc=aicc,nulllik=nulllik,aicNull=aicNull,aiccNull=aiccNull,LRTsocTransTS=LRTsocTransTS,LRTsocTransPV=LRTsocTransPV,asocialVar=asocialVar,varNames=varNames,task=task,group=group,...) 

       

       
    }
)


#Function for implementing the initialization
gammaTadaFit<-function(tadata,sParam=NULL,asocialVar=NULL,startValue=NULL,bounded=FALSE,task=FALSE,group=FALSE,additive=TRUE, optim="nlminb",lower=NULL){

	
	if(bounded==T) {
	
	cat("Bounded parameterisation not yet available for time of acquisition models");
	return("Bounded parameterisation not yet available for time of acquisition models");	
		
	}	
	
	new("gammaTadaFit",tadata=tadata,sParam=sParam,asocialVar=asocialVar,startValue=startValue,bounded=bounded,task=task,group=group,additive=additive,optim=optim,lower=lower)	
}



setMethod("anova",
    signature(object = "gammaTadaFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.null(object@optimisation$par[1])){
    
    		table<-data.frame(Df=c(1,0),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));

    
    	}else{   

   		   table<-data.frame(Df=c(length(object@optimisation$par),length(object@optimisation$par)-noSParam),LogLik=c(object@loglik,object@nulllik),AIC=c(object@aic,object@aicNull),AICc=c(object@aicc,object@aiccNull),LR=c(object@LRTsocTransTS,NA),p=c(object@LRTsocTransPV,NA),row.names=c("With Social Transmission","Without Social Transmission"));
   		 
   		}

		if(object@additive){atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine additively\n", class = c("anova", "data.frame"))}else{
atable<-structure(table, heading = "Likelihood Ratio Test for Social Transmission:\n\nNull model includes all other specified variables\nSocial transmission and asocial learning assumed to combine multiplicatively\n", class = c("anova", "data.frame"))			
		}
		
		return(atable);

    }
)


setMethod("summary",
    signature(object = "gammaTadaFit"),
    function (object, ...) 
    {

		if(object@additive==T)cat("Summary of Additive Social Transmission Model\nTime of Acquisition Data\n");
		if(object@additive==F)cat("Summary of Multiplicative Social Transmission Model\nTime of Acquisition Data\n");


		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

# 		if(length(object@optimisation$par)<3){


  			if(object@bounded==FALSE) {
  				cat("Unbounded parameterisation.");
	    	   	sumtable<-data.frame(Estimate=c(object@optimisation$par),Bounded=c(object@optimisation$par[1: noSParam]/(1+object@optimisation$par[1: noSParam]),rep("",length(object@optimisation$par)-noSParam)),row.names=object@varNames);   
    	
    		}		
    	
    	
    	
 #   	}else{		

#    	}
		
		
		cat("\n\nCoefficients\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
       
    }
)

plotGammaBaseRate<-function(model=NULL,shape=NULL,rate=NULL,xlim,resolution=1000,output="plot",ylim=NULL){
	if(is.null(model)){}else{
		shape=model@optimisation$par[which(model@varNames=="Shape")];
		rate=1/model@optimisation$par[which(model@varNames=="1/Rate scaling")];
	}
	x<-seq(xlim[1],xlim[2],length=resolution);
	y<-dgamma(x,shape,rate)/(1-pgamma(x,shape,rate))
	if(output=="matrix"){return(matrix(data=c(x,y),nrow=length(x)))}else{
		plot(x,y,type="l",xlim=xlim, ylim=ylim)
	}
}



#A function that fits all models given a set of variables/groups/tasks and social learning contrasts and returns  a table ordered by AIC or AICc
aicTable<-function(data,asocialVar=NULL,group=F,task=F,sParamMatrix=NULL,pure=F,aic="aicc",stratify="both",interval=NULL){

if(class(data)=="oaData"){return(aicTableOA(data,asocialVar= asocialVar,group= group,task= task,sParamMatrix= sParamMatrix,pure= pure,aic= aic,stratify=stratify,interval=interval))}
if(class(data)=="character"){
	if(class(eval(as.name(data[1])))=="oaData") {return(aicTableOA(data,asocialVar= asocialVar,group= group,task= task,sParamMatrix= sParamMatrix,pure= pure,aic= aic,stratify=stratify))}
}
tadata<-data;

#Set sParam if not defined
if(is.null(sParamMatrix)) {
	if(is.character(tadata)){sParamMatrix<-rbind(rep(1,length(tadata)))}else{sParamMatrix<-1}
}

#Calculate all possible combinations of ILVs
if(is.null(asocialVar)){asocialVarMatrix=0}else{
tempList<-NULL
for(i in 1:length(asocialVar)){
	if(is.null(tempList)){tempList<-list(c(0,1))}else{
		tempList<-c(tempList,list(c(0,1)));
	}
}
asocialVarMatrix<-expand.grid(tempList)
}

if(group){groupVect<-c(F,T)}else{groupVect<-F}
if(task){taskVect<-c(F,T)}else{taskVect<-F}

aiccTable<-aicTable<-NULL;

#Run through all combinations of models
for(sParamIndex in 1:dim(sParamMatrix)[1]){
	sParam<-sParamMatrix[sParamIndex,]
	for(baseline in c("constant","non-constant")){
		for(group in groupVect){
			for(task in taskVect){
				for(ilvComb in 1:dim(as.matrix(asocialVarMatrix))[1]){
					#Extract the appropriate ILV vector
					if(is.null(asocialVar)){ilv<-NULL}else{
						tempILV<-as.numeric(asocialVarMatrix[ilvComb,]*asocialVar);
						ilv<-tempILV[tempILV>0];
					}
					if (length(ilv)==0) ilv<-NULL;
					if(is.null(ilv)&!group&!task){
						model<-NULL;
						if(baseline=="constant"){model<-tadaFit(tadata=tadata,asocialVar=ilv,group=group,task=task,sParam=sParam)}else
						{model<-gammaTadaFit(tadata=tadata,asocialVar=ilv,group=group,task=task,sParam=sParam)}
					if(is.null(ilv)) ilv<-0;	
						if(is.null(aiccTable)){
							aiccTable<-c(baseline,"NA",group,task,ilv,sParamIndex,"social",model@aicc)
						}else{
							aiccTable<-rbind(aiccTable,c(baseline,"NA",group,task,ilv,sParamIndex,"social",model@aicc))
						}
						aiccTable<-rbind(aiccTable,c(baseline,"NA",group,task,ilv,sParamIndex,"asocial",model@aiccNull))
						
												
						if(is.null(aicTable)){
							aicTable<-c(baseline,"NA",group,task,ilv,sParamIndex,"social",model@aic)
						}else{
							aicTable<-rbind(aicTable,c(baseline,"NA",group,task,ilv,sParamIndex,"social",model@aic))
						}
						if(sParamIndex==1){aicTable<-rbind(aicTable,c(baseline,"NA",group,task,ilv,sParamIndex,"asocial",model@aicNull))}
										
					
					}else{		
					
						for(additive in c(TRUE,FALSE)){
							if(is.null(ilv)){}else{if(ilv==0) ilv<-NULL};
							model<-NULL;
							if(baseline=="constant"){model<-tadaFit(tadata=tadata,asocialVar=ilv,group=group,task=task,additive=additive,sParam=sParam)}else
							{model<-gammaTadaFit(tadata=tadata,asocialVar=ilv,group=group,task=task,additive=additive,sParam=sParam)}
										
						if(is.null(ilv)) ilv<-0;
						
						if(is.null(aiccTable)){
							aiccTable<-c(baseline,additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aicc)
						}else{
							aiccTable<-rbind(aiccTable,c(baseline,additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aicc))
						}
						if(additive&(sParamIndex==1)){aiccTable<-rbind(aiccTable,c(baseline,"NA",group,task,paste(ilv,collapse=" "),sParamIndex,"asocial",model@aiccNull))}
						
												
						if(is.null(aicTable)){
							aicTable<-c(baseline,additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aic)
						}else{
							aicTable<-rbind(aicTable,c(baseline,additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aic))
						}
						if(additive&(sParamIndex==1)){aicTable<-rbind(aicTable,c(baseline,"NA",group,task,paste(ilv,collapse=" "),sParamIndex,"asocial",model@aicNull))}
	
						}
					}		
				}
			}
		}
	}


	if(pure){
	
		model<-pureSocialTada(tadata=tadata,sParam=sParam);
		aiccTable<-rbind(aiccTable,c("constant","NA",F,F,0,sParamIndex,"pure social",model@aicc))
	}
}

if(aic=="aicc"){
	dimnames(aiccTable)<-list(1:dim(aiccTable)[1],c("Baseline","Additive?","Group?","Task?","ILVs","sParamIndex","Social?","AICc"))
	aiccTable<-aiccTable[order(aiccTable[,8]),];
	deltaAICc<-round(as.numeric(aiccTable[,8])-as.numeric(aiccTable[1,8]),2);
	aiccTable<-cbind(aiccTable,deltaAICc);
	return(aiccTable);
	
}else{
	dimnames(aicTable)<-list(1:dim(aicTable)[1],c("Baseline","Additive?","Group?","Task?","ILVs","sParamIndex","Social?","AIC"))
	aicTable<-aicTable[order(aicTable[,8]),]	
	deltaAIC<-round(as.numeric(aicTable[,8])-as.numeric(aicTable[1,8]),2);
	aicTable<-cbind(aicTable,deltaAIC);
	return(aicTable)

}

}


upDateTimes<-function(idname=NULL,orderAcq,timeAcq,endTime,updateTimes,assMatrix,asoc,demons,taskid="1",groupid="1",diffvect=NA, groupvect=NA, taskvect=NA, return){

	#statusTracker records the individuals who have acquired the behaviour
	#statusTracker2 records the individuals capable of transmitting to others

	timeVect<-cbind(as.factor(rep(c("acq","upd"),each=length(timeAcq))),c(timeAcq,updateTimes),rep(orderAcq,2))
	timeVect<-timeVect[order(timeVect[,2]),]

   	#Create default names vector if none is provided, if it is, convert to a factor
		if(is.null(idname)) idname<-(1:dim(assMatrix)[1]);
		

		time1<-TAtime1<-TAtime2<-vector();
		time2<-vector();
		status<-vector();
		id<-vector();
		identity<-vector();
		stMetric<-vector();
		totalMetric<-vector();
		learnMetric<-vector();
		totalAsoc<-vector();
		group<-task<-vector();
			
		#Make sure the asoc is in matrix form
		asoc<-as.matrix(asoc);
	
		#Calculate the asocial learning variables for the learning individual at each step
		learnAsoc<-matrix(nrow=dim(asoc)[2],ncol=length(orderAcq))
		for(i in 1:length(orderAcq)) learnAsoc[,i]<-asoc[orderAcq[i]]

		#Define functions for calculating total st Metric
		newFunc<-function(x) x*(1-statusTracker)
		newFunc2<-function(x) x*statusTracker2

		
		#Generate a variable for tracking the status of individuals in the group in question
		#Generate a variable for tracking the status of individuals in the group in question
		if(is.null(demons)){
			statusTracker<-rep(0,dim(assMatrix)[1]);
			statusTracker2<-rep(0,dim(assMatrix)[1]);
		}else{
			statusTracker<-statusTracker2<-demons;
		}


		#Loop through acquisition and update events
		for(i in 1:(dim(timeVect)[1])){
	
			#Loop through each individual
			for(j in 1:dim(assMatrix)[1]){

				#Only naive individuals considered
				if(statusTracker[j]==0){

					#Record variables for Cox model
					time1<-c(time1,i-1);
					time2<-c(time2,i);
					if(i>1){TAtime1<-c(TAtime1 ,timeVect[i-1,2])}else{TAtime1<-c(TAtime1 ,0)}
					TAtime2<-c(TAtime2, timeVect[i,2]);
					identity<-c(identity,j);
					stMetric<-c(stMetric,sum(assMatrix[j,]*statusTracker2));
					id<-c(id,idname[j]);
					if(!is.na(groupvect[1]))group<-c(group,groupvect[j])
					if(!is.na(taskvect[1]))task<-c(task,taskvect[j])

					#Record status as one if individual acquires trait, zero otherwise for Cox model
					if(j==timeVect[i,3]){
						if(timeVect[i,1]==1) status<-c(status,1)

			#Record the social transmission metric and asocial learning variables for the learning individual (ML model)
						learnMetric<-c(learnMetric,sum(assMatrix[j,]*statusTracker2));

					}else{
						status<-c(status,0);

					}
				}
			}
			
			#Calculate total st metric over all individuals in the group for each acquisition event (ML model)
			if(timeVect[i,1]==1){
				newMatrix<-apply(assMatrix,2,newFunc)
				if(is.na(diffvect[1])){
					totalMetric<-c(totalMetric,sum(apply(newMatrix,1,newFunc2)));
				}else{
					 totalMetric<-rbind(totalMetric,as.vector(tapply(apply(apply(newMatrix,1,newFunc2),2,sum),diffvect,sum)))
				}				
			}

		
			#Set statusTracker to one if individual acquires trait
			if(timeVect[i,1]==1) statusTracker[timeVect[i,3]]<-1;
			#Set statusTracker2 to one if update time is reached
			if(timeVect[i,1]==2) statusTracker2[timeVect[i,3]]<-1;
		}


#And for the ML models

indVar<-matrix(asoc[identity,],ncol=ncol(asoc),dimnames=dimnames(asoc));
if(is.na(groupvect[1])) group<-rep(groupid,length(time1));
if(is.na(taskvect[1])) task<-rep(taskid,length(time1));
coxdata<-data.frame(id=as.factor(id),time1,time2,status,identity,stMetric,indVar,group,task);

			#Loop through each individual
			for(j in 1:dim(assMatrix)[1]){

				#Only naive individuals considered
				if(statusTracker[j]==0){

					#Record variables for TADA model
					TAtime1<-c(TAtime1 ,timeVect[i,2])
					TAtime2<-c(TAtime2, endTime);
					identity<-c(identity,j);
					stMetric<-c(stMetric,sum(assMatrix[j,]*statusTracker2));
					id<-c(id,idname[j]);
					if(!is.na(groupvect[1]))group<-c(group,groupvect[j])
					if(!is.na(taskvect[1]))task<-c(task,taskvect[j])
					status<-c(status,0);

				}
			}
		

#Record individual variables for each line of data in the Cox model and TADA

indVar<-matrix(asoc[identity,],ncol=ncol(asoc),dimnames=dimnames(asoc));
if(is.na(groupvect[1])) group<-rep(groupid,length(TAtime1));
if(is.na(taskvect[1])) task<-rep(taskid,length(TAtime1));
nbdadata<-data.frame(id=as.factor(id),time1=TAtime1,time2=TAtime2,status,identity,stMetric,indVar,group,task,time=(TAtime2-TAtime1));

#And for the ML models
if(is.na(groupvect[1])) {group<-rep(groupid,length(orderAcq))}else{group<-rep("NA",length(orderAcq))};
if(is.na(taskvect[1])) {task<-rep(taskid,length(orderAcq))}else{task<-rep("NA",length(orderAcq))};
mldata<-data.frame(group,task,orderAcq,learnMetric,totalMetric,asoc[orderAcq,]);

if(return=="nbdadata"){return(nbdadata)}
if(return=="coxdata"){return(coxdata)}
if(return=="mldata"){return(mldata)}

}


#Define class of object for the pure social transmission model
setClass("pureSocialTada",representation(optimisation="list",sParam="numeric",loglik="numeric",aic="numeric", aicc="numeric",varNames="character")); 


#Method for initializing object- including model fitting
setMethod("initialize",
    signature(.Object = "pureSocialTada"),
    function (.Object, tadata,sParam,startValue,...) 
    {

		#If the tadata is a character vector containing multiple oaData objects, combine into a single object first
		#Also set the default sParam if nothing is specified
		if(is.character(tadata)){		
			if(is.null(sParam)) sParam<-rep(1,length(tadata));			tadata<-combineTadaData(tadata,sParam=sParam);

		}else{
			sParam<-1;				
		}

		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		sampleSize<-sum(tadata@nbdaData$status);
		
		#Set staring values if not specified by the user
		if(is.null(startValue)){
			 		startValue<-c(rep(0,noSParam));
		}

		if(noSParam==1){
			fit1<-optimise(pureTadaLikelihood,interval=c(0,(max(tadata@nbdaData$stMetric)*max(tadata@nbdaData$time2))),sParam=sParam,tadata=tadata);			
		}else{
			fit1<-nlminb(start=startValue,pureTadaLikelihood,lower=rep(0,noSParam),sParam=sParam,tadata=tadata);
		}

		loglik<-fit1$objective;
		#Calculate aic and for model without social transmission
		k<-noSParam;
		aic<-(2*k)+(2*loglik);
		aicc<-aic+(2*k*(k+1)/(sampleSize-k-1));

		#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)){}else{
			if(aicc<aic) aicc<-NaN;
		}
		
		#Extract names of variables
	
		varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling");

        callNextMethod(.Object,optimisation=fit1,sParam=sParam,loglik=loglik, aic=aic,aicc=aicc,varNames=varNames,...) 

       

       
    }
)

#Function for implementing the initialization
pureSocialTada<-function(tadata,sParam=NULL,startValue=NULL){


	
	new("pureSocialTada",tadata=tadata,sParam=sParam,startValue=startValue)	
}



setMethod("anova",
    signature(object = "pureSocialTada"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.null(object@optimisation$par[1])){
    		table<-data.frame(Df=1,LogLik=object@loglik,AIC=object@aic,AICc=object@aicc,row.names="Pure Social Transmission");
    	}else{   
			table<-data.frame(Df=length(object@optimisation$par),LogLik=object@loglik,AIC=object@aic,AICc=object@aicc,row.names="Pure Social Transmission");
   		}

		atable<-structure(table, heading = "Pure social transmission model\n\nAll acquisition through social network\n", class = c("anova", "data.frame"))			

		
		return(atable);

    }
)


setMethod("summary",
    signature(object = "pureSocialTada"),
    function (object, ...) 
    {

		cat("Summary of Pure Social Transmission Model\nTime of Acquisition Data\n");

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

  		cat("Social transmission 1 constrained to 1.");
		if(is.null(object@optimisation$par)){
		    sumtable<-data.frame(Estimate=c(1,object@optimisation$minimum),row.names=object@varNames);
		}else{
		    sumtable<-data.frame(Estimate=c(1,object@optimisation$par),row.names=object@varNames);   
    
    	}	
  
		
		cat("\n\nCoefficients\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
       
    }
)

#Gets likelihood for a multiplicative nbda model 
pureTadaLikelihood<-function(l,tadata,sParam=NULL){

	#Constrain first social learning parameter to 0
	l<-c(1,l)


	if(is.null(sParam)){
		noSParam<-1
		
	}else{
		#Calculate the number of social transmission parameters to be fitted
		noSParam<-length(levels(as.factor(sParam[sParam>0])))
	}

#Found that paramterising by 1/rate works better for optimisation
l[noSParam+1]<-1/l[noSParam+1];

	#Calulate the appropriate value of s for each line of data
	if(is.null(tadata@nbdaData$sParamIndex)){
	
		sVect<-rep(l[1],dim(tadata@nbdaData)[1])
	
	}else{
		
		stemp<-1*(tadata@nbdaData$sParamIndex>0)
		stemp[which(tadata@nbdaData$sParamIndex>0)]<-l[tadata@nbdaData$sParamIndex[tadata@nbdaData$sParamIndex>0]]
		sVect<-stemp;
		
	}

	lpSocTrans<-sVect*tadata@nbdaData[6];
		

			likelihood<-tadata@nbdaData$status*(log(l[noSParam+1])+log(lpSocTrans)-(l[noSParam+1]*(lpSocTrans))*tadata@nbdaData$time)+(1-tadata@nbdaData$status)*((-l[noSParam+1]*(lpSocTrans))*tadata@nbdaData$time);




	return(-sum(likelihood));

}



#Gets likelihood for a multiplicative nbda model 
PSLdiscreteTadaLikelihood<-function(l,tadata,sParam=NULL,stepLength=1){

#Constrain s1 to 1
l<-c(1,l)

#If the data is a character vector containing multiple oaData objects, make separate calls to addLikelihood and add together


if(is.character(tadata)){		

	
	#If sParam is NULL this means only one s parameter is required
	if(is.null(sParam)) sParam<-rep(1,length(tadata));
	#Calculate the number of different s parameters
	noSParam<-length(levels(as.factor(sParam[sParam>0])))

	#For the purposes of working out how many task and group levels there are only:
	combinedData<-combineTadaData(tadata,sParam=sParam);
	sampleSize<-sum(combinedData@nbdaData$status);
	
	sParamVect<-l[1:noSParam];
	rateScale<-l[(noSParam+1)];
		asocVarVect<-NULL;
		asocLength<-0;
		
	totalLikelihood<-0;
			
	for(i in 1:length(tadata)){
	
		if(length(stepLength)==1) stepLength<-rep(stepLength,eval(as.name(tadata[i]))@endTime);

		if(dim(as.matrix(stepLength))[2]==1) {indStepLength<-as.matrix(stepLength)[,1]} else{indStepLength<-as.matrix(stepLength)[,i]}

		#Build the parameter vector for the diffusion in question by selecting the correct st parameter and adding the others on the end
			newl<-c(l[sParam[i]],rateScale)
			
			totalLikelihood<- totalLikelihood + PSLdiscreteTadaLikelihood(l=newl,tadata=eval(as.name(tadata[i])),stepLength=stepLength);

				
	}
			
		return(totalLikelihood);
					
}else{

	timeSolve<-rep(tadata@endTime+1,length=dim(tadata@oadata@assMatrix)[1])

	for(i in 1:length(tadata@oadata@orderAcq)){
		timeSolve[tadata@oadata@orderAcq[i]]<-tadata@timeAcq[i]
	}

	if(length(stepLength)==1) stepLength<-rep(stepLength,tadata@endTime)

	logLikelihood<-0
	statusTracker<-tadata@oadata@demons
	i<-0
	statusTracker[which(timeSolve==i)]<-1;	

	#Cycle through timeSteps, updating log likelihood and status of individuals
	for(i in 1:tadata@endTime){
		socialRate<-apply((tadata@oadata@assMatrix*statusTracker),2,sum)*l[1];
		stepRate<-((1/l[2])*(socialRate)*stepLength[i]);

		loglik<-((1-1*(timeSolve==i))*(stepRate)-log(1-exp(-stepRate))*(timeSolve==i))
		loglik[stepRate==0]<-0
		logLikelihood<-logLikelihood +sum(loglik*(1-statusTracker));
		statusTracker[which(timeSolve==i)]<-1;
	}

	return(logLikelihood)
	
}
}


#Define class of object for the fitted discrete tada model
setClass("PSLdiscreteTadaFit",representation(optimisation="list",sParam="numeric",loglik="numeric",aic="numeric", aicc="numeric",varNames="character")); 


#Method for initializing object- including model fitting
setMethod("initialize",
    signature(.Object = "PSLdiscreteTadaFit"),
    function (.Object, tadata,sParam,startValue,stepLength,...) 
    {

		if(is.character(tadata)){		

	
		#If sParam is NULL this means only one s parameter is required
		if(is.null(sParam)) sParam<-rep(1,length(tadata));
		#Calculate the number of different s parameters
		noSParam<-length(levels(as.factor(sParam[sParam>0])))

		#For the purposes of working out how many task and group levels there are only:
		combinedData<-combineTadaData(tadata,sParam=sParam);
		sampleSize<-sum(combinedData@nbdaData$status);
		maxRate<-max(combinedData@nbdaData$time2)*max(combinedData@nbdaData$stMetric)

		}else{
			combinedData<-tadata;
			sParam<-1;
			noSParam<-1;
			maxRate<-max(combinedData@nbdaData$time2)*max(combinedData@nbdaData$stMetric)
			sampleSize<-sum(tadata@nbdaData$status);

		}
		
		#Set staring values if not specified by the user
		if(is.null(startValue)){
			 		startValue<-c(rep(0,noSParam-1),1);
		}

	if(noSParam==1){
		fit1<-optimise(PSLdiscreteTadaLikelihood,interval=c(0,maxRate),tadata=tadata, sParam=sParam, stepLength=stepLength);
	}else{
		fit1<-nlminb(start=startValue, PSLdiscreteTadaLikelihood,lower=rep(0,noSParam-1),tadata=tadata,sParam=sParam, stepLength=stepLength);
	}

		#Calculate aic and for model
		loglik<-fit1$objective
		k<-noSParam;
		aic<-(2*k)+(2*loglik);
		aicc<-aic+(2*k*(k+1)/(sampleSize-k-1));

			#To prevent a low AICc when there are more parameters than data!
		if(is.nan(aic)|is.nan(aicc)|(aicc==Inf)){}else{
			if(aicc<aic) aicc<-NaN;
		}
				
	#Extract names of variables

		varNames<-c(paste("Social Transmission",1:noSParam),"1/Rate scaling");

        callNextMethod(.Object,optimisation=fit1,sParam=sParam,loglik=loglik, aic=aic,aicc=aicc,varNames=varNames,...) 
       
    }
)


#Function for implementing the initialization
PSLdiscreteTadaFit<-function(tadata,sParam=NULL,startValue=NULL,stepLength=1){
	new("PSLdiscreteTadaFit",tadata=tadata,sParam=sParam,startValue=startValue,stepLength=stepLength)	
}


setMethod("anova",
    signature(object = "PSLdiscreteTadaFit"),
    function (object, ...) 
    {

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	   	if(is.null(object@optimisation$par[1])){
    
    		table<-data.frame(Df=1,LogLik=c(object@loglik),AIC=c(object@aic),AICc=c(object@aicc),row.names=c("Pure Social Transmission"));

    
    	}else{   

   		   table<-data.frame(Df=c(length(object@optimisation$par)),LogLik=c(object@loglik),AIC=c(object@aic),AICc=c(object@aicc),row.names=c("Pure Social Transmission"));
   		 
   		}

				atable<-structure(table, heading = "Pure social transmission model\n\nAll acquisition through social network\n", class = c("anova", "data.frame"))
		
		return(atable);

    }
)


setMethod("summary",
    signature(object = "PSLdiscreteTadaFit"),
    function (object, ...) 
    {

		cat("Summary of Pure Social Transmission Model\nDicrete Time of Acquisition Data\n");


		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

  		cat("Social transmission 1 constrained to 1.");
		if(is.null(object@optimisation$par)){ 
			sumtable<-data.frame(Estimate=c(1,object@optimisation$minimum),row.names=object@varNames);
		}else{
		    sumtable<-data.frame(Estimate=c(1,object@optimisation$par),row.names=object@varNames);   
    
    	}	 		
  		
		
		cat("\n\nCoefficients\n");
		print(sumtable);
		cat("\n\n")
		anova(object);
   
       
    }
)



nullSummary<-function (object,...) 
{


		cat("Summary of Null Model");
		if(class(object)[1]=="addFit"){cat("\nOrder of Acquisition Data\n")}else{
			cat("\nTime of Acquisition Data\n")
		}

		noSParam<-length(levels(as.factor(object@sParam[object@sParam>0])))

	    sumtable<-data.frame(Estimate=c(object@optimisationNull$par),row.names=object@varNames[-(1:noSParam)]);   
  
	

		
		cat("\n\nCoefficients\n");
		print(sumtable);
       
}



combinePairInStratum<-function(taData1, taData2){
	

	noIn1<-dim(taData1@oadata@assMatrix)[1]
	noIn2<-dim(taData2@oadata@assMatrix)[1]
	totalInd<-noIn1+noIn2

	newAssMatrix<-matrix(data=0,nrow=totalInd,ncol=totalInd);
	newAssMatrix[1:noIn1,1:noIn1]<-taData1@oadata@assMatrix;
	newAssMatrix[(noIn1+1):totalInd,(noIn1+1):totalInd]<-taData2@oadata@assMatrix;

	newOrderTemp<-c(taData1@oadata@orderAcq,(taData2@oadata@orderAcq+noIn1));
	newTimeTemp<-c(taData1@timeAcq,taData2@timeAcq);
	endTimeTemp<-c(rep(taData1@endTime,noIn1),rep(taData2@endTime,noIn2));
	
	if(is.na(taData1@oadata@updateTimes[1])) {UD1<-taData1@timeAcq}else {UD1<-taData1@oadata@updateTimes}
	if(is.na(taData2@oadata@updateTimes[1])) {UD2<-taData2@timeAcq}else {UD2<-taData2@oadata@updateTimes}
	newUpdatesTemp<-c(UD1,UD2);

	if(is.na(taData1@oadata@demons[1])) {demons1<-rep(0,noIn1)}else{demons1<-taData1@oadata@demons}
	if(is.na(taData2@oadata@demons[1])) {demons2<-rep(0,noIn2)}else{demons2<-taData1@oadata@demons}
	newDemons<-c(demons1,demons2);

	newOrder<-newOrderTemp[order(newTimeTemp)]
	newTime<-newTimeTemp[order(newTimeTemp)]
	endTimeVect<-endTimeTemp[order(newTimeTemp)]
	newUpdates<-newUpdatesTemp[order(newTimeTemp)]

	if(is.na(taData1@oadata@diffvect[1])){diffvect1<-rep(1,noIn1)}else{diffvect1<-taData1@oadata@diffvect};
	if(is.na(taData2@oadata@diffvect[1])){diffvect2<-rep(max(diffvect1)+1,noIn2)}else{diffvect2<-max(diffvect1)+taData2@oadata@diffvect};
	diffvect<-c(diffvect1,diffvect2);

	if(is.na(taData1@oadata@groupvect[1])) {groupvect1<-rep(taData1@oadata@groupid,noIn1)}else{groupvect1<-taData1@oadata@groupvect};
	if(is.na(taData2@oadata@groupvect[1])) {groupvect2<-rep(taData2@oadata@groupid,noIn2)}else{groupvect2<-taData2@oadata@groupvect};
	groupvect<-c(groupvect1,groupvect2)

	if(is.na(taData1@oadata@taskvect[1])) {taskvect1<-rep(taData1@oadata@taskid,noIn1)}else{taskvect1<-taData1@oadata@taskvect};
	if(is.na(taData2@oadata@taskvect[1])) {taskvect2<-rep(taData2@oadata@taskid,noIn2)}else{taskvect2<-taData2@oadata@taskvect};
	taskvect<-c(taskvect1,taskvect2);

	newAsoc<-rbind(taData1@oadata@asoc,taData2@oadata@asoc)

	return(taData(idname=NULL, assMatrix=newAssMatrix, asoc=newAsoc,orderAcq=newOrder,groupvect=groupvect, taskvect=taskvect, diffvect=diffvect,demons=newDemons,timeAcq= newTime,endTime=max(endTimeVect),updateTimes=newUpdates));
}

combineInStratum<-function(taNames){

	newTaObject<-eval(as.name(taNames[1]));
	
	for(i in 2:length(taNames)){
		newTaObject <-combinePairInStratum(newTaObject,eval(as.name(taNames[i])));

	}
	return(newTaObject@oadata)
}



#A function that fits all models given a set of variables/groups/tasks and social learning contrasts and returns  a table ordered by AIC or AICc
aicTableOA<-function(oadata,asocialVar=NULL,group=F,task=F,sParamMatrix=NULL,pure=F,aic="aicc",stratify="both",interval=NULL){

#Set sParam if not defined

if(is.character(oadata)){
	if(is.null(sParamMatrix)) sParamMatrix<-rbind(rep(1,length(oadata)));
	subdata<-eval(as.name(oadata[1]));
}else{
	if(is.null(sParamMatrix)) sParamMatrix<-1;
	subdata<-oadata;
}

#Calculate all possible combinations of ILVs
if(is.null(asocialVar)){asocialVarMatrix=0}else{
tempList<-NULL
for(i in 1:length(asocialVar)){
	if(is.null(tempList)){tempList<-list(c(0,1))}else{
		tempList<-c(tempList,list(c(0,1)));
	}
}
asocialVarMatrix<-expand.grid(tempList)
}

if(group){groupVect<-c(F,T)}else{groupVect<-F}
if(task){taskVect<-c(F,T)}else{taskVect<-F}

aiccTable<-aicTable<-NULL;

#Run through all combinations of models
for(sParamIndex in 1:dim(sParamMatrix)[1]){
	sParam<-sParamMatrix[sParamIndex,]
		for(group in groupVect){
			for(task in taskVect){
				for(ilvComb in 1:dim(as.matrix(asocialVarMatrix))[1]){
					#Extract the appropriate ILV vector
					if(is.null(asocialVar)){ilv<-NULL}else{
						tempILV<-as.numeric(asocialVarMatrix[ilvComb,]*asocialVar);
						ilv<-tempILV[tempILV>0];
					}
					if (length(ilv)==0) ilv<-NULL;
					if(is.null(ilv)&!group&!task){
						model<-NULL;
						model<-addFit(oadata=oadata,asocialVar=ilv,group=group,task=task,sParam= sParamMatrix[sParamIndex,],interval=interval)
					if(is.null(ilv)) ilv<-0;	
						if(is.null(aiccTable)){
							aiccTable<-c("NA",group,task,ilv,sParamIndex,"social",model@aicc)
						}else{
							aiccTable<-rbind(aiccTable,c("NA",group,task,ilv,sParamIndex,"social",model@aicc))
						}
						if(sParamIndex==1){aiccTable<-rbind(aiccTable,c("NA",group,task,ilv,sParamIndex,"asocial",model@aiccNull))}
						
												
						if(is.null(aicTable)){
							aicTable<-c("NA",group,task,ilv,sParamIndex,"social",model@aic)
						}else{
							aicTable<-rbind(aicTable,c("NA",group,task,ilv,sParamIndex,"social",model@aic))
						}
						if(sParamIndex==1){aicTable<-rbind(aicTable,c("NA",group,task,ilv,sParamIndex,"asocial",model@aicNull))}
										
					
					}else{		
					
						for(additive in c(TRUE,FALSE)){
							if(is.null(ilv)){}else{if(ilv==0) ilv<-NULL};
							model<-NULL;
							if(additive){
								model<-addFit(oadata=oadata,asocialVar=ilv,group=group,task=task,sParam= sParamMatrix[sParamIndex,])
							}else{
								if(is.null(ilv)){
									if(!group&!task) f1<-NULL
									if(group&task) f1<-formula(paste("~.","+group+task",collapse=""))
									if(group&!task) f1<-formula(paste("~.","+group",collapse=""))
									if(!group&task) f1<-formula(paste("~.","+task",collapse=""))									
								}else{
									
									if(!group&!task) f1<-formula(paste("~.",paste("+",(unlist(dimnames(subdata@asoc)[2]))[ilv],collapse="",sep=""),collapse=""))
									if(group&task) f1<-formula(paste(paste("~.",paste("+",(unlist(dimnames(subdata@asoc)[2]))[ilv],collapse="",sep=""),collapse=""),"+group+task",collapse=""))
									if(group&!task) f1<-formula(paste(paste("~.",paste("+",(unlist(dimnames(subdata@asoc)[2]))[ilv],collapse="",sep=""),collapse=""),"+group",collapse=""))
									if(!group&task) f1<-formula(paste(paste("~.",paste("+",(unlist(dimnames(subdata@asoc)[2]))[ilv],collapse="",sep=""),collapse=""),"+task",collapse=""))

								}
							
								if(!is.null(f1)) model<-multiCoxFit(oadata=oadata,formula=f1,sParam= sParamMatrix[sParamIndex,],stratify=stratify)
									
							}
										
						if(is.null(ilv)) ilv<-0;
						
						if(is.null(aiccTable)){
							aiccTable<-c(additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aicc)
						}else{
							aiccTable<-rbind(aiccTable,c(additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aicc))
						}
						if(additive&(sParamIndex==1)){aiccTable<-rbind(aiccTable,c("NA",group,task,paste(ilv,collapse=" "),sParamIndex,"asocial",model@aiccNull))}
						
												
						if(is.null(aicTable)){
							aicTable<-c(additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aic)
						}else{
							aicTable<-rbind(aicTable,c(additive,group,task,paste(ilv,collapse=" "),sParamIndex,"social",model@aic))
						}
						if(additive&(sParamIndex==1)){aicTable<-rbind(aicTable,c("NA",group,task,paste(ilv,collapse=" "),sParamIndex,"asocial",model@aicNull))}
	
						}
					}		
				}
			}
		}
	}


if(aic=="aicc"){
	dimnames(aiccTable)<-list(1:dim(aiccTable)[1],c("Additive?","Group?","Task?","ILVs","sParamIndex","Social?","AICc"))
	aiccTable<-aiccTable[order(aiccTable[,7]),];
	deltaAICc<-round(as.numeric(aiccTable[,7])-as.numeric(aiccTable[1,7]),2);
	aiccTable<-cbind(aiccTable,deltaAICc);
	return(aiccTable);
	
}else{
	dimnames(aicTable)<-list(1:dim(aicTable)[1],c("Additive?","Group?","Task?","ILVs","sParamIndex","Social?","AIC"))
	aicTable<-aicTable[order(aicTable[,8]),]	
	deltaAIC<-round(as.numeric(aicTable[,8])-as.numeric(aicTable[1,8]),2);
	aicTable<-cbind(aicTable,deltaAIC);
	return(aicTable)

}

}


	
#This is a temporary function needed to calculate the profile likelihood below
splitParamsLikelihoodTADA<-function(l,which, value,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE){
	
	newl<-rep(NA,length(l)+1)
	newl[-which]<-l
	newl[which]<-value
	tadaLikelihood(l=newl,tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive)
	
}


#Now for a function that calculates the profile likelihood on the negative log scale
#This is the likelihood for one parameter fixed to a specified value, after the other parameters have been optimised
#Specify which parameter is fixed with which, referring to its place in the parameter vector, and what its value is fixed to with value

profileLikelihoodTADA<-function(value, which, model,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	start<-model@optimisation$par[-which]
	if(length(model@optimisation$par)==2){
		if(is.null(intervalB)){intervalB<-c(0,99999)}
		model2<-optim(par=start, fn= splitParamsLikelihoodTADA,which=which, value=value, method="Brent",lower= intervalB[1], upper= intervalB[2], tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive)	
	}else{
	    model2<-optim(par=start, fn= splitParamsLikelihoodTADA,control=list(maxit=10000,parscale=rep(1,length(start))+start*(start>100)), which=which, value=value,tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive)
	}
	return(model2$value)
}

plotProfLikelihoodTADA<-function(range,which,resolution, model,tadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	x<-seq(range[1],range[2],length=resolution)
	profLik<-rep(NA,length(x))
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	for (i in 1:length(x)){
		profLik[i]<-profileLikelihoodTADA(value=x[i], which=which,model=model, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)
		plot(x[1:i],profLik[1:i],type="l",xlim=range,ylim=c(maxLik,max(profLik[1:i])),xlab="Parameter value",ylab="Profile likelihood (log scale)")
		abline(h=cutoff, lty=2)
	}
	return(cbind(x,profLik))
}

#The function gives the difference between the value of the ProfLik at a point, and the point the ProfLik intersects with the cutoff line
#The function is used only to allow the profLikCI function below to work
distanceFromIntersectionTADA<-function(value,which, model,tadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	return(abs(profileLikelihoodTADA(value=value, which=which,model=model, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)-cutoff))
}

#Find profile likelihood confidence intervals. Arguments as above except the user must give a pair of non-overlapping ranges within which the upper and lower
#points of the confidence interval lie. The plotProfLik function above can be used to find these intervals.
profLikCItada<-function(which,upperInt,lowerInt=NULL, model,tadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	upperPoint<-optimise(distanceFromIntersectionTADA,interval=upperInt,which=which,model=model,inflation=inflation,conf=conf, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)$minimum
	cat("Upper=",upperPoint)
	flush.console()
	if(is.null(lowerInt)){lowerPoint<-NA}else{
		lowerPoint<-optimise(distanceFromIntersectionTADA,interval=lowerInt,which=which,model=model,inflation=inflation,conf=conf, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)$minimum
	}
	cat("\nLower=",lowerPoint,"\n")
	flush.console()
	return(c(lowerPoint,upperPoint))
}


#Now the versions for the gamma TADA version


#This is a temporary function needed to calculate the profile likelihood below
splitParamsLikelihoodgammaTADA<-function(l,which, value,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE){
	
	newl<-rep(NA,length(l)+1)
	newl[-which]<-l
	newl[which]<-value
	gammaTadaLikelihood(l=newl,tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive)
	
}

#Now for a function that calculates the profile likelihood on the negative log scale
#This is the likelihood for one parameter fixed to a specified value, after the other parameters have been optimised
#Specify which parameter is fixed with which, referring to its place in the parameter vector, and what its value is fixed to with value

profileLikelihoodgammaTADA<-function(value, which, model,tadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	start<-model@optimisation$par[-which]
    model2<-optim(par=start, fn= splitParamsLikelihoodgammaTADA,control=list(maxit=10000,parscale=rep(1,length(start))+start*(start>100)), which=which, value=value,tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive)
	return(model2$value)
}

plotProfLikelihoodgammaTADA<-function(range,which,resolution, model,tadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	x<-seq(range[1],range[2],length=resolution)
	profLik<-rep(NA,length(x))
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	for (i in 1:length(x)){
		profLik[i]<-profileLikelihoodgammaTADA(value=x[i], which=which,model=model, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)
		plot(x[1:i],profLik[1:i],type="l",xlim=range,ylim=c(maxLik,max(profLik[1:i])),xlab="Parameter value",ylab="Profile likelihood (log scale)")
		abline(h=cutoff, lty=2)
	}
	return(cbind(x,profLik))
}

#The function gives the difference between the value of the ProfLik at a point, and the point the ProfLik intersects with the cutoff line
#The function is used only to allow the profLikCI function below to work
distanceFromIntersectiongammaTADA<-function(value,which, model,tadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	return(abs(profileLikelihoodgammaTADA(value=value, which=which,model=model, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)-cutoff))
}

#Find profile likelihood confidence intervals. Arguments as above except the user must give a pair of non-overlapping ranges within which the upper and lower
#points of the confidence interval lie. The plotProfLik function above can be used to find these intervals.
profLikCIgammatada<-function(which,upperInt,lowerInt=NULL, model,tadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	upperPoint<-optimise(distanceFromIntersectiongammaTADA,interval=upperInt,which=which,model=model,inflation=inflation,conf=conf, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)$minimum
	cat("Upper=",upperPoint)
	flush.console()
	if(is.null(lowerInt)){lowerPoint<-NA}else{
		lowerPoint<-optimise(distanceFromIntersectiongammaTADA,interval=lowerInt,which=which,model=model,inflation=inflation,conf=conf, tadata= tadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,task= task,group= group,additive= additive,intervalB=intervalB)$minimum
	}
	cat("\nLower=",lowerPoint,"\n")
	flush.console()
	return(c(lowerPoint,upperPoint))
}


	
#This is a temporary function needed to calculate the profile likelihood below
splitParamsLikelihoodAddOADA<-function(l,which, value,oadata,sParam=NULL,bounded=FALSE,asocialVar=NULL){
	
	newl<-rep(NA,length(l)+1)
	newl[-which]<-l
	newl[which]<-value
	addLikelihood(l=newl,data= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar)
	
}



#Now versions for the additive OADA

profileLikelihoodAddOADA<-function(value, which, model,oadata,sParam=NULL,bounded=FALSE,asocialVar=NULL,intervalB=NULL){
	start<-model@optimisation$par[-which]
	if(length(model@optimisation$par)==0){
		return(addLikelihood(value,data= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar))
	}
	if(length(model@optimisation$par)==2){
		if(is.null(intervalB)){intervalB<-c(-99,99)}
		model2<-optim(par=start, fn= splitParamsLikelihoodAddOADA,which=which, value=value, method="Brent",lower= intervalB[1], upper= intervalB[2], oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar)	
	}else{
	    model2<-optim(par=start, fn= splitParamsLikelihoodAddOADA,which=which, value=value,oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar)
 	   if(model2$convergence>0) model2<-optim(par=model2$par, fn= splitParamsLikelihoodOADA,which=which, value=value,oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar)
	}
	return(model2$value)
}




plotProfLikelihoodAddOADA<-function(range,which,resolution, model,oadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,intervalB=NULL){
	x<-seq(range[1],range[2],length=resolution)
	profLik<-rep(NA,length(x))
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	for (i in 1:length(x)){
		profLik[i]<-profileLikelihoodAddOADA(value=x[i], which=which,model=model, oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,intervalB=intervalB)
		plot(x[1:i],profLik[1:i],type="l",xlim=range,ylim=c(maxLik,max(profLik[1:i])),xlab="Parameter value",ylab="Profile likelihood (log scale)")
		abline(h=cutoff, lty=2)
	}
	return(cbind(x,profLik))
}


#This is a temporary function needed to calculate the profile likelihood below
splitParamsLikelihoodMultiCoxOADA<- function(l,which, value,oadata,bounded=FALSE, formula =NULL){
	
	newl<-rep(NA,length(l)+1)
	newl[-which]<-l
	newl[which]<-value
	multiCoxLikelihood(s=newl,oadata= oadata,bounded= bounded, formula = formula)
	
}

profileLikelihoodMultiCoxOADA<-function(value, which, model,oadata,sParam=NULL,bounded=FALSE,formula=NULL,intervalB=NULL){
	start<-model@optimisation$par[-which]
	if(length(model@optimisation$par)==1){
		return(multiCoxLikelihood(value,oadata= oadata,bounded= bounded, formula = formula))
	}
	if(length(model@optimisation$par)==2){
		if(is.null(intervalB)){intervalB<-c(0,99)}
		model2<-optim(par=start, fn= splitParamsLikelihoodMultiCoxOADA,which=which, value=value, method="Brent",lower= intervalB[1], upper= intervalB[2], oadata= oadata,bounded= bounded,formula= formula)	
	}else{
	    model2<-optim(par=start, fn= splitParamsLikelihoodMultiCoxOADA,which=which, value=value,oadata= oadata,bounded= bounded, formula = formula)
 	   if(model2$convergence>0) model2<-optim(par=model2$par, fn= splitParamsLikelihoodOADA,which=which, value=value,oadata= oadata,sbounded= bounded, formula = formula)
	}
	return(model2$value)	
}

#Currently only works with a single s parameter
plotProfLikelihoodMultiCoxOADA<-function(range,which,resolution, model,oadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,formula=NULL,intervalB=NULL,stratify="both"){
	if(is.character(oadata)) oadata<-combineOaCoxData(oadata,sParam=sParam)
	x<-seq(range[1],range[2],length=resolution)
	profLik<-rep(NA,length(x))
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	for (i in 1:length(x)){
		profLik[i]<-profileLikelihoodMultiCoxOADA(x[i],which=which,model=model,oadata=oadata,formula=formula,bounded=bounded,stratify)
		plot(x[1:i],profLik[1:i],type="l",xlim=range,ylim=c(maxLik,max(profLik[1:i])),xlab="Parameter value",ylab="Profile likelihood (log scale)")
		abline(h=cutoff, lty=2)
	}
	return(cbind(x,profLik))
}



#The function gives the difference between the value of the ProfLik at a point, and the point the ProfLik intersects with the cutoff line
#The function is used only to allow the profLikCI function below to work
distanceFromIntersectionAddOADA<-function(value,which, model,oadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	return(abs(profileLikelihoodAddOADA(value=value, which=which,model=model, oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,intervalB=intervalB)-cutoff))
}

#The function gives the difference between the value of the ProfLik at a point, and the point the ProfLik intersects with the cutoff line
#The function is used only to allow the profLikCI function below to work
distanceFromIntersectionMultiCoxOADA<-function(value,which, model,oadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,formula=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	maxLik<-model@optimisation$objective
	cutoff=maxLik+qchisq(1-conf,1,lower.tail=F)*inflation/2
	return(abs(profileLikelihoodMultiCoxOADA(value=value,which=which,model=model,oadata= oadata,bounded= bounded,formula= formula)-cutoff))
}

#Find profile likelihood confidence intervals. Arguments as above except the user must give a pair of non-overlapping ranges within which the upper and lower
#points of the confidence interval lie. The plotProfLik function above can be used to find these intervals.
profLikCIaddOADA<-function(which,upperInt,lowerInt=NULL, model,oadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL){
	upperPoint<-optimise(distanceFromIntersectionAddOADA,interval=upperInt,which=which,model=model,inflation=inflation,conf=conf, oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,intervalB=intervalB)$minimum
	cat("Upper=",upperPoint)
	flush.console()
	if(is.null(lowerInt)){lowerPoint<-NA}else{
		lowerPoint<-optimise(distanceFromIntersectionAddOADA,interval=lowerInt,which=which,model=model,inflation=inflation,conf=conf, oadata= oadata,sParam= sParam,bounded= bounded,asocialVar= asocialVar,intervalB=intervalB)$minimum
	}
	cat("\nLower=",lowerPoint,"\n")
	flush.console()
	return(c(lowerPoint,upperPoint))
}


profLikCIMultiCoxOADA <-function(which,upperInt,lowerInt=NULL, model,oadata,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,formula=formula,task=FALSE,group=FALSE, intervalB=NULL){
	if(is.character(oadata)) oadata<-combineOaCoxData(oadata,sParam=sParam)
	upperPoint<-optimise(distanceFromIntersectionMultiCoxOADA,interval=upperInt,which=which,model=model,inflation=inflation,conf=conf, oadata= oadata,sParam= sParam,bounded= bounded,formula= formula,intervalB=intervalB)$minimum
	cat("Upper=",upperPoint)
	flush.console()
	if(is.null(lowerInt)){lowerPoint<-NA}else{
		lowerPoint<-optimise(distanceFromIntersectionMultiCoxOADA,interval=lowerInt,which=which,model=model,inflation=inflation,conf=conf, oadata= oadata,sParam= sParam,bounded= bounded,formula= formula,intervalB=intervalB)$minimum
	}
	cat("\nLower=",lowerPoint,"\n")
	flush.console()
	return(c(lowerPoint,upperPoint))
}


#Now wrapper functions to make things easier for the user!

plotProfLikelihood<- function(range,which,resolution, model,data,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL,formula=NULL){

	if(class(model)=="gammaTadaFit"){
		return(plotProfLikelihoodgammaTADA(range=range,which=which,resolution=resolution, model=model,tadata=data,inflation=inflation,conf=conf, sParam=sParam,bounded=bounded,asocialVar=asocialVar,task=task,group=group,additive=additive,intervalB=intervalB))
	}
	if(class(model)=="tadaFit"){
		return(plotProfLikelihoodTADA(range=range,which=which,resolution=resolution, model=model,tadata=data,inflation=inflation,conf=conf, sParam=sParam,bounded=bounded,asocialVar=asocialVar,task=task,group=group,additive=additive,intervalB=intervalB))
	}
	if(class(model)=="addFit"){
		return(plotProfLikelihoodAddOADA(range=range,which=which,resolution=resolution, model=model,oadata=data,inflation=inflation,conf=conf, sParam=sParam,bounded=bounded,asocialVar=asocialVar,intervalB=intervalB))
	}
	if(class(model)=="multiCoxFit"){
		return(plotProfLikelihoodMultiCoxOADA(range=range,which=which,resolution=resolution, model=model,oadata=data,inflation=inflation,conf=conf, sParam=sParam,bounded=bounded,formula=formula,intervalB=intervalB))
	}
}


profLikCI<-function(which,upperInt,lowerInt=NULL, model,data,inflation=1,conf=0.95,sParam=NULL,bounded=FALSE,asocialVar=NULL,task=FALSE,group=FALSE,additive=TRUE,intervalB=NULL, formula=formula){
	if(class(model)=="gammaTadaFit"){
		return(profLikCIgammatada(which=which,upperInt=upperInt,lowerInt=lowerInt, model=model,tadata=data,inflation=inflation,conf=conf,sParam=sParam,bounded=bounded,asocialVar=asocialVar,task=task,group=group,additive=additive,intervalB=intervalB))
	}
	if(class(model)=="tadaFit"){
		return(profLikCItada(which=which,upperInt=upperInt,lowerInt=lowerInt, model=model,tadata=data,inflation=inflation,conf=conf,sParam=sParam,bounded=bounded,asocialVar=asocialVar,task=task,group=group,additive=additive,intervalB=intervalB))
	}
	if(class(model)=="addFit"){
		return(profLikCIaddOADA(which=which,upperInt=upperInt,lowerInt=lowerInt, model=model,oadata=data,inflation=inflation,conf=conf,sParam=sParam,bounded=bounded,asocialVar=asocialVar,intervalB=intervalB))
	}
		if(class(model)=="multiCoxFit"){
		return(profLikCIMultiCoxOADA(which=which,upperInt=upperInt,lowerInt=lowerInt, model=model,oadata=data,inflation=inflation,conf=conf,sParam=sParam,bounded=bounded,formula=formula))
	}
}

plotAssociations<-function(data,lty=1, symbol=NULL, xlab="Acquisition event", ylab="Total connection to informed individuals",title=NULL,plotID=T,offset=c(0.1,0),xlim=NULL, ylim=NULL){
if(class(data)=="oaData"){oadata<-data}
if(class(data)=="taData"){oadata<-data@oadata}

if(is.null(xlim)){xlim<-c(0,max(oadata@coxdata$time2)+0.5)}

	if(is.null(symbol)){
		plot(oadata@coxdata$time2,oadata@coxdata$stMetric,col=oadata@coxdata$status+1,xlab=xlab, ylab=ylab,main=title,xlim=xlim, ylim=ylim);
		points(oadata@coxdata$time2[oadata@coxdata$status==1],oadata@coxdata$stMetric[oadata@coxdata$status==1],col=2);
		lines(oadata@coxdata$time2[oadata@coxdata$status==1],oadata@coxdata$stMetric[oadata@coxdata$status==1],col=2, lty=lty);		
	}else{
		plot(oadata@coxdata$time2,oadata@coxdata$stMetric,col=oadata@coxdata$status+1,pch=as.numeric(as.factor(oadata@coxdata[,6+symbol])),xlab=xlab, ylab=ylab, main=title,xlim=xlim, ylim=ylim);
		points(oadata@coxdata$time2[oadata@coxdata$status==1],oadata@coxdata$stMetric[oadata@coxdata$status==1],col=2,pch=as.numeric(as.factor(oadata@coxdata[,6+symbol]))[oadata@coxdata$status==1]);
		lines(oadata@coxdata$time2[oadata@coxdata$status==1],oadata@coxdata$stMetric[oadata@coxdata$status==1],col=2, lty=lty);
	}
	if(plotID){
			text(oadata@coxdata$time2[oadata@coxdata$status==1]+offset[1],oadata@coxdata$stMetric[oadata@coxdata$status==1]+offset[2],col=2,labels=oadata@coxdata$id[oadata@coxdata$status==1]);		
	}
}