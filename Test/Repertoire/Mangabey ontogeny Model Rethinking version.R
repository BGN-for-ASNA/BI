#load necessary packages

library(rethinking) 
# to install the rethinking package please visit: https://github.com/rmcelreath/rethinking 


### upload the mangabey dataset

dm = read.csv("input/full model with repeats.csv", header = T)

### This script provide one example for the main dataset for the analysis. We provide 9 different data sets for all the sub models we ran (see details in the manuscript). 
# The script for the analysis is the same for each of this datasets.  


#### I/ prepare the data for the ulam model (rethinking package) ####

### define the number of individuals 

N1 = length(unique(dm$id))

### define the number of different vocalisations

M1 = length(colnames(dm)) - 7

## define the number of 'recordings' 

J1 = nrow(dm)

# Maximum number of different utterance on 1 focal 

### create a vector to store the different number per day

xx = c()

for ( i in 1:nrow(dm)){
  
  xx[i] = length(which(dm[i,7:ncol(dm)]>0))
  
  
}

length(xx) #341


K1 = max(xx)



#### define the duration

d1 = dm$foc.duration


### define the identity of the caller

id1 = as.numeric(as.factor(dm$id))

dm$id.num = id1 

#### transform the age classes into numerical

dm$age.class.num = NA
dm$age.class.num[which(dm$age.class=='I')] = 1  # assign value 1 to infants
dm$age.class.num[which(dm$age.class=='J')] = 2  # assign value 2 to junvenils
dm$age.class.num[which(dm$age.class=='S')] = 3  # assign value 3 to subadults
dm$age.class.num[which(dm$age.class=='A')] = 4  # assign value 4 to adults


#### transform the sex into numerical

dm$sex.num = NA
dm$sex.num[which(dm$sex=='F')] = 1  # assign value 1 to females
dm$sex.num[which(dm$sex=='M')] = 2  # assign value 2 to males


#### transform the group into numerical

dm$group.num = NA
dm$group.num[which(dm$group == 'TCP')] = 1  # assign value 1 to TCP group
dm$group.num[which(dm$group == 'TMP')] = 2  # assign value 2 to TMP group


### create a list of age, sex and group for each individual in the dataset

u.id = sort(unique(id1))
age = dm$age.class.num[match(u.id,dm$id.num)]
sex = dm$sex.num[match(u.id,dm$id.num)]
group = dm$group.num[match(u.id,dm$id.num)]




### calculate the length of each utterance (colnames)

voc.list = colnames(dm[,12:ncol(dm)-4])

voclen <- c()  ### define the length of each unique vocalisations

for (i in 1: length(voc.list)){
  xx = unlist(strsplit(voc.list[i], split ='_'))  
  voclen[i] = length(xx)
}

### standardise the length of each vocalisation

vlength <- standardize(log(voclen))



length(age) #74
length(sex) #74
length(unique(dm$id)) #74

### XX1 : prepare the data to input it in the model later

XX1 = as.matrix(dm[,8:(ncol(dm)-4)])


dat1 <- list(
  N = N1,
  M = M1,
  J = J1,
  K = K1,
  d = d1,
  id = id1,
  Y = XX1 )

# convert Y matrix to long form with m column
YY <- rep(NA,length(dat1$Y))
MM <- YY
ID <- YY
JJ <- YY
k <- 1

for ( j in 1:nrow(dat1$Y) ) for ( m in 1:ncol(dat1$Y) ) {
  YY[k] <- dat1$Y[j,m]
  MM[k] <- m
  ID[k] <- dat1$id[j]
  JJ[k] <- j
  k <- k + 1
}



dat_list1 <- list(
  Y = YY,    # behavior counts
  ID = ID,   # individual ID
  BID = MM,  # behavior ID
  J = JJ,    # recording ID
  D = dat1$d, # duration of recording,
  M = max(MM),
  age = age,
  sex = sex,
  group = group,
  vlen = vlength
)


#### Run the Model ####

m_full2 <- ulam(
  alist(
    # Y > 0
    Y|Y>0 ~ custom( log(p) + poisson_log_lpmf(Y|log_lambda1) ),
    # Y = 0
    Y|Y==0 ~ custom( log_sum_exp( 
      log(p) + poisson_log_lpmf(0|log_lambda1) , 
      log1m(p)
    ) ),
    # models for rate of behavior 
    log_lambda1 <- log(L[BID]) + 
      la[BID,group[ID]]*sigmal + 
      # age-sex offsets for each group
      lB_mean[age[ID],sex[ID]] + 
      lB1[BID,age[ID],sex[ID]]*phil + 
      lB[BID,age[ID],sex[ID],group[ID]]*taul +
      lbvl*vlen[BID] +
      log(D[J]),
    
    save> logit(p) <-   
      ### part of the model calculating the probability to have each vocalistion for each age sex group classes
      # vocalization intercepts for each group
      a_mean[BID] + a[BID,group[ID]]*sigma + 
      # age-sex offsets for each group
      B_mean[age[ID],sex[ID]] + 
      B1[BID,age[ID],sex[ID]]*phi + 
      B[BID,age[ID],sex[ID],group[ID]]*tau + 
      # association of vocalization length with intercept
      bvl*vlen[BID],
    # priors
    # rate priors
    vector[M]:L ~ exponential(1),
    matrix[M,2]:la ~ normal(0,1),
    matrix[4,2]:lB_mean ~ normal(0,1),
    real["90,4,2"]:lB1 ~ normal(0,1),
    real["90,4,2,2"]:lB ~ normal(0,1),
    # p priors
    c(bvl,lbvl) ~ normal(0,1),
    vector[M]:a_mean ~ normal(0,1),
    matrix[M,2]:a ~ normal(0,1),
    matrix[4,2]:B_mean ~ normal(0,1),
    real["90,4,2"]:B1 ~ normal(0,1),
    real["90,4,2,2"]:B ~ normal(0,1),
    # scale priors
    c(sigma,tau,phi,sigmal,taul,phil) ~ exponential(1)
  ), data=dat_list1 , chains=3 , cores=3 , sample=TRUE , iter=500) 


#### model output of the major parameters


precis(m_full2,3,pars=c("a_mean","a","B_mean","B","B1","sigma","tau","phi","sigmal","taul","phil"))


postf2 <- extract.samples(m_full2)


S <- 750 # number of samples

### calculate the estimated repertoire size for each sex age category for both groups
#### Group 2 is TMP group i.e.,  the one missing data for both sexes in age class 1 and 2 (Infants and Juveniles)

p_age_sexf2 <- array(NA,c(S,dat_list1$M,4,2,2)) # sample,vocalization,age,sex
for ( age in 1:4 ) for ( sex in 1:2 ) for (group in 1:2) for ( v in 1:dat_list1$M ) {
  p_age_sexf2[,v,age,sex, group] <- sapply( 1:S , function(s) with(postf2,
                                                                   inv_logit( 
                                                                     a_mean[s,v] + a[s,v,group]*sigma[s] + 
                                                                       B_mean[s,age,sex] + 
                                                                       B1[s,v,age,sex]*phi[s] + 
                                                                       B[s,v,age,sex,group]*tau[s] +
                                                                       bvl[s]*vlength[v]
                                                                   )
  ))
}


prob.v2 = apply( p_age_sexf2 , 2:5, mean ) ###probability for calls to be present

rownames(prob.v2) = voc.list


#### Calculate the repertoire size for each age sex class

rep_age_sexf2 <- array(NA,c(S,4,2,2))

for ( age in 1:4 ) for ( sex in 1:2 ) for (group in 1:2) for ( s in 1:S ) {
  rep_age_sexf2[s,age,sex, group] <- sum( p_age_sexf2[s,,age,sex, group] )
}

repf2 = apply( rep_age_sexf2 , 2:3 , mean )

apply( rep_age_sexf2 , 2:3 , mean )

#          [,1]      [,2]
# [1,] 28.22738 24.143496
# [2,] 11.68407  7.664085
# [3,] 15.58795  8.787543
# [4,] 11.74383  9.555013

cif2 = apply( rep_age_sexf2, 2:3 , PI, 0.89 )

# , , 1

#         [,1]      [,2]      [,3]      [,4]
# 5%  16.29761  7.632797  9.875211  8.233753
# 94% 44.32512 16.923644 22.820699 16.041766

# , , 2

#         [,1]      [,2]     [,3]      [,4]
# 5%  15.01138  5.113594  4.79357  4.925638
# 94% 34.90351 11.200381 15.43208 15.903374


rep_age_sexf2[,1,1,1]



#### evaluate the last table details ####

apply(p_age_sexf2, 2:4, mean)
apply(p_age_sexf2, 2:4, sd)


p_age_sexf2[,,age,sex,] 


# now simulate repertoires and calculate maximum vocalization length for each
maxlen_age_sexf2 <- array(NA,c(S,4,2))

for ( age in 1:4 ) for ( sex in 1:2 ) for ( s in 1:S ) {
  sim_rep <- rbern( dat_list1$M , prob=p_age_sexf2[s,,age,sex,] )
  lengths <- sim_rep * voclen
  maxlen_age_sexf2[s,age,sex] <- max( lengths )
}
 
maxlen_age_sexf2[,,1]

apply( maxlen_age_sexf2 , 2:3 , mean ) ### estimated mean max length for each age sex class 

# [,1]     [,2]
# [1,] 8.929333 8.676000
# [2,] 6.610667 5.773333
# [3,] 7.452000 6.042667
# [4,] 6.864000 6.138667

apply( maxlen_age_sexf2 , 2:3 , PI, 0.89 ) ### 89% CI for the max length

# , , 1

#      [,1] [,2] [,3] [,4]
# 5%     5    4    4    4
# 94%   11   11   11   11

# , , 2

#      [,1] [,2] [,3] [,4]
# 5%     5    2    3    3
# 94%   11   11   11   11


### create some contrast to calculate the difference between age and sex classes

### create table result for max length ####

tab.max = expand.grid(c(1,2,3,4),c(1,2),c(1,2,3,4),c(1,2))
colnames(tab.max) =c('ID1.age.num','ID1.sex.num', 'ID2.age.num','ID2.sex.num')

tab.max$mean.dif = NA
tab.max$sd = NA
tab.max$l.ci = NA
tab.max$u.ci = NA

for (i in 1:nrow(tab.max)){
  
  ## take posterior for ID 1
  xx1 = maxlen_age_sexf2[,tab.max$ID1.age.num[i],tab.max$ID1.sex.num[i]]
  
  ## take posteriro for ID 2
  xx2 = maxlen_age_sexf2[,tab.max$ID2.age.num[i],tab.max$ID2.sex.num[i]]
  
  tab.max$mean.dif[i] = mean(xx1-xx2)
  tab.max$sd[i] = sd(xx1-xx2)
  tab.max$l.ci[i] = quantile(xx1-xx2, probs = c(0.045, 0.945))[1]
  tab.max$u.ci[i] = quantile(xx1-xx2, probs = c(0.045, 0.945))[2]
  
  
}

xx = expand.grid(c('I','J','S','A'), c('F','M'),c('I','J','S','A'), c('F','M'))

xx$age.sex1 = paste(xx$Var1,xx$Var2,sep = '_')
xx$age.sex2 = paste(xx$Var3,xx$Var4,sep = '_')

tab.max = cbind(tab.max,xx[,c('age.sex1','age.sex2')])

### remove categories with themselves

tab.max = tab.max[-which(tab.max$age.sex1 == tab.max$age.sex2),]

nrow(tab.max)
### in this tab you will notice that some comparisons appear twice in both direction you just will have to suppress them



