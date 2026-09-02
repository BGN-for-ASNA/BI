#  Introduction

This repository presents a Bayesian model designed by Richard McElreath allowing to evaluate the size of signal repertoire (vocal/gestural/other) at the individual level.
The model accounts for 
  - The duration individuals have been observed
  - The number of recordings collected
  - The frequency at which each individual vocalizes
  - The commonness of each signal type

#  Applicability
This model can thus be used for samples with unbalanced datasets on each individual and can be used to compare individuals across age, sex, populations and even species. 
It can be apply to single signals but also, like here, to signal combinations (call sequences or gesture sequences). 
A detail description of the model can be found at:  https://doi.org/10.1038/s42003-025-07922-2 

#  Citation
Sigmundson, R., Girard-Buttoz, C., Le Floch, A., Souha Azaiez, T., McElreath, R., Zuberbühler, K., Wittig, R. M., & Crockford, C. (2025). Vocal sequence diversity and length remain stable across ontogeny in a catarrhine monkey (Cercocebus atys). Communications Biology, 8:465. https://doi.org/10.1038/s42003-025-07922-2

#  Content
The script: "Mangabey ontogeny Model Rethinking version.R" provides the code for the model.
The "input" folder comprises all the datasets used in the paper that apply the same model structure on them.  

#  Resources needed
Library needed for running the model: Rethinking Package: https://github.com/rmcelreath/stat_rethinking_2024
The stan code for this model can be found at : https://github.com/rmcelreath/cg_vocal_repertoires

# Details

The model to provides an estimate of individual full repertoire. The primary problem in estimating this repertoire is that a finite sample will tend to underestimate the total repertoire. This is especially true if some vocalizations are used rarely. This problem has been recognized specifically in vocal repertoires for decades, and is shared with diversity estimation more generally. We build upon previous work by constructing a new probabilistic estimator of individual repertoire that accounts for undersampling. 
Specifically, let M be the number of unique vocalizations in the population. Each of N individuals has an unknown repertoire phi, which is an M-length vector of zeros and ones, indicating which vocalizations the individual possesses. When an individual is observed to produce vocalization m, then phi_m is assigned 1. The estimation comes when m is not observed in the sample. Assume that vocalization m is produced at a rate lambda_m, when phi_m = 1. Then in a recording of duration d, the observed count of vocalization m, Y_m, is Poisson distributed with mean d lambda_m. For Y_m > 0, the probability of the data is given by the Poisson probability mass function 
	Pr( Y_m = y | lambda_m , d , p_m ) = p_m (lambda_m d)^y exp(–lambda_m d) / y!
where p_m is the unknown prevalence of vocalization m in the population. When Y_m = 0, we require instead a mixture probability accounting for the possibility that phi_m = 1 but that m nevertheless went unobserved. This probability is given by:
	Pr( Y_m = 0 | lambda_m , d , p_m ) = p_m Pr(Y_m=0|phi_m=1) + (1-p_m)Pr(Y_m=0|phi_m=0) = p_m exp(–lambda_m d) + (1 – p_m)
This probability expression allows us to compute the probability that an individual possesses m, Pr(phi_m=1), given m was not observed. By Bayes theorem:
	Pr(phi_m=1) = Pr(Y_m=0|phi_m=1) Pr(phi_m=1) / Pr(Y_m=0| lambda_m , d , p_m)
We implemented this model in the Stan probabilistic programming language108. We stratified the variables lambda and p by age category (infant, juvenile, subadult, adult) and sex, but this is not an essential feature of the approach and is easily modified in the provided code. The detailed sample size for the data used in the analyses and in particular the number of individuals of each sex and age class as well as the number of utterances analysed are all presented in Table S1. We validated our implementation by simulating synthetic samples of vocalizations in which ground truth repertoires are known and verifying that the code functions as intended. In particular, we tested the ability of our model to recover ground truth on a simulated dataset of 50 individuals in a theoretical species with 50 different utterance types and with 20 recordings per individuals. This test led to accurate recovering of ground truth even for individuals where the observed vocal repertoire size fell far below the true repertoire size (see details at https://github.com/rmcelreath/cg_vocal_repertoires/blob/main/simulation.r). In addition to the Stan code, we provide an implementation using the rethinking package, which may be more convenient to modify 109. The code uses Hamiltonian Monte Carlo to produce samples from the posterior distributions for the repertoire of each individual. 
