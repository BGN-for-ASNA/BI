Hi Richard,

I am contacting you regarding the Bayesian inference (BI) package I have been developing for python and R. I think we reach a step where all features of the package have been implemented and tested.

The package incorporate the 24 models:
    
- 1. Linear Regression for continuous variable.qmd
- 2. Multiple continuous Variables.qmd
- 3. Interaction between continuous variables.qmd
- 4. Categorical variable.qmd
- 5. Binomial model.qmd
- 6. Beta binomial model.qmd
- 7. Poisson model.qmd
- 8. Gamma-Poisson.qmd
- 9. Multinomial model.qmd
- 10. Dirichlet model (wip).qmd
- 11. Zero inflated.qmd
- 12. Survival analysis.qmd
- 13. Varying intercepts.qmd
- 14. Varying slopes.qmd
- 15. Gaussian processes.qmd
- 16. Measuring error (wip).qmd
- 17. Missing data (wip).qmd
- 18. Latent variable.qmd
- 19. PCA.qmd
- 20. Network model.qmd
- 21. Network with block model.qmd 
- 22. Network control for data collection biases (wip).qmd
- 23. Network Metrics.qmd
- 24. Network Based Diffusion analysis (wip).qmd

As you can see most of them are from rethinking and without your video, book and classes I followed I would not have been able to realize this package. I thus believe it would be fair to have you as co-author in the manuscript (with Cody and Mary).

To be honest there is also some think I would like you to help us on the package that will make me feel more confident when publishing the package. 

First, for each model of rethinking we made the following test protocol:
    - Compare posterior distribution for both BI package and STAN
    - Run simulation and parameter recovery for the BI model
 
But for some of them I still have issues regarding parameter recovery (for the poisson and negative binomial model). I will appreciate if you could check what could be the issue.

Second, for each model I made a documentation that is suppose around 2 min reads that dive from overall and simple explanation of the model to the full mathematical details. I wonder if you would have time to check some of the particularly the following sections:

    - 13. Varying intercepts.qmd
    - 14. Varying slopes.qmd
    - 15. Gaussian processes.qmd