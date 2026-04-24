Received: 19 September 2025     | Accepted: 23 January 2026
DOI: 10.1111/2041-210x.70272


RESEARCH ARTICLE




Generalized graphical mixed models connect ecological theory
with widely used statistical models

James T. Thorson

Resource Ecology and Fisheries
Management Division, Alaska Fisheries            Abstract
Science Center, National Marine Fisheries
                                                 1. Ecological dynamics are often analysed across multiple sites, times, and variables.
Service, NOAA, Seattle, Washington, USA
                                                     Ecologists typically represent interactions across space, time, and variables using
Correspondence
                                                     generalized linear mixed models (GLMMs), generalized additive models (GAM),
James T. Thorson
Email: james.thorson@noaa.gov                        and structural equation models (SEM).
                                                 2. Here, I introduce the family of generalized graphical mixed models (GGMMs) and
Handling Editor: Nicolas Lecomte
                                                     show that it extends GLMMs, GAMs, and SEMs. GGMMs represent ecological
                                                     systems using a mathematical graph, where each analytic unit (node) has a direct
                                                     effect on other units via specified linear interactions contained in a joint path ma-
                                                     trix (edges). The GGMM first requires defining the dimensions and resolution of
                                                     analytical units, and I illustrate concepts using combinations of sites, times, ages,
                                                     and taxa. Next, the path matrix is constructed by combining elementary ecologi-
                                                     cal relationships like ecological interactions, evolutionary trade-­offs, time lags,
                                                     and spatial diffusion. Finally, the GGMM is expressed as a simultaneous equation,
                                                     efficiently estimated as a Gaussian Markov random field, and can be used for
                                                     prediction, inference, and causal analysis.
                                                 3. I demonstrate GGMMs using three contrasting case studies and a simulation ex-
                                                     periment, using a range of simple or complex software including an R-­package
                                                     (tinyVAST), base-­R code for simulating dynamics, and bespoke estimation code
                                                     using RTMB. Analysing age-­structured population dynamics using tinyVAST shows
                                                     that correlations by cohort can improve out-­of-­sample predictions. Simulating
                                                     spatial dynamics in R shows that diffusive movement can be represented (or
                                                     eliminated) in spatio-­temporal models by including correlations across space,
                                                     time, and a space–time interaction. Estimating separate Ornstein-­Uhlenbeck pa-
                                                     rameters (OU) in a phylogenetic structural equation model using RTMB shows
                                                     that geographic range has faster stabilizing selection than body size or specific
                                                     metabolism across mammals. Finally, a simulation experiment confirms that the
                                                     OU parameters can be better estimated while jointly interpolating missing data,
                                                     which is not feasible using existing R-­packages (phylosem or phylolm).




This is an open access article under the terms of the Creative Commons Attribution-­NonCommercial-­NoDerivs License, which permits use and distribution in
any medium, provided the original work is properly cited, the use is non-­commercial and no modifications or adaptations are made.
Published 2026. This article is a U.S. Government work and is in the public domain in the USA. Methods in Ecology and Evolution published by John Wiley & Sons
Ltd on behalf of British Ecological Society.

1290   | 	﻿
         wileyonlinelibrary.com/journal/mee3                                                                           Methods Ecol Evol. 2026;17:1290–1302.
                                                                                                                                             | 1291




                                                                                                                                                        2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
THORSON



                                              4. I conclude that GGMMs connect ecological theory with statistical models that
                                                 are applied for inference, prediction, and causal analysis throughout ecology. In
                                                 particular, GGMMs allow analysts to sequentially add (and test) mechanisms dur-
                                                 ing bespoke development of simulation or estimation models while also reverting
                                                 to common statistical models when appropriate.

                                              KEYWORDS
                                              diffusion, generalized additive model, generalized linear mixed model, mathematical graph,
                                              phylogenetic path analysis, species interactions, structural equation model




1    |   I NTRO D U C TI O N                                              models (GLMM; Bolker et al., 2009), generalized additive mod-
                                                                          els (GAM; Wood, 2017), and structural equation models (SEM;
Most ecological studies involve measuring, explaining, and predict-       Grace, 2006) as their core toolbox for statistical analysis. For ex-
ing dynamics across multiple locations, times, and variables. For ex-     ample, GAMs are widely used for estimating habitat utilization
ample, global conservation involves essential biodiversity variables      (Miller, 2025), GLMMs are used to control for pseudo-­replication in
(measurement of biomass across sites, times, and species), where          the analysis of experimental designs (Bolker et al., 2009), and there
statistical models are then used to fill in missing elements in this      is growing recognition that SEM (and variants like path analysis) have
three-­coordinate array (Jetz et al., 2019). Alternatively, macroevo-     a distinct role for causal analysis (Grace, 2024). However, ecologists
lution and paleoecology seek to identify how species traits change        might ultimately seek to combine elements across these tools (e.g.
over time among species subject to extinction and speciation events       incorporating random effects in causal analysis), and there is little
(Hautmann, 2020). Ecological subfields generally differ in how they       conceptual guidance for how GLMM, GAM, and SEM models can be
discretize space, time, and variables, but these ordinates remain         unified in a single theoretical framework.
ubiquitous across ecology.                                                    I therefore introduce a family of generalized graphical mixed mod-
    Ecologists then use these measurements across space, time, and        els (GGMMs) that connects ecological theory with these widely used
variables for three distinct tasks:                                       statistical models. A GGMM starts by defining a mathematical ‘graph’,
                                                                          where any combination of variables, times, and sites are defined as
1. Inference, that is, measuring the functions and parameters that        nodes (visualized as boxes), and nodes are then linked using directed
    give rise to ecological dynamics in observational and experimen-      edges (visualized as arrows) that represent linear structural interactions.
    tal systems, so that these parameters can then be compared            The edges are quantified using a sparse path matrix in a simultaneous
    with ecological theory or other measured values across space          equation, and developing a GGMM involves constructing this path ma-
    and time.                                                             trix from elementary ecological processes including spatial diffusion,
2. Prediction, that is, estimating the value for a system variable        time lags, ecological interactions among species, or evolutionary trade-­
    where it was not specifically measured, for example, to allow bio-    offs among traits. The resulting GGMM then includes GLMMs, GAMs,
    diversity variables to be compared with target values.                and SEM as nested submodels and is useful for inference, prediction,
3. Causal analysis, that is, estimating the value for a system variable   and causal analysis. I specifically emphasize two insights:
    under some hypothetical change, allows policy makers to com-
    pare the likely outcome of different potential management strate-     1. Graphs can be expressed using a sparse path matrix within
    gies and better understand mechanisms of the system.                      a simultaneous equation and efficiently fitted as a Gaussian
                                                                              Markov random field;
    These different tasks all involve some combination of experi-         2. Graphs can be assembled from elementary ecological relation-
mental and observational studies, but an analysis suitable for one            ships, such as time lags, spatial diffusion, evolutionary related-
task may not be suitable for another (Levins, 1966). For example, a           ness, and unstructured interactions. The joint path matrix is then
predictive model will often not be suitable for causal analysis (Arif         computed by summing across the interaction (Kronecker product)
& MacNeil, 2022), a model used to infer ecological parameters may             of the path matrices resulting from these different elementary
explain a small proportion of predictive variance, and a model with           relationships.
good in-­sample prediction may not provide any transferable infer-
ence about system parameters.                                                 I demonstrate these insights using three examples drawn from
    Given these contrasting goals, ecologists draw upon a large           population dynamics, movement ecology, and macroevolution, and
toolbox of mechanistic and statistical models. Here, I argue that         include a simulation experiment testing performance for the third
most ecologists use some combination of generalized linear (mixed)        example.
           |




                                                                                                                                                          2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
1292                                                                                                                                        THORSON



2      |       M ATE R I A L S A N D M E TH O DS                           k1 and k2. This ‘box and arrow’ visualization is widely used in path
                                                                           analysis (Wright, 1921), and I discuss the connections to causal
2.1 | Graphs, simultaneous equations, and                                  modelling in a later section.
random fields                                                                  This simultaneous equation (and associated graphical represen-
                                                                           tation) can be rearranged as a Gaussian Markov random field (GMRF;
In the following, I will construct a probabilistic graphical model         Rue & Held, 2005), where vec(B) follows a multivariate normal dis-
by defining elementary graphs corresponding to basic ecological            tribution such that the inverse covariance (‘precision’) matrix can be
theories (each of which is represented by a path matrix), comput-          constructed directly:
ing a joint path matrix by summing across the Kronecker product                                                    (        )
of these elementary path matrices, and then using that joint path                                  vec(B) ∼ MVN 0, Q−1
                                                                                                  (           )(    )−1 (           )               (2)
matrix to define a GMRF for latent variables that is fitted as a gener-
                                                                                               Q = I − PTjoint LT L      I − Pjoint
alized linear mixed model (GLMM). To explain the process in detail,
I first introduce mathematical graphs including their construction,        The probability density in the first line can be computed effi-
representation using simultaneous equations, and estimation using          ciently as long as joint path matrix Pjoint and exogenous variation L
Gaussian Markov random fields (GMRFs), corresponding to Insight            are sparse, that is, species primarily interact with a constrained set
#1 from the introduction (see Supporting Information 1 for more            of other species, within a localized neighbourhood, and where in-
mathematical details).                                                     teractions occur simultaneously or over a reduced set of lags (see
    As an illustrative example, I introduce the essential bio-             Supporting Information 1 for more details). Although Q is often
geographic variable B which contains biomass bs,c,t at each site           sparse, Q−1 will typically be dense, and therefore can represent
s ∈ {1, 2, … S}, species c ∈ {1, 2, … , C}, and time t ∈ {1, 2, … , T}.    covariance that propagates over large distances and/or time lags
Given S = 100 sites, C = 100 species, and T = 10 times, we obtain          (e.g. first-­order spatial diffusion results in an exponential correla-
an array B containing STC = 100,000 state-­variables representing          tion function across space, see Lindgren et al. (2011)). To identify
biomass for each site-­species-­t ime. Ecologists are then interested      parameters in Pjoint and L, we also require them to have constraints
in interactions among species, which might include non-­
                                                       l ocal              where for example, species interactions might be stationary
properties (i.e. source-­sink and predator avoidance behaviours) as        across space and/or time. These constraints show up where ele-
well as both simultaneous and lagged effects (e.g. where predator          ment qk2 ,k1 of precision Q will be equal to qk4 ,k3 for other sites, times,
c1 in time t1 consumes juvenile prey, and therefore has delayed            or variables.
impact on adult biomass for prey c2 in time t2 = t1 + Δt). In the
limit, non-­l ocal and lagged interactions can result in approximately
(SCT)2 ∕ 2 = 5 × 109 interactions (excluding interactions backwards        2.2 | Elementary graphs as building blocks
in time), so we obtain a staggering number of potential interac-
tions even using this relatively coarse discretization of space, time,     I next discuss how state-­variables B can be defined, and the joint
and taxonomy. I therefore seek some conceptually clear, computa-           path matrix Pjoint can then be constructed from elementary ecologi-
tionally efficient, and statistically justified simplification for these   cal relationships, such as time lags, spatial diffusion, evolutionary
non-­local and lagged interactions.                                        relatedness, and unstructured interactions (Insight #2 from the in-
    A graphical model represents each element of state-­variable B         troduction; see Figure 1).
as a node (a.k.a. vertex, visualized as a box), and represents linear          Constructing a graphical model first involves defining the
relationships as directed edges (visualized as arrows). This can then      state-­
                                                                                 variables and their resolution, which corresponds to the
be represented as a simultaneous equation:                                 value recorded, dimensions, and length along each dimension of
                                                                           the state-­variable B. As discussed previously, B might be a three-­
                        vec(B) = Pjoint vec(B) + vec(E)                    dimensional array for community abundance (Space × Time × Taxa),
                                           (       )                 (1)
                           vec(E) ∼ MVN 0, LT L                            where the analyst then decides how many species to include
                                                                           (or aggregate into functional groups) for Taxa, how many spatial
where Pjoint is the K × K ‘joint’ path matrix where K is the length        jurisdictions or habitat patches to include (or how to discret-
of vec(B), containing elements 𝜌k2 ,k1 that measure the effect of          ize continuous spatial coordinates) for Space, and how to aggre-
variable k1 on k2 (directed edges). Similarly, E is the array of ex-       gate time intervals (e.g. quarters, years or decades) in Time. I
ogenous variation representing processes that are not explicitly           demonstrate various two-­dimensional problems in the following
represented in the modelled interactions and LT L is the K × K co-         case studies, that is, the evolution of multiple traits (trait values
variance matrix for this unmodeled variation, represented using its        across Taxa × Trait), diffusive animal movement (densities across
square-­root L containing 𝜆k2 ,k1 (undirected edges). In its graphical     Space × Time), and age-­s tructured abundance (abundance across
representation, the joint path matrix Pjoint can then be visualized        Age × Time), and use these varied examples to show how larger
as one-­h eaded arrows where 𝜌k2 ,k1 points from k1 to k2, and the         dimensions can be assembled from a small number of elementary
exogenous variance L as two-­h eaded arrows where 𝜆k2 ,k1 connects         processes.
                                                                                                                                                      | 1293




                                                                                                                                                                2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
THORSON




F I G U R E 1 Schematic visualizing the workflow for specifying a generalized graphical mixed model for a hypothetical example involving
three species (resource, consumer, and predator) over eight sites and five time intervals. This workflow involves: (1) defining the dimensions
for the state-­variable B; (2) hypothesizing mechanisms; (3) constructing elementary path matrices; (4) assembling the joint path matrix given
hypotheses and their associated path matrices; and (5) evaluating the probability of the state-­variable as a Gaussian Markov random field.
See Figure 2 for more visualization regarding Step-­3.



   A graphical model can then be constructed by hypothesizing a                      space into habitat patches (e.g. Hanski et al., 1994), and spatial
set of elementary processes that specify a linear relationship among                 adjacency and diffusion (and resulting spatial autocorrelation) can
values (nodes) in response B. In the following, I focus on continu-                  also be defined in this context;
ous response values (i.e. densities, continuous traits), and emphasize            3. Evolutionary dynamics: Ecologists often study evolutionary dy-
four elementary processes (see Figure 2), where each corresponds                     namics along a lineage (a phylogeny for species or a pedigree for
to a simultaneous equation and an associated path matrix P that is                   individuals). A phylogeny is often represented as a tree, wherein a
common in ecological analysis:                                                       parent (individual or taxon) will split into two or more descendants
                                                                                     and the evolutionary path matrix Pphylogeny is non-­zero for each
1. Time-­lagged dynamics: Variable bt can often be predicted                         pair of descendant (row) and ancestor (column). Genetic drift oc-
   from its first-­
                  order lag (bt−1) and higher order lags (bt−2, etc.)                curring within a quadratic fitness landscape will result in a stabiliz-
   (Turchin, 2003). The first-­order lag can be computed using lag-­1                ing selection towards the fitness peak (Lande, 1976);
   matrix Plag1, defined such that such that 𝜌t2 ,t1 = 1 if t2 = t1 + 1           4. Interactive dynamics:      Finally,   ecologists   are   often    inter-
   and 0 otherwise (i.e. a sparse banded matrix). A lag-­2 matrix                    ested in structural linkages among variables. For example, a
   is then calculated as Plag2 = Plag1 2;                                            trophic cascade arises from two negative species interactions
2. Diffusive dynamics: Ecological variables are often more similar                   Predator → Consumer and Consumer → Producer, where the prod-
   when they are close together (Tobler, 1970), and this spatial                     uct of these two negative direct effects results in a positive in-
   autocorrelation can arise as animals or their physical habitat                    direct effect from predators to producers. Interactions can then
   undergoes diffusion (Lindgren et al., 2011). In two dimensions                    be used to construct the path matrix Pinteraction with whatever
   and    discretizing   space     into   square    grid    cells,    diffusion      pattern is hypothesized. Representing a species interaction as a
   results in a location s = (x, y) affecting its four neighbours                    linear effect (i.e. such that a change in predator density causes a
   {(x, y + 1), (x + 1, y), (x, y − 1), (x − 1, y)} such   that      each   row      𝜌prey,predator impact on prey density) can be viewed as a first-­order
   of Pdiffusion is non-­zero for only four ‘adjacent’ elements, and                 Taylor series approximation to a more general (e.g. Lotka-­Volterra)
   Pdiffusion then serves as the ‘weight matrix’ in a simultaneous                   function. Ecologists often represent interactions within smaller
   autoregressive spatial model (Ver Hoef et al., 2018). Alternatively,              tightly-­coupled ‘species modules’ (Holt, 1997), so representing
   metapopulation and metacommunity models often discretize                          species interactions will often result in Pinteraction being sparse.
       |




                                                                                                                                                     2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
1294                                                                                                                                   THORSON




F I G U R E 2 Graphs (left column) representing common simplifying assumptions for ecological dynamics, and the path matrix P (right
column) in simultaneous equation y = Py + 𝛜 that results from each graph (with grey box when 𝜌i,j ≠ 0 corresponding to graph arrows, and
white boxes where 𝜌i,j = 0), showing first-­order autoregressive dynamics (top row), spatial diffusion from a central location (x, y) when using
square boxes to discretize a spatial domain in two dimensions with four adjacent grid cells (2nd row), a dated phylogeny (3rd row) showing
ancestral nodes {s5, s6, s7} and extant species {s1, s2, s3, s4}, and interactions among four variables A → B, A → C , B → D, and C → D (4th row).
                                                                                                                                                                | 1295




                                                                                                                                                                         2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
THORSON



    Other elementary graphs could also be developed for specific                      increasingly familiar with developing bespoke code using high-­level
state-­variables, e.g. a growth-­transition matrix Pgrowth when state-­               code such as JAGS, STAN, or NIMBLE (Kéry & Kellner, 2024), so I
variable B represents abundance by size and time.                                     recommend greater emphasis on developing bespoke GGMMs via a
    These elementary relationships can then be combined to struc-                     progression similar to that presented here.
ture a larger multivariate model (see Figure 1). For example, an
analyst might specify interactive dynamics where predator X af-
fects prey Y and prey Y affects consumer Z (Graph-­4 above) and                       2.3 | Case study 1: Tracking cohorts in
where all taxa exhibit spatial diffusion (Graph-­
                                                2). This involves                     age-­structured demographics
two interactions 𝜌X→Y and 𝜌Y→Z in 3 × 3 matrices PXY and PYZ where
Pinteraction = 𝜌X→Y PXY + 𝜌Y→Z PY→Z, and an S × S matrix Pdiffusion while             As a first example, ecologists are often interested in predicting
estimating the strength of diffusion 𝜅, where the joint path matrix is:               abundance at age na,t for A ages and T years, which arises via survival
                                                                                      from the preceding age and year na−1,t−1. However, na,t might vary
                      (                       )     (                       )
         Pjoint = 𝜌XY PXY ⊗ 𝜅Pdiffusion + 𝜌YZ PYZ ⊗ 𝜅Pdiffusion                       for all ages in a single year and therefore be predicted from na−1,t,
                  ����������������������������� �����������������������������         or it might be affected by changes in survey availability for an age
                       diffusive effect of            diffusive effect of       (3)
                     predator on consumer           consumer on producer              that is consistent across years (i.e. predictable from na,t−1). I therefore
                                                                                      explore a model with three interactions, arising from a lag-­1 process
where C = A ⊗ B is the Kronecker product of two matrices, such that                   among years and a separate lag-­1 process among ages which I call
resulting matrix C has dimensions a1 b1 × a2 b2 when matrix A has di-                 GYear and GAge, respectively, to distinguish the two versions of the
mension a1 × a2 and matrix B has dimension b1 × b2. Therefore, Pjoint is              lag-­1 matrix Plag1:
the 3S × 3S matrix arising from three parameters (two interactions and
                                                                                                 (               )   (               )   (                 )
one diffusion rate), given that interaction matrix are 3 × 3 and Pdiffusion           Pjoint = 𝜌1 GAge ⊗ IYear + 𝜌2 IAge ⊗ GYear + 𝜌3 GAge ⊗ GYear
                                                                                               ������������������� ������������������� ��������������������� (4)
has dimension S × S. As shown in Equation (3), the joint path matrix is                             na−1,t → na ,t      na,t−1 → na ,t       na−1,t−1 → na ,t
constructed via the sum of Kronecker products of the elementary path
matrices (see general form in Supporting Information 1:Equation S1.14).               where GAge is the A × A lag-­1 matrix Plag1 among A ages, IAge is the A × A
    More generally, many ecological and evolutionary processes                        identity matrix, and GYear and IYear are the corresponding T × T lag-­1
can be represented by combining time lags (Plag1), diffusive spatial                  and identity matrices across years (see Supporting Information 2 for
dynamics (Pdiffusion), phylogenetic relationships (Pphylogeny), and in-               more details). I fit this model to proportional abundance-­at-­age for rex
teractions (Pinteraction), beyond the illustrative examples we have                   sole in the Gulf of Alaska, which was sampled intermittently from 1992
discussed so far. For example, combining species interactions with                    to 2022 (McGilliard, 2024). I specify a log-­linked Tweedie distribution
species-­specific diffusion rates (e.g. extending Equation 3) results                 (Foster & Bravington, 2013) for measurement errors, which provides a
in emergent spatial patterns wherein species are spatially clustered                  convenient distribution for compositional data in bottom trawl surveys
even in a homogenous environment (Levin, 1974; Turing, 1952),                         (Thorson, Monnahan, & Hulson, 2023), and fit the model using package
and can therefore represent a ‘neutral model’ for spatial patterns in                 tinyVAST (Thorson et al., 2025) release 1.4.0 in the R statistical envi-
meta-­community theory (Leibold et al., 2004). Similarly, vector au-                  ronment, which uses TMB and the Laplace approximation (Kristensen
toregressive (VAR) models are constructed by combining time lags                      et al., 2016; Skaug & Fournier, 2006) to marginalize across unmeasured
and species interactions (e.g. Thorson et al., 2024), and have been                   state-­variables. I then use 10-­fold cross-­validation to compare parsi-
widely used to predict how species interactions affect community                      mony among the eight models arising from estimating or fixing at zero
                                                                                                          {            }
stability (Ives et al., 2003; Wootton & Emmerson, 2005). Past stud-                   the three parameters 𝜌1 , 𝜌2 , 𝜌3 from Equation 4, which represent the
ies have also extended VAR models to represent spatial autocorrela-                   relative importance of within-­cohort, within-­year, and within-­age driv-
tion (i.e. combining time lags, interactions, and diffusion), and these               ers for observed abundance-­at-­age (see Data accessibility statement
spatial VARs have been used to study the community impact of in-                      to access the R-­script).
vasive insects or recovering predators (Schliep et al., 2018; Thorson
et al., 2017). Representing community dynamics as a network across
space and species using phylogenetic and trait similarity remains an                  2.4 | Example 2: Diffusion-­enhanced
active area of research (García-­C allejas et al., 2025).                             spatio-­temporal models
    To further illustrate, I next introduce how these simultaneous
equations (and associated graphs) arise in three varied ecological                    Ecologists have a long-­running interest in spatial ecology including
examples (Thorson, 2026). These examples represent different sub-                     the trajectory and rate of expanding range edges for invasive species.
fields (population dynamics, spatial ecology, and macroevolution),                    Spatial statisticians have developed non-­separable spatio-­temporal
but are also organized to proceed from relatively simple code (an                     models that incorporate diffusive dynamics (Lindgren et al., 2023),
R-­package tinyVAST for estimation), to intermediate (bespoke sim-                    but these see little use in ecology to date. Here, I present a novel
ulation code in base-­R), and then to full complexity (bespoke esti-                  demonstration that graphical models can represent both separable
mation code using RTMB). Ultimately, I believe that ecologists are                    and diffusion-­enhanced spatio-­temporal dynamics using an additive
       |




                                                                                                                                                                      2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
1296                                                                                                                                                     THORSON



path matrix, resulting from a lag-­1 matrix Plag1 in time and a spatial                                                     ⎡Q      0    0⎤
diffusion matrix Pdiffusion:                                                                                                ⎢ 1              ⎥
                                                                                                                            ⎢                ⎥
                                                                                                               Qphylogeny = ⎢ 0     Q2   0⎥                     (7)
           (                     )   (                 )   (                       )                                        ⎢                ⎥
Pjoint = 𝜌1 Itime ⊗ Pdiffusion + 𝜌2 Plag1 ⊗ Ispace + 𝜌3 Plag1 ⊗ Pdiffusion                                                  ⎢0      0    Q3 ⎥⎦
                                                                                                                            ⎣
         ������������������������� ��������������������� ���������������������������
                ds,t → ds+1,t            ds,t → ds,t+1            ds,t → ds+1,t+1

                                                                                    (5)   where Q1, Q2, and Q3 are the evolutionary precisions matrices given
                                                                                          OU parameters 𝜃 1, 𝜃 2, and 𝜃 3 for log-­size, log-­metabolism, and log-­
I demonstrate this using a deterministic simulation by visual-                            range respectively. I fit this model using a dated phylogeny across
izing the density matrix D resulting from diffusive dynamics                              5911 mammal species and 185 million years of evolutionary history
         (         )−1
vec(D) = I − Pjoint vec(E), where vec(E) is an indicator vector such                      (Upham et al., 2019), where Pphylogeny,j is the 11821 × 11821 matrix
that Escenter ,1 = 1 for location scenter at the centre of the spatial domain             across all tips and ancestral nodes. I specifically fit the model using
in time t = 1 and Es,t = 0 elsewhere. I visualize this diffusive process                  bespoke code in RTMB (Kristensen, 2024), which allows analysts to
over a square spatial domain discretized into 21 rows and 21 columns                      write bespoke GGMMs by defining elementary graphs in R for effi-
(S = 441 square grid cells) and T = 3 times, while fixing 𝜌1 = 0.8 and                    cient computation involving sparse matrices (see Data accessibility
𝜌2 = 0.1, and varying the value of 𝜌3. Diffusive dynamics are expected                    statement for more details to access code). I then record the esti-
to result in a linear increase in the mean-­square displacement for                       mated OU parameters (𝜃 1, 𝜃 2, and 𝜃 3 and associated standard errors),
the utilization distribution over time (see Supporting Information 3                      which represent the strength of stabilizing selection for each trait
for more details). I demonstrate this as a simulation experiment (not                     (Lande, 1976).
conditioned upon fitting data) using R code, including the Matrix
package (Bates et al., 2023) to interface with Eigen (Guennebaud &
Jacob, 2010) for efficient computation involving sparse matrices (see                     2.6 | Simulation experiment
Data accessibility statement).
                                                                                          Finally, I provide a simulation experiment to demonstrate how
                                                                                          GGMMs compare with existing ecological software, specifically
2.5 | Case study 3: Phylogenetic trait imputation                                         emphasizing how they can (1) represent ecological mechanisms
with varying stabilization rates                                                          flexibility while (2) usefully propagating uncertainty. To do so, I use
                                                                                          the fitted model from Case Study #3 and simulate 200 replicated
Ecologists also study how traits covary among natural populations,                        data sets from each of three scenarios, representing high, medium,
seeking to identify evolutionary trade-­offs that arise from adapta-                      or low data availability (i.e. where each trait is independently
tion to shared fitness constraints. Recent research has developed                         missing measurements for 50%, 70%, or 90% of species). For each
phylogenetic SEM from the Kronecker product of trait-­interactions                        of 600 simulated data sets, I then fit the GGMM, and compare it
and a single evolutionary matrix that is shared across traits (Thorson,                   with either:
Maureaud, et al., 2023; Thorson & van der Bijl, 2023), but this does
not allow different traits to have different rates of stabilizing selec-                  1. Phylogenetic linear models: fitting three PLM using R-­
                                                                                                                                                   package
tion. I therefore present a novel extension, where I calculate the joint                     phylolm (Tung Ho & Ané, 2014), either treating log-­body size
precision from a simultaneous equation that includes phylogenetic                            as response with no predictor (log(size) ∼ 1), log-­metabolism as
path matrix Pphylogeny and an interaction matrix Pinteraction. This results                  response with log-­size as predictor (log(metabolism) ∼ log(size)),
in joint precision:                                                                          or   log-­
                                                                                                      area    as   response       with   log-­
                                                                                                                                             size   as   predictor
                                                                                             (log(area) ∼ log(size)). However, phylolm cannot jointly conduct
                           (           )          (           )
                   Qjoint = I − PTjoint Qphylogeny I − Pjoint                       (6)      inference (estimate parameters) and prediction (impute missing
                                                                                             values), so each model is fitted only to those species that
where Pjoint = I ⊗ Pinteraction, Pinteraction is the C × C matrix of interac-                have all relevant traits measured (a ‘pairwise complete data’).
tions among traits, and I is the S × S identity matrix, and Qphylogeny is                    For each model, I then record the estimated OU parameter.
the block-­diagonal matrix of evolutionary precisions for each trait (see                 2. Phylogenetic structural equation models: fitting a single phylogenetic
Supporting Information 4 for more details).                                                  SEM using R-­package phylosem (Thorson & van der Bijl, 2023), and
    To illustrate, I download three traits from PanTHERIA (Jones                             using the same path diagram as the GGMM. However, phylosem
et al., 2009), representing specific metabolic rate (mL O2/g), adult                         is restricted to assuming a single OU parameter across all three
body mass (g), and home range size (km2). Body size has the high-                            traits (a ‘separable’ precision matrix). I therefore record the
est proportion of data (3340 measurements), while other traits have                          imputed value for traits.
fewer measurements (Supporting Information 5, Table S1). I specify a
phylogenetic SEM with two interactions log(size) → log(metabolism)                           I then compare the error in estimated OU parameters between
and log(size) → log(range). I also estimate the Ornstein-­Uhlenbeck                       the GGMM and phylolm, and the imputed traits between the GGMM
(OU) parameter 𝜃 c for each trait, used to calculate:                                     and phylosem. Finally, I also calculate the z-­scored error from GGMM
                                                                                                                                                 | 1297




                                                                                                                                                            2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
THORSON


         ̂
(i.e. z = ̂𝜃 (− 𝜃) where ̂
                         𝜃 is the estimated value, 𝜃 the simulated (true)     stronger than year effects (𝜌2 = 0.3), and strong cohorts are also vis-
          SE ̂ 𝜃
                   ( )
value, and SE       𝜃 the estimated standard error), where a well-­
                  ̂ ̂                                                         ually apparent starting at age-­7 around 2005 and again in 2011, and

performing estimator with have z-­scored errors that (approximately)          progress visually through subsequent ages and years. Leave-­year-­out

follow a normal distribution.                                                 cross-­validation (Supporting Information 5, Figure S1) confirms that
                                                                              including these interactions can result in skillful predictions for years
                                                                              without direct measurements.
                                                                                  The GGMM simulating diffusion-­
                                                                                                                enhanced spatio-­
                                                                                                                                temporal
3    |   R E S U LT S                                                         dynamics in R (Figure 4) shows that diffusion arises by combining
                                                                              interactions across time (ns,t → ns,t+1 using 𝜌time) and interactions
In the GGMM fitting to proportional abundance-­at-­age for rex sole           across spatial neighbours (ns,t → ns+1,t using 𝜌space) (Figure 4 top
in the Gulf of Alaska using an R-­package (tinyVAST), 10-­fold cross-­        row). In this scenario, mean-­squared displacement (MSD) shows
validation indicates that the model with interactions along cohorts           a close-­to-­linear increases over time with rate 𝜌space, as expected
(𝜌1 = na,t → na+1,t+1) and along years (𝜌2 = na,t → na,t+1) has lowest pre-   given diffusive dynamics (where the departure from a linear in-
dictive error (Supporting Information 5, Table S2). The fitted model          crease in MSD arises from spatial boundary effects). The model
(Figure 3) estimates that cohort effects (𝜌1 = 0.73) are substantially        then reverts to separable spatio-­temporal dynamics when add-
                                                                              ing a parameter lagged spatial effect (ns,t → ns+1,t+1 using 𝜌spacetime)
                                                                              and fixing 𝜌spacetime = − 𝜌space 𝜌time (Figure 4 bottom row). Finally,
                                                                              intermediate dynamics arise when − 𝜌space 𝜌time < 𝜌spacetime < 0. For
                                                                              example, when 𝜌spacetime = − 0.5𝜌space 𝜌time the MSD starts at 0.1
                                                                              but then increases 0.05 per time interval (Figure 4 middle row).
                                                                              Importantly, 𝜌spacetime allows for a continuous bridge between two
                                                                              ecological hypotheses, where a hotspot remains stationary or dif-
                                                                              fuses outwards over time.
                                                                                   The GGMM using RTMB to estimate interactions among adult
                                                                                         [    ]                            (     )
                                                                              body size ln(g) , specific metabolic rate [ln mL g2 ], and range size
                                                                                                                              0
                                                                                 (    )
                                                                              [ln km2 ] for 5911 mammal species over 185 million years (Figure 5)
                                                                              estimates a isometric (𝜌 = 1.00) scaling of range size with adult body
                                                                              size, and an allometric (𝜌 = 0.69) scaling of metabolic rate with body
                                                                              size. Additionally, it estimates weakest stabilizing selection for body
                                                                              size, with a nearly 20% correlation between two species separated by
                                                                              150 million years of divergent evolution. By contrast, range size has
                                                                              strongest stabilizing selection (conditional upon body size), where
                                                                              a 20% correlation arises at approximately 25 million years of diver-
                                                                              gence. Finally, specific metabolism has an intermediate strength for
                                                                              stabilizing selection (20% correlation at 60 million years), conditional
                                                                              upon body size.
                                                                                  Finally, the simulation experiment shows that the GGMM and
                                                                              PLM both provide (approximately) unbiased estimates of the log-­OU
                                                                              parameters (representing the strength of stabilizing selection) for all
                                                                              three traits (Figure 6 top row, where error distributions are centred
                                                                              on zero). However, the PLM cannot jointly estimate parameters and
                                                                              impute missing traits, and therefore is restricted to pairwise complete
                                                                              data. As a result, it has much larger errors (i.e. wider intervals) for the
                                                                              strength of stabilizing selection for specific metabolism and range size
                                                                              when 70%–90% of taxa are missing trait measurements. Alternatively,
                                                                              the GGMM and phylogenetic structural equation model (PSEM) have
F I G U R E 3 The estimated interactions (top panel) when                     very similar levels of error when imputing missing traits, suggesting
predicting proportional abundance at age na,t for rex sole in the Gulf        that pooling the OU parameter across traits has little impact on im-
of Alaska showing the estimated effect of survival along a cohort 𝜌3          puted trait estimates. Finally, the z-­scored errors in estimated log-­OU
and effects along a year 𝜌2 (see Equation 4), as well as the observed
                                                                              parameters (Figure S2) confirms that the standard errors provided
na,t (middle panel) for each year (x-­axis) and age (y-­axis) showing
low (purple) to high (yellow) values, and the estimated na,t (bottom          by GGMM are generally reasonable (i.e. the simulation distribution
panel) including the estimated value for years with no direct                 matches the desired normal distribution). However, there appears
sampling (white spaces in the middle panel).                                  to be some positive skewness for the simulation distribution (e.g.
           |




                                                                                                                                                       2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
1298                                                                                                                                     THORSON




F I G U R E 4 Visualizing diffusion-­enhanced spatio-­temporal dynamics (top row), intermediate dynamics (middle row), and non-­diffusive
(separable) spatio-­temporal dynamics (bottom row), including the graph (1st column) linking a focal cell (s, t) and adjacent cells in a given
time (s + 1, t), the same cell in the next time (s, t + 1), or adjacent cells in the next time (s + 1, t + 1). Separable dynamics arise when
𝜌s+1,t+1 = − 𝜌s,t+1 𝜌s+1,t (bottom row). I also visualize resulting dynamics from a concentrated density (purple is 0 density, yellow is high
density) in time t = 1 (2nd column), and how this density evolves in times 2 (3rd column) and 3 (4th column). For each panel, I also calculate
the mean-­squared displacement (MSD) as the variance of the density function. Diffusive dynamics results in MSD increasing linearly over
time, although the diffusive MSD in time-­3 is slightly lower due to boundary effects.


bottom-­right panel of Figure S2), suggesting that a likelihood profile    size within the mammal lineage. Finally, a simulation experiment
might be more appropriate than assuming that errors are normally           confirmed that the GGMM can improve upon existing software (e.g.
distributed.                                                               phylogenetic linear and SEM) by jointly imputing missing traits and
                                                                           estimating separate stabilizing selection for each trait, and that it can
                                                                           usefully propagate estimation uncertainty.
4      |       DISCUSSION                                                     As shown in our second case study (diffusion-­enhanced spatio-­
                                                                           temporal models), GGMMs can be used as a spatial smoother simi-
I introduced the family of GGMMs, which represent variables as             lar to GAMs. In particular, the diffusion-­enhanced spatio-­temporal
nodes and interactions as directed edges and are efficiently fitted        model constructs a joint precision matrix from structural parame-
as a GMRF. In particular, ecological variables are often indexed           ters representing spatial diffusion, temporal autocorrelation, and
by space, time, and category (e.g. species or age). Ecological             the space–time interaction (see Figure 4). This precision matrix then
interactions in GGMMs can be specified by combining several                serves a similar function to the penalty matrix that is used to mini-
elementary graphical structures representing time lags, phylogenetic       mize ‘wiggliness’ in a GAM (Miller, 2025). However, the GGMM has
relatedness, spatial diffusion, ecological interactions among species,     two added benefits relative to a GAM: (1) it derives the precision
or evolutionary trade-­offs among species traits. I then construct         matrix from ecological interpretable parameters where for example,
the joint path matrix for the GMRF from the sum of Kronecker               the space–time interaction parameter 𝜌3 determines whether spa-
products of the path matrices for elementary graphs. Using three           tial hotspots propagate outwards in space over time; and (2) instead
varied case studies, I specifically showed that interactions among         of using the path matrix to construct a precision (penalty) matrix,
elementary graphical structures can represent population dynamics          the path matrix can be repurposed to represent how animal diffu-
(age structure), movement dynamics (diffusion-­
                                              enhanced spatio-­            sion affects, for example, the association between animal densities
temporal variation), and evolutionary dynamics (stabilizing selection      and habitat (Lindmark et al., 2026). Deriving the joint path matrix
among traits). This then yielded novel insights, for example, that         from ecological mechanisms and using this to define a penalty ma-
stabilizing selection is stronger for adult home range than body           trix therefore reveals a rich class of new dynamics for ecological
                                                                                                                                             | 1299




                                                                                                                                                        2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
THORSON



                                                                            models, for example, when estimating non-­local and lagged impacts
                                                                            of habitat changes resulting from diffusive and lagged interactions.
                                                                                Similarly, there is growing interest in using structural causal
                                                                            models to re-­interpret a wide range of ecological analyses (Arif &
                                                                            MacNeil, 2022; Byrnes & Dee, 2025; Grace, 2024). Usefully, GGMMs
                                                                            predict the covariance across space, time, and categories by con-
                                                                            structing the path matrix as the sum across structural interaction
                                                                            like spatial diffusion, time lags, and interactions among species.
                                                                            Therefore, GGMMs can be used in the causal modelling workflow,
                                                                            that is, developing a graph from scientific knowledge, testing whether
                                                                            it can be rejected as being inconsistent with available data, and sub-
                                                                            sequently using it to explore the potential impact of policy changes
                                                                            due to both direct and indirect effects (summarizing fig. 2 from Arif
                                                                            & MacNeil, 2023). Usefully, GGMMs allow us to explicitly formulate
                                                                            (and communicate) scientific assumptions about causal relationships
                                                                            (represented by the path matrix), and how these then result in mod-
                                                                            elled correlations (the inverse of the precision matrix). Ultimately, I
                                                                            agree with Ives (2022) that ecological understanding at large spatial,
                                                                            temporal, and taxonomic scales requires a synthesis of observational
                                                                            and experimental methods, and hope that GGMMs provide a useful
                                                                            path to integrate causal (SEM) and descriptive (GLMM and GAM)
                                                                            statistics.
                                                                                In summary, I see GGMMs as a useful avenue to integrate eco-
                                                                            logical theory (i.e. specific ecological interactions across space, time,
F I G U R E 5 The estimated interaction (top panel) among three             and eco-­evolutionary variables) with statistical estimation (hierar-
log-­transformed traits (adult body mass [g], basal metabolic rate
                                                                            chical modelling tools). The models can be fitted using a range of
[mL O2/hour], and range size [km2]) for 4999 mammal species (with
                                                                            existing R-­packages (e.g. tinyVAST), simulated within R, or fitted to
3340, 661, and 547 available measurements respectively), as well
as the estimated correlation over time for residual patterns (bottom        new dynamics using bespoke code in RTMB. By providing a unified
panel) showing the correlation (y-­axis) over 185 million years (x-­axis)   framework across ecological and evolutionary analyses, I hope that
of evolutionary history for mammals (shaded interval shows the              GGMMs will allow researchers to move more easily between predic-
95% confidence interval).                                                   tive, inferential, and causal analyses.




F I G U R E 6 Estimation performance in a simulation experiment involving 200 replicates per scenario, where data are simulated based on
estimated values from real-­world data for the third case study (see Figure 4), comparing performance using the generalized graphical mixed
model (GGMM, red), an alternative model that assumes that evolutionary rate is constant among traits (PSEM using package phylosem,
green), or a model that estimates separate rates but cannot impute missing values during inference (PLM using package phylolm, blue). I show
the distribution of errors (y-­axis) in estimates of the log-­Ornstein Uhlenbeck parameter (top row; PSEM is not shown because it incorrectly
assumes a constant value across traits) or the average root mean squared error for estimated trait values for each individual taxon (bottom
row; PLM is not shown because it does not impute missing values) for three traits (columns) and three simulation scenarios where trait
measurements are missing for 50%, 70%, or 90% of species.
       |




                                                                                                                                                                     2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
1300                                                                                                                                                 THORSON



AU T H O R C O N T R I B U T I O N S                                                    and Ecological Statistics, 20(4), 533–552. https://​doi.​org/​10.​1007/​
James T. Thorson completed all stages of the research and writing.                      s1065​1-­​012-­​0233-­​0
                                                                                  García-­C allejas, D., Lavorel, S., Ovaskainen, O., Peltzer, D. A., &
                                                                                        Tylianakis, J. M. (2025). Species traits and landscape structure
AC K N OW L E D G E M E N T S                                                           can drive scale-­dependent propagation of effects in ecosystems.
I thank Han Wu, who is continuing research related to the diffusion-­                   Nature Communications, 16(1), 7998. https://​doi.​org/​10.​1038/​
enhanced model presented here, and Carey McGilliard, who devel-                         s4146​7-­​025-­​63208​-­​5
                                                                                  Grace, J. B. (2006). Structural equation modeling and natural systems.
oped the rex sole proportional abundance-­at-­age explored here. I
                                                                                        Cambridge University Press.
also thank Chris Wikle and Jonathan Bradley for earlier discussions               Grace, J. B. (2024). An integrative paradigm for building causal knowl-
of multivariate spatio-­temporal analysis. The manuscript was im-                       edge. Ecological Monographs, 94(4), e1628. https://​doi.​org/​10.​
proved by comments from Cole Monnahan and Lindsay Campbell, as                          1002/​ecm.​1628
                                                                                  Guennebaud, G., & Jacob, B. (2010). Eigenv3. https://​libei​gen.​gitlab.​io/​
well as three anonymous reviewers.
                                                                                  Hanski, I., Kuussaari, M., & Nieminen, M. (1994). Metapopulation struc-
                                                                                        ture and migration in the butterfly Melitaea Cinxia. Ecology, 75(3),
F U N D I N G I N FO R M AT I O N                                                       747–762. https://​doi.​org/​10.​2307/​1941732
The author has no funding information to report.                                  Hautmann, M. (2020). What is macroevolution? Palaeontology, 63(1),
                                                                                        1–11. https://​doi.​org/​10.​1111/​pala.​12465​
                                                                                  Holt, R. D. (1997). Community modules. Multitrophic Interactions in Terrestrial
C O N FL I C T O F I N T E R E S T S TAT E M E N T                                      Ecosystems, 36th Symposium of the British Ecological Society, 333–349.
The author has no conflict of interest to report.                                 Ives, A. R. (2022). Random errors are neither: On the interpretation of
                                                                                        correlated data. Methods in Ecology and Evolution, 13(10), 2092–
                                                                                        2105. https://​doi.​org/​10.​1111/​2041-­​210X.​13971​
DATA AVA I L A B I L I T Y S TAT E M E N T
                                                                                  Ives, A. R., Dennis, B., Cottingham, K. L., & Carpenter, S. R. (2003).
All code and data required to replicate analyses and figures are
                                                                                        Estimating community stability and ecological interactions from
available via GitHub (https://​github.​com/​James​-­​Thors​on/​graph​ical_​             time-­series data. Ecological Monographs, 73(2), 301–330.
mixed_​model/​​) using release 1.0.0, with DOI provided by Zenodo                 Jetz, W., McGeoch, M. A., Guralnick, R., Ferrier, S., Beck, J., Costello,
(https://​zenodo.​org/​recor​ds/​18509086) (Thorson, 2026). The mam-                    M. J., Fernandez, M., Geller, G. N., Keil, P., Merow, C., Meyer, C.,
                                                                                        Muller-­Karger, F. E., Pereira, H. M., Regan, E. C., Schmeller, D. S., &
mal phylogeny was downloaded from VertLife (https://​vertl​ife.​org/​
                                                                                        Turak, E. (2019). Essential biodiversity variables for mapping and
phylo​subse​t s/​) and was developed by Upham et al. (2019). The mam-                   monitoring species populations. Nature Ecology & Evolution, 3(4),
mal traits were accessed from PanTHERIA (Jones et al., 2009), avail-                    539–551. https://​doi.​org/​10.​1038/​s 4155​9-­​019-­​0 826-­​1
able online from ESA archives (https://​esapu​bs.​org/​archi​ve/​ecol/​           Jones, K. E., Bielby, J., Cardillo, M., Fritz, S. A., O'Dell, J., Orme, C. D.
                                                                                        L., Safi, K., Sechrest, W., Boakes, E. H., Carbone, C., Connolly, C.,
E090/​184/​metad​ata.​htm). The proportional abundance-­at-­age data
                                                                                        Cutts, M. J., Foster, J. K., Grenyer, R., Habib, M., Plaster, C. A.,
for rex sole in the Gulf of Alaska is publicly available (https://​github.​             Price, S. A., Rigby, E. A., Rist, J., … Purvis, A. (2009). PanTHERIA:
com/​n oaa-­​afsc/​goa_​rex/​b lob/​main/​r uns/​2025_​cie_​review/​2021_​              A species-­level database of life history, ecology, and geography of
accep​ted_​model_​inputs/​GOA_​Rex_8_​2021.​dat) from the 2024                          extant and recently extinct mammals. Ecology, 90(9), 2648. https://​
                                                                                        doi.​org/​10.​1890/​0 8-­​1494.​1
stock assessment (McGilliard, 2024) and distributed for a Center for
                                                                                  Kéry, M., & Kellner, K. F. (2024). Applied statistical modelling for ecologists:
Independent Experts 2025 review of the rex sole assessment.                             A practical guide to Bayesian and likelihood inference using R, JAGS,
                                                                                        NIMBLE, Stan and TMB. Elsevier.
ORCID                                                                             Kristensen, K. (2024). RTMB: “R” Bindings for “TMB”. https://​CRAN.​R-­​
                                                                                        proje​c t.​org/​packa​ge=​RTMB
James T. Thorson      https://orcid.org/0000-0001-7415-1010
                                                                                  Kristensen, K., Nielsen, A., Berg, C. W., Skaug, H., & Bell, B. M. (2016).
                                                                                        TMB: Automatic differentiation and Laplace approximation. Journal
REFERENCES                                                                              of Statistical Software, 70(5), 1–21. https://​doi.​org/​10.​18637/​​jss.​
Arif, S., & MacNeil, M. A. (2022). Predictive models aren't for causal in-              v070.​i 05
      ference. Ecology Letters, 25(8), 1741–1745. https://​doi.​org/​10.​1111/​   Lande, R. (1976). Natural selection and random genetic drift in pheno-
      ele.​14033​                                                                       typic evolution. Evolution, 30, 314–334.
Arif, S., & MacNeil, M. A. (2023). Applying the structural causal model           Leibold, M. A., Holyoak, M., Mouquet, N., Amarasekare, P., Chase, J. M.,
      framework for observational causal inference in ecology. Ecological               Hoopes, M. F., Holt, R. D., Shurin, J. B., Law, R., Tilman, D., Loreau,
      Monographs, 93(1), e1554. https://​doi.​org/​10.​1002/​ecm.​1554                  M., & Gonzalez, A. (2004). The metacommunity concept: A frame-
Bates, D., Maechler, M., & Jagan, M. (2023). Matrix: Sparse and dense                   work for multi-­scale community ecology. Ecology Letters, 7(7), 601–
      matrix classes and methods. https://​CRAN.​R-­​proje​c t.​org/​packa​ge=​         613. https://​doi.​org/​10.​1111/j.​1461-­​0248.​2004.​0 0608.​x
      Matrix                                                                      Levin, S. A. (1974). Dispersion and population interactions. The American
Bolker, B. M., Brooks, M. E., Clark, C. J., Geange, S. W., Poulsen, J. R.,              Naturalist, 108(960), 207–228. https://​doi.​org/​10.​1086/​282900
      Stevens, M. H. H., & White, J. S. S. (2009). Generalized linear mixed       Levins, R. (1966). The strategy of model building in population biology.
      models: A practical guide for ecology and evolution. Trends in                    American Scientist, 54, 421–431.
      Ecology & Evolution, 24(3), 127–135.                                        Lindgren, F., Bakka, H., Bolin, D., Krainski, E., & Rue, H. (2023). A
Byrnes, J. E. K., & Dee, L. E. (2025). Causal inference with observational              diffusion-­based spatio-­temporal extension of Gaussian Matérn fields.
      data and unobserved confounding variables. Ecology Letters, 28(1),                (arXiv:2006.04917) arXiv. https://​doi.​org/​10.​48550/​​arXiv.​2006.​
      e70023. https://​doi.​org/​10.​1111/​ele.​70023​                                  04917​
Foster, S. D., & Bravington, M. V. (2013). A Poisson–Gamma model for              Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between
      analysis of ecological non-­negative continuous data. Environmental               Gaussian fields and Gaussian Markov random fields: The stochastic
                                                                                                                                                        | 1301




                                                                                                                                                                   2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
THORSON


     partial differential equation approach. Journal of the Royal Statistical    Upham, N. S., Esselstyn, J. A., & Jetz, W. (2019). Inferring the mammal
     Society. Series B, Statistical Methodology, 73(4), 423–498. https://​            tree: Species-­level sets of phylogenies for questions in ecology,
     doi.​org/​10.​1111/j.​1467-­​9868.​2011.​0 0777.​x                               evolution, and conservation. PLoS Biology, 17(12), e3000494.
Lindmark, M., Anderson, S. C., & Thorson, J. T. (2026). Estimating scale-­            https://​doi.​org/​10.​1371/​journ​al.​pbio.​3 000494
     dependent covariate responses using two-­dimensional diffusion              Ver Hoef, J. M., Hanks, E. M., & Hooten, M. B. (2018). On the relationship
     derived from the stochastic partial differential equation method.                between conditional (CAR) and simultaneous (SAR) autoregressive
     Methods in Ecology and Evolution, 17(1), 207–218. https://​doi.​org/​            models. Spatial Statistics, 25, 68–85. https://​doi.​org/​10.​1016/j.​
     10.​1111/​2041-­​210x.​70177​                                                    spasta.​2018.​0 4.​0 06
McGilliard, C. R. (2024). Assessment of the Rex Sole Stock in the Gulf           Wood, S. N. (2017). Generalized additive models: An introduction with R.
     of Alaska. North Pacific Fishery Management Council. https://​                   CRC Press.
     w w w.​n pfmc.​o rg/​w p-­​c onte​n t/​P DFdo​c umen​t s/​S AFE/​2 024/​    Wootton, J. T., & Emmerson, M. (2005). Measurement of interac-
     GOArex.​p df                                                                     tion strength in nature. Annual Review of Ecology, Evolution, and
Miller, D. L. (2025). Bayesian views of generalized additive modelling.               Systematics, 36, 419–444.
     Methods in Ecology and Evolution, 16(3), 446–455. https://​doi.​org/​       Wright, S. (1921). Correlation and causation. Journal of Agricultural
     10.​1111/​2041-­​210X.​14498​                                                    Research, 20(7), 557–585.
Rue, H., & Held, L. (2005). Gaussian Markov random fields: Theory and ap-
     plications (1st ed.). CRC Press.
Schliep, E. M., Lany, N. K., Zarnetske, P. L., Schaeffer, R. N., Orians, C.      S U P P O R T I N G I N FO R M AT I O N
     M., Orwig, D. A., & Preisser, E. L. (2018). Joint species distribution      Additional supporting information can be found online in the
     modelling for spatio-­temporal occurrence and ordinal abundance             Supporting Information section at the end of this article.
     data. Global Ecology and Biogeography, 27(1), 142–155. https://​doi.​
                                                                                 Supporting Information 1. Derivation of a Gaussian Markov random
     org/​10.​1111/​geb.​12666​
Skaug, H., & Fournier, D. (2006). Automatic approximation of the mar-            field from a simultaneous equation model.
     ginal likelihood in non-­Gaussian hierarchical models. Computational        Supporting Information 2. Model details for case study 1.
     Statistics & Data Analysis, 51(2), 699–709.                                 Supporting Information 3. Model details for case study 2.
Thorson, J. T. (2026). James-­Thorson-­NOAA/graphical_mixed_model:
                                                                                 Supporting Information 4. Model details for case study 3.
     Release for accepted version [Computer software]. Zenodo. https://​
     doi.​org/​10.​5281/​zenodo.​18509086                                        Supporting Information 5. Additional figures and tables.
Thorson, J. T., Anderson, S. C., Goddard, P., & Rooper, C. N. (2025). tiny-      Table S1. Count of trait measurements available for each individual
     VAST: R package with an expressive Interface to specify lagged              trait (along the diagonal) or any pair of traits (off-­diagonal), from the
     and simultaneous effects in multivariate Spatio-­temporal models.
                                                                                 4999 species available in PanTHERIA that can be matched with the
     Global Ecology and Biogeography, 34(4), e70035. https://​doi.​org/​10.​
                                                                                 Vertlife phylogeny based on scientific binomial.
     1111/​geb.​70035​
Thorson, J. T., Andrews, A. G., III, Essington, T. E., & Large, S. I. (2024).    Table S2. Results from a simple-­random 10-­fold cross-­validation
     Dynamic structural equation models synthesize ecosystem dy-                 experiment for each of eight models for proportional abundance-­at-­
     namics constrained by ecological mechanisms. Methods in Ecology             age for rex sole in the Gulf of Alaska, arising from every combination
     and Evolution, 15(4), 744–755. https://​doi.​org/​10.​1111/​2041-­​210X.​
                                                                                 of estimating three potential interaction parameters or fixing
     14289​
Thorson, J. T., Maureaud, A. A., Frelat, R., Mérigot, B., Bigman, J. S.,         them at zero. For each model, we list the interaction parameters
     Friedman, S. T., Palomares, M. L. D., Pinsky, M. L., Price, S. A., &        included, the number of fixed effects, the cross-­validation root
     Wainwright, P. (2023). Identifying direct and indirect associations         mean squared error, and the proportion of cross-­validation mean-­
     among traits by merging phylogenetic comparative methods and
                                                                                 squared error relative to the null model. Note that each model
     structural equation models. Methods in Ecology and Evolution, 14(5),
     1259–1275. https://​doi.​org/​10.​1111/​2041-­​210X.​14076​                 includes 22 parameters in addition to the interactions: an intercept
Thorson, J. T., Monnahan, C. C., & Hulson, P.-­J. F. (2023). Data weighting:     for each age a ∈ {2, 3, … , 20}; the variance for exogenous variation
     An iterative process linking surveys, data synthesis, and popula-           in the graphical model; and the dispersion and power parameters
     tion models to evaluate mis-­specification. Fisheries Research, 266,
                                                                                 for the Tweedie distribution for residual variation.
     106762. https://​doi.​org/​10.​1016/j.​fishr​es.​2023.​106762
Thorson, J. T., Munch, S. B., & Swain, D. P. (2017). Estimating partial reg-     Figure S1. Observed (black bullets) and predicted (lines) proportional
     ulation in spatiotemporal models of community dynamics. Ecology,            abundance-­at-­age (y-­
                                                                                                       axis) for ages 2–20 (x-­
                                                                                                                              axis) in each year
     98(5), 1277–1289. https://​doi.​org/​10.​1002/​ecy.​1760                    1992–2022 (panels) for rex sole in the Gulf of Alaska, showing the
Thorson, J. T., & van der Bijl, W. (2023). phylosem: A fast and simple
                                                                                 prediction using all data (black line) or using a leave-­year-­out cross-­
     R package for phylogenetic inference and trait imputation using
     phylogenetic structural equation models. Journal of Evolutionary            validation design (red line). Note that (1) the age-­20 category includes
     Biology, 36(10), 1357–1364. https://​doi.​org/​10.​1111/​jeb.​14234​        all animals aged 20+, and (2) the red and black lines are identical in
Tobler, W. R. (1970). A computer movie simulating urban growth in the            years with no data (no black dots, e.g. 1993) because the leave-­year-­
     Detroit region. Economic Geography, 46, 234–240.
                                                                                 out cross-­validation fits to all available data during that model run.
Tung Ho, L. s., & Ané, C. (2014). A linear-­time algorithm for Gaussian
                                                                                 Figure S2. Histograms showing the count of simulation replicates
     and non-­Gaussian trait evolution models. Systematic Biology, 63(3),
     397–408.                                                                    (y-­axis) that result in a given z-­value (z = ̂𝜃(− 𝜃) , x-­axis) for estimates
                                                                                                                                 ̂ ̂
                                                                                                                                 SE 𝜃
Turchin, P. (2003). Complex population dynamics: A theoretical/empirical
                                                                                 of the natural-­
                                                                                                logarithm of the Ornstein-­
                                                                                                                          Uhlenbeck parameter
     synthesis. Princeton University Press.
                                                                                 relative to the known (true) value used to simulate data for each of
Turing, A. M. (1952). The chemical basis of morphogenesis. Philosophical
     Transactions of the Royal Society of London. Series B, Biological           200 simulation replicates, when fitting a generalized graphical mixed
     Sciences, 237(641), 37–72.                                                  model. The parameter is estimated separately for each of three
       |




                                                                                                                                            2041210x, 2026, 4, Downloaded from https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70272 by Max-Planck-Institut Für, Wiley Online Library on [21/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
1302                                                                                                                              THORSON



traits (rows), and we explore three scenarios where 50%, 70%, or
90% of taxa are independently missing a measurement for a given         How to cite this article: Thorson, J. T. (2026). Generalized
trait (columns). A well-­performing model will accurately estimate      graphical mixed models connect ecological theory with widely
uncertainty, such that the z-­value will follow a normal distribution   used statistical models. Methods in Ecology and Evolution, 17,
(black line, showing the normal probability density function            1290–1302. https://doi.org/10.1111/2041-210x.70272
multiplied by the number of replicates and histogram bin width).
