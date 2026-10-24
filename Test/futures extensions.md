
### 1. La famille « Détection Imparfaite & Démographie » (Priorité #1 — Demande directe de R3)

Actuellement, ton Tableau 1 couvre les modèles *Zero-inflated* et les *Variables latentes* génériques, mais il manque les modèles canoniques de l'écologie de terrain. Grâce à l'énumération discrète automatique (Point 1.6) et à `DiscreteHMCGibbs` (Point 1.7), `BayesForge` peut surclasser `Stan` et `unmarked`/`ubms` sur :

1. **Modèle d'Occupation Simple & Dynamique (MacKenzie / multi-saisons) :** Estime la présence réelle $z_i \in \{0, 1\}$, la probabilité de colonisation ($\gamma$), d'extinction ($\epsilon$) et de détection ($p_{i,t}$). *(Répond directement à la critique du Reviewer 3).*
2. **Modèle d'Occupation Multi-Espèces (Communautés / Dorazio-Royle) :** Combine les effets aléatoires hiérarchiques par espèce avec l'occupation latente $z_{i,s}$ et une matrice de corrélation résiduelle (ou facteurs latents) entre espèces. Sur GPU, vectoriser 100 espèces $\times$ 500 sites est fulgurant.
3. **Modèles *N-Mixture* (Royle) d'abondance :** Estime la taille de population latente $N_i \sim \text{Poisson}(\lambda_i)$ à partir de comptages répétés $y_{i,t} \sim \text{Binomial}(N_i, p_{i,t})$ via `DiscreteHMCGibbs`.
4. **Capture-Marquage-Recapture (Cormack-Jolly-Seber / CJS & Jolly-Seber) :** Modélise l'état de survie latent $z_{i,t} \in \{0, 1\}$ d'individus marqués dans le temps.
5. **Distance Sampling Hiérarchique :** Estime la densité animale lorsque la probabilité de détection décroît avec la distance à l'observateur (demi-normale ou *hazard-rate*).

---

### 2. Écologie du Mouvement & Séries Temporelles à États Cachés (Demande directe de l'Associate Editor, L.40)

L'Associate Editor s'attendait explicitement à voir un exemple sur les états comportementaux cachés et la télémétrie GPS après avoir lu ton introduction.

6. **Hidden Markov Models (HMM) de mouvement animal :** Décompose les trajectoires GPS (longueur de pas $\sim \text{Gamma}$, angle de tournant $\sim \text{Von Mises}$) en $K$ états comportementaux latents discrets (ex. *repos*, *foraging*, *transit*) avec l'algorithme Forward-Backward vectorisé sur GPU.
7. **State-Space Models (SSM) de suivi télémétrique et de dynamique des populations :** Sépare l'erreur de localisation GPS/Argos (modèle d'observation) de la vraie trajectoire continue de l'animal ou de la croissance stochastique d'une population (modèle de processus Gompertz / Ricker).
8. **Integrated Step-Selection Analysis (iSSA) hiérarchique :** Modèle de régression de Poisson/Cox conditionnel avec pentes aléatoires par individu pour la sélection d'habitat fine échelle.

---

### 3. Réseaux Sociaux Dynamiques, Multiniveaux & Apprentissage (Ton domaine phare)

Tu as déjà une couverture exceptionnelle des réseaux statiques (SRM, SBM, contrôle de biais, arêtes BISON, NBDA, EWA). Pour aller encore plus loin :

9. **SRM / SBM Dynamique (Longitudinaux / Temps discret ou continu) :** Modélise comment la réciprocité dyadique et la structure en blocs évoluent d'une saison ou d'une année à l'autre (processus autorégressif $\text{AR}(1)$ sur les effets latents du SRM).
10. **Réseaux Multicouches (Multiplex Networks) :** Couple simultanément plusieurs types d'interactions (ex. *épouillage*, *agression*, *proximité spatiale*) dans un tenseur 3D pour estimer la corrélation inter-couches au niveau individuel et dyadique.
11. **Modèles d'Apprentissage par Renforcement (RL : Rescorla-Wagner / Q-Learning hiérarchique) :** Complément direct à ton modèle *EWA* pour modéliser la prise de décision cognitive et le foraging social dans le temps.



---

### 4. Écologie des Communautés, Génétique Quantitative & Macroévolution

Pour compléter tes modèles phylogénétiques et de variables latentes actuels :

12. **JSDM (Joint Species Distribution Model / architecture type *HMSC* ou *gllvm*) :** Couple des facteurs latents spatiaux, des traits fonctionnels d'espèces et une matrice de covariance phylogénétique pour modéliser l'assemblage de centaines d'espèces simultanément. Sur CPU (dans R avec `Hmsc`), ces modèles prennent des jours ; sur GPU dans `BayesForge`, c'est un cas d'usage rêvé.
13. **L'« Animal Model » en Génétique Quantitative ($h^2$ et Matrice $\mathbf{G}$) :** Utilise la matrice de parenté génétique (pedigree ou GRM génomique) pour partitionner la variance phénotypique en variance génétique additive ($\mathbf{G}$), effets maternels et environnement permanent. Mathématiquement identique au PGLMM, mais avec un public énorme en biologie évolutive.


14. **Modèles d'évolution des traits Ornstein-Uhlenbeck (OU) multi-optima :** Au-delà du mouvement brownien standard en phylogénie, modélise la sélection stabilisante ($\alpha$) vers différents optima écologiques ($\theta$).

---

### 5. Écologie Spatiale & Spatio-Temporelle Haute Performance

Tu as déjà les *Processus Gaussiens (GP)*, mais un GP exact en $\mathcal{O}(N^3)$ sature au-delà de $N = 3\,000$ points, même sur GPU.

15. **HSGP (Hilbert Space Gaussian Processes) ou GP de rang faible :** Approximation spectrale des Processus Gaussiens qui fait passer la complexité de $\mathcal{O}(N^3)$ à $\mathcal{O}(N \cdot M)$, permettant de modéliser des dizaines de milliers de coordonnées spatiales en quelques secondes.
16. **Modèles spatiaux sur graphe / voisinage (CAR / ICAR / BYM2) :** Indispensables pour les données écologiques ou épidémiologiques agrégées par régions, parcelles forestières ou bassins versants.

---

### 6. Inférence Causale & Équations Structurelles

Tu mentionnes l'inférence causale dès l'introduction et McElreath (*Statistical Rethinking*) est la colonne vertébrale de ton Tableau 1.

17. **SEM Bayésien (Structural Equation Modeling / Path Analysis) & Médiation Causale :** Permet d'estimer simultanément des chaînes causales complètes ($X \rightarrow M \rightarrow Y$) avec variables latentes, données manquantes et contrôle phylogénétique ou réseau.
18. **Modèles Non-Linéaires Mécanistes (Croissance de von Bertalanffy / Courbes de performance thermique TPC / Réponse fonctionnelle de Holling) :** Des modèles où l'équation biologique est explicitement non-linéaire (et où `brms` devient vite lourd à écrire et lent à compiler).

---

### Résumé : Le « Top 5 » à coder en priorité absolue (Impact maximal / Effort minimal)

Si tu dois n'en choisir que 5 pour la prochaine version (et pour blinder le papier), voici ceux qui sont les plus rapides à écrire avec tes briques actuelles et qui répondent directement aux critiques :

| Rang | Modèle à ajouter | Pourquoi c'est stratégique | Briques `BF` utilisées |
| --- | --- | --- | --- |
| **1** | **Occupation Multi-Saisons & Multi-Espèces** | Répond directement au Reviewer 3 ; démontre la supériorité sur Stan via l'énumération discrète GPU. | `m.dist.bernoulli` + `enumerate="parallel"` + `varying_effects` |
| **2** | **HMM de Mouvement Animal (États cachés)** | Répond directement à l'Associate Editor (L.40) ; énorme communauté en bio-logging/télémétrie. | `scan` / `forward_algorithm` + `Gamma` / `VonMises` |
| **3** | **Modèle *N-Mixture* d'abondance (Royle)** | Montre tout l'intérêt de ton nouveau sampler `DiscreteHMCGibbs` (Point 1.7) sur un support discret. | `m.dist.poisson` (latent) + `m.dist.binomial` + `discrete_gibbs` |
| **4** | **JSDM (Joint Species Distribution Model)** | Tueur absolu sur GPU (passe de plusieurs jours dans R/`Hmsc` à quelques minutes dans `BF`). | Combinaison de ton `PCA`/`Latent` + `Phylogenetic` + `Varying slopes`<br> |
| **5** | **Animal Model (Génétique Quantitative / Matrice G)** | Zéro effort de code (c'est exactement ton PGLMM avec une matrice de pedigree/GRM au lieu d'un arbre), mais ouvre tout le marché de la génétique évolutive.

 | `m.effects` / Cholesky de la matrice de parenté $A$ |