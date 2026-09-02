prepare_primates_GDPM_data <- function(primates_data_file, primates_tree_file){
    d <- read.csv(primates_data_file)
    tree <- read.tree(primates_tree_file)
    
    setdiff(d$taxon, tree$tip.label) # check for discrep between tree names and dataframe names
    d_matched <- d[match(tree$tip.label, d$taxon),] # make sure trait data and phylogeny are in the same order
    identical(d_matched$taxon, tree$tip.label)
    
    # clade data used for posterior predictive checks
    primates2 <- coevolve::primates$data
    d_matched <- left_join(d_matched, primates2, by = c("taxon" = "species"))
    
    body <- d_matched$body
    brain <- d_matched$brain
    
    # replace NA's with -99
    longevity <- d_matched$max_longevity
    longevity[is.na(longevity)] <- -99 # placeholder for Stan
    
    maturity <- d_matched$fem_maturity
    maturity[is.na(maturity)] <- -99
    
    # Use coevolve package to make initial stan data list and stancode, using dummy data for latent variables
    data_list <- coevolve::coev_make_standata(
      data = d_matched %>% mutate(life_history = rnorm(nrow(d_matched)), diet = rnorm(nrow(d_matched)), beta = rnorm(nrow(d_matched))),
      variables = list(
        life_history = "normal",
        beta = "normal"
      ),
      id = "taxon",
      tree = tree
    )
    
    data_list$N_latent <- 2 # number of latent traits
    data_list$y <- as.matrix(data.frame(body = body, brain = brain, longevity = longevity, maturity = maturity)) # observed traits
    data_list$J <- ncol(data_list$y) # number of observed traits
    data_list$miss <- as.matrix(data_list$y == -99) * 1 # missing data indicator matrix
    data_list$y_mean <- apply(data_list$y, 2, function(x) mean(x[x != -99])) # sample means for each observed trait, used for scaling in stan model
    
    data_list$clade <- d_matched$clade
  
  return(data_list)
}