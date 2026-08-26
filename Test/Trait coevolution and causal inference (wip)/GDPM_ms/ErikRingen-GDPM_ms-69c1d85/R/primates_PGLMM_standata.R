prepare_primates_PGLMM_data <- function(primates_data_file, primates_tree_file){
    d <- read.csv(primates_data_file)
    tree <- read.tree(primates_tree_file)
    
    setdiff(d$taxon, tree$tip.label) # check for discrep between tree names and dataframe names
    d_matched <- d[match(tree$tip.label, d$taxon),] # make sure trait data and phylogeny are in the same order
    identical(d_matched$taxon, tree$tip.label)

    dist_mat <- cophenetic.phylo(tree)
    dist_mat <- dist_mat / max(dist_mat)
    
    # Scale variables and replace NA's with -99
    longevity <- d_matched$max_longevity/mean(d_matched$max_longevity, na.rm=T)
    longevity[is.na(longevity)] <- -99 # placeholder for Stan

    body <- d_matched$body
    brain <- d_matched$brain

    prop_fruit <- d_matched$prop_fruit
    prop_fruit[is.na(prop_fruit)] <- -99
    
    folivore <- ifelse(d_matched$diet_cat == "Fol", 1, 0)

    maturity <- d_matched$fem_maturity/mean(d_matched$fem_maturity, na.rm=T)
    maturity[is.na(maturity)] <- -99

    data_list <- list(
      N_species = nrow(d),
      dist_mat = dist_mat,
      brain_weight = brain,
      body_weight = body,
      mean_body = mean(body),
      longevity = longevity,
      prop_fruit = prop_fruit,
      folivore = folivore,
      maturity = maturity,
      N_body_miss = sum(body == -99),
      N_longevity_miss = sum(longevity == -99),
      N_maturity_miss = sum(maturity == -99),
      N_fruit_miss = sum(prop_fruit == -99)
    )
  
  return(data_list)
}