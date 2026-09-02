if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos="http://cran.us.r-project.org")
}
remotes::install_local("STbayes_repo", dependencies=TRUE)
