```r
#' Samples from an Asymmetric Laplace distribution.
#'
#' The Asymmetric Laplace distribution is a generalization of the Laplace distribution,
#' where the two sides of the distribution are scaled differently. It is defined by
#' a location parameter (`loc`), a scale parameter (`scale`), and an asymmetry parameter (`asymmetry`).
#'
#' \deqn{f(x) = \frac{1}{2 \text{scale}} \left( \frac{1}{\text{scale}} \exp\left(-\frac{|x - \text{loc}|}{\text{scale} \cdot \text{asymmetry}}}\right) \text{ if } x < \text{loc} + \text{scale} \cdot \text{asymmetry}
+ \frac{1}{\text{scale}} \exp\left(-\frac{|x - \text{loc}|}{\text{scale} / \text{asymmetry}}\right) \text{ if } x > \text{loc} - \text{scale} / \text{asymmetry}}
#'
#' @export
#' @importFrom reticulate py_none tuple
#' @param loc A numeric vector or single numeric value representing the location parameter of the distribution.
#' @param scale A numeric vector or single numeric value representing the scale parameter of the distribution.
#' @param asymmetry A numeric vector or single numeric value representing the asymmetry parameter of the distribution.
#' @param shape A numeric vector specifying the shape of the output.  This is used to set the batch shape when `sample=FALSE` (model building) or as `sample_shape` to draw a raw JAX array when `sample=TRUE` (direct sampling).
#' @param event Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
#' @param mask A logical vector indicating which observations to mask.
#' @param create_obj Logical; If `TRUE`, returns the raw NumPyro distribution object instead of creating a sample site.
#'
#' @return When `sample=FALSE`: A NumPyro AsymmetricLaplace distribution object (for model building).
#'         When `sample=TRUE`: A JAX array of samples drawn from the AsymmetricLaplace distribution (for direct sampling).
#'         When `create_obj=TRUE`: The raw NumPyro distribution object (for advanced use cases).
#'
#' @examples
#' library(BI)
#' m = importBI('cpu')
#' bi.dist.asymmetric_laplace(loc=0.0, scale=1.0, asymmetry=1.0, sample=TRUE)
#'
#' @seealso \url{https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplace}
```
bi.dist.asymmetric_laplace=function(loc=0.0, scale=1.0, asymmetry=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), sample=FALSE, seed=0, shape=c(), event=0, create_obj=FALSE) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bi$dist$asymmetric_laplace(loc=loc,  scale= scale,  asymmetry= asymmetry,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj)
}
