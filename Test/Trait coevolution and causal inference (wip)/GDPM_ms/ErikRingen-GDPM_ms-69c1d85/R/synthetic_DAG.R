# Install and load required packages if not already installed
# install.packages("DiagrammeR")
# This file creates a DAG diagram - only execute when explicitly called
# Wrapped in a function to prevent execution during tar_source()
create_synthetic_DAG <- function() {
  if (!requireNamespace("DiagrammeR", quietly = TRUE) || 
      !requireNamespace("DiagrammeRsvg", quietly = TRUE)) {
    warning("DiagrammeR packages not available, skipping DAG creation")
    return(invisible(NULL))
  }
  
  library(DiagrammeR)
  library(DiagrammeRsvg)

  g <-grViz("
  digraph dynamic_dag {
    # Graph settings
    rankdir=LR;
    layout=dot;
    
    # Define node attributes for latent variables (circles)
    node [shape=circle, fixedsize=true, width=0.7, fontname='Helvetica'];
    
    # Create latent nodes with HTML labels and smaller subscripts
    { rank=same; eta1_t [label=<η<SUB><FONT POINT-SIZE='10'>t,1</FONT></SUB>>]; 
                eta2_t [label=<η<SUB><FONT POINT-SIZE='10'>t,2</FONT></SUB>>];
                eta3_t [label=<η<SUB><FONT POINT-SIZE='10'>t,3</FONT></SUB>>];
                w1_t [label=<w<SUB><FONT POINT-SIZE='10'>t,1</FONT></SUB>>, fixedsize=true, width=0.5, penwidth = 0.5]; 
                w2_t [label=<w<SUB><FONT POINT-SIZE='10'>t,2</FONT></SUB>>, fixedsize=true, width=0.5, penwidth = 0.5];
                w3_t [label=<w<SUB><FONT POINT-SIZE='10'>t,3</FONT></SUB>>, fixedsize=true, width=0.5, penwidth = 0.5];}
                
    { rank=same; eta1_t2 [label=<η<SUB><FONT POINT-SIZE='10'>t+1,1</FONT></SUB>>]; 
                eta2_t2 [label=<η<SUB><FONT POINT-SIZE='10'>t+1,2</FONT></SUB>>]; 
                eta3_t2 [label=<η<SUB><FONT POINT-SIZE='10'>t+1,3</FONT></SUB>>];
                w1_t2 [label=<w<SUB><FONT POINT-SIZE='10'>t+1,1</FONT></SUB>>, fixedsize=true, width=0.5, penwidth = 0.5]; 
                w2_t2 [label=<w<SUB><FONT POINT-SIZE='10'>t+1,2</FONT></SUB>>, fixedsize=true, width=0.5, penwidth = 0.5];
                w3_t2 [label=<w<SUB><FONT POINT-SIZE='10'>t+1,3</FONT></SUB>>, fixedsize=true, width=0.5, penwidth = 0.5];}
    
    # Create manifest variables (squares)
    node [shape=box, fixedsize=true, width=1.2 , penwidth=1];
    { rank=same; eta1_t2_m [label='promiscuity']; eta2_t2_m [label='sperm size'];  eta3_t2_m [label='predation']; }
    
    # Force vertical ordering with invisible edges
    edge [style=invis];
    eta1_t -> eta2_t -> eta3_t;
    eta1_t2 -> eta2_t2 -> eta3_t2;
    eta1_t2_m -> eta2_t2_m -> eta3_t2_m;
    
    # Time series edges (gray)
    edge [color='black', penwidth=1, style=solid];
    eta1_t -> eta1_t2;
    eta2_t -> eta2_t2;
    eta3_t -> eta3_t2;
    
    # Edges from drift
    edge [color='#888888', penwidth=1];
    w1_t -> eta1_t;
    w2_t -> eta2_t;
    w3_t -> eta3_t;
    w1_t2 -> eta1_t2;
    w2_t2 -> eta2_t2;
    w3_t2 -> eta3_t2;
    
    # Reciprocal causation (blue)
    edge [color='skyblue', penwidth=1];
    eta1_t -> eta2_t2;
    eta1_t -> eta3_t2;
    
    edge [color='indianred', penwidth=1];
    eta3_t -> eta1_t2;
    eta3_t -> eta2_t2;
    
    # Measurement model edges (black)
    edge [color='black', penwidth=1];
    eta1_t2 -> eta1_t2_m;
    eta2_t2 -> eta2_t2_m;
    eta3_t2 -> eta3_t2_m;
  }
")

  DiagrammeRsvg::export_svg(g) |>
  cat(file = "figures/dags/synthetic_dag.svg")
  
  return(g)
}
