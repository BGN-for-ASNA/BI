# Install and load required packages if not already installed
# install.packages("DiagrammeR")
# This file creates a DAG diagram - only execute when explicitly called
# Wrapped in a function to prevent execution during tar_source()
create_primates_DAG <- function() {
  if (!requireNamespace("DiagrammeR", quietly = TRUE) || 
      !requireNamespace("DiagrammeRsvg", quietly = TRUE)) {
    warning("DiagrammeR packages not available, skipping DAG creation")
    return(invisible(NULL))
  }
  
  library(DiagrammeR)
  library(DiagrammeRsvg)

  g <- grViz("
  digraph dynamic_dag {
   # Graph settings
   rankdir=LR;
   layout=dot;
   splines=true;
   nodesep=0.3;
   ranksep=1;
   
   # Time t
   { rank=same;
       w1_t [label=<w<SUB><FONT POINT-SIZE='10'>t,1</FONT></SUB>>, shape=circle, width=0.5, fixedsize=true, penwidth=0.5, color=orange];
       w2_t [label=<w<SUB><FONT POINT-SIZE='10'>t,2</FONT></SUB>>, shape=circle, width=0.5, fixedsize=true, penwidth=0.5, color=cornflowerblue];
       eta1_t [label=<η<SUB><FONT POINT-SIZE='10'>t,1</FONT></SUB>>, shape=circle, width=0.7, color=orange];
       eta2_t [label=<η<SUB><FONT POINT-SIZE='10'>t,2</FONT></SUB>>, shape=circle, width=0.7, color=cornflowerblue];
       
       # Force ordering within time t
       w1_t -> w2_t -> eta1_t -> eta2_t [style=invis];
   }
   
   # Time t+1
   { rank=same;
       w1_t2 [label=<w<SUB><FONT POINT-SIZE='10'>t+1,1</FONT></SUB>>, shape=circle, width=0.5, fixedsize=true, penwidth=0.5, color=orange];
       w2_t2 [label=<w<SUB><FONT POINT-SIZE='10'>t+1,2</FONT></SUB>>, shape=circle, width=0.5, fixedsize=true, penwidth=0.5, color=cornflowerblue];
       eta1_t2 [label=<η<SUB><FONT POINT-SIZE='10'>t+1,1</FONT></SUB>>, shape=circle, width=0.7, color=orange];
       eta2_t2 [label=<η<SUB><FONT POINT-SIZE='10'>t+1,2</FONT></SUB>>, shape=circle, width=0.7, color=cornflowerblue];
       
       # Force ordering within time t+1
       w1_t2 -> w2_t2 -> eta1_t2 [style=invis];
   }
   
   # Create manifest variables (squares)
   node [shape=box, fixedsize=true, width=1.35, penwidth=1];
   { rank=same; 
       eta1_t2_m1 [label='body weight', color=orange];
       eta1_t2_m2 [label='longevity', color=orange];
       eta1_t2_m3 [label='age at maturity', color=orange];
       eta2_t2_m1 [label='brain weight', color=cornflowerblue];
   }
   
   # Force manifest variable ordering
  eta1_t2_m3 -> eta1_t2_m2 -> eta1_t2_m1 -> eta2_t2_m1 [style=invis];
   
   # Time series edges (black)
   edge [color='black', penwidth=1];
   eta1_t -> eta1_t2 [color=orange];
   eta2_t -> eta2_t2 [color=cornflowerblue];

   eta1_t -> eta2_t2 [color=orange];
   
   eta2_t -> eta1_t2 [color=cornflowerblue];

   # Edges from drift
   edge [color='#darkgrey', penwidth=0.5];
   w1_t -> eta1_t [color=orange];
   w2_t -> eta2_t [color=cornflowerblue];
   w1_t2 -> eta1_t2 [color=orange];
   w2_t2 -> eta2_t2 [color=cornflowerblue];

   # Correlated drift
   edge [color='darkgrey', penwidth=0.5];
   w1_t -> w2_t [dir=none];
   w1_t2 -> w2_t2 [dir=none];

   # Measurement model edges (black)
   edge [color='black', penwidth=1];
   eta2_t2 -> eta2_t2_m1 [color=cornflowerblue];
   eta1_t2 -> eta1_t2_m1 [color=orange];
   eta1_t2 -> eta1_t2_m2 [color=orange];
   eta1_t2 -> eta1_t2_m3 [color=orange];
   eta1_t2_m1 -> eta2_t2_m1 [color=black];
}
")
  #g
  # 
  DiagrammeRsvg::export_svg(g) %>%
  cat(file = "figures/dags/primate_dag.svg")
  
  return(g)
}
