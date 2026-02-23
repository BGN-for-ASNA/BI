"""
MCP Tools for BayesInference operations.
"""

import sys
import json
import traceback
from typing import Any, Dict, Optional, List
import pandas as pd

# Import BI package
try:
    from BI import bi
except ImportError:
    # Handle case where BI is not installed
    bi = None

from .utils import serialize_posterior, format_summary_table, safe_serialize


# Global model instance storage
_model_instances = {}


def initialize_model(
    platform: str = "cpu",
    model_id: str = "default",
    rand_seed: bool = True
) -> Dict[str, Any]:
    """
    Initialize a new BI model instance.
    
    Args:
        platform: Platform to use ("cpu", "gpu", or "tpu")
        model_id: Identifier for this model instance
        rand_seed: Whether to use random seed
        
    Returns:
        Status message and model information
    """
    try:
        if bi is None:
            return {
                "success": False,
                "error": "BayesInference package not installed. Please install it first."
            }
        
        # Create model instance
        model = bi(platform=platform, rand_seed=rand_seed, print_devices_found=False)
        
        # Store instance
        _model_instances[model_id] = model
        
        return {
            "success": True,
            "model_id": model_id,
            "platform": platform,
            "message": f"Model initialized successfully on {platform}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def load_dataset(
    dataset_name: str,
    as_dict: bool = False
) -> Dict[str, Any]:
    """
    Load a built-in dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        as_dict: Whether to return as dictionary (default: pandas DataFrame)
        
    Returns:
        Dataset data and metadata
    """
    try:
        if bi is None:
            return {
                "success": False,
                "error": "BayesInference package not installed"
            }
        
        # Create temporary model instance to access datasets
        temp_model = bi(print_devices_found=False)
        
        # Load dataset
        dataset_methods = {
            "howell1": temp_model.load.howell1,
            "milk": temp_model.load.milk,
            "iris": temp_model.load.iris,
            "chimpanzees": temp_model.load.chimpanzees,
            "reedfrogs": temp_model.load.reedfrogs,
            "tulips": temp_model.load.tulips,
            "ucbadmit": temp_model.load.ucbadmit,
            "trolley": temp_model.load.trolley,
            "elephants": temp_model.load.elephants,
            "waffle_divorce": temp_model.load.WaffleDivorce,
        }
        
        if dataset_name not in dataset_methods:
            return {
                "success": False,
                "error": f"Unknown dataset: {dataset_name}. Available: {list(dataset_methods.keys())}"
            }
        
        # Load the dataset
        data = dataset_methods[dataset_name](frame=True)
        
        if as_dict:
            data_dict = data.to_dict(orient='list')
            return {
                "success": True,
                "dataset_name": dataset_name,
                "data": data_dict,
                "shape": list(data.shape),
                "columns": list(data.columns)
            }
        else:
            return {
                "success": True,
                "dataset_name": dataset_name,
                "data": data.to_string(max_rows=20),
                "shape": list(data.shape),
                "columns": list(data.columns),
                "head": data.head(10).to_dict(orient='records')
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def simulate_data(
    code: str,
    model_id: str = "default",
    seed: int = 0,
    platform: str = "cpu"
) -> Dict[str, Any]:
    """
    Simulate data using BI distributions.
    
    Args:
        code: Python code to generate data (can use 'm' for bi instance)
        model_id: Model instance identifier
        seed: Random seed
        platform: Platform to use if model not initialized
        
    Returns:
        Simulated data and variables
    """
    try:
        # Get or create model instance
        if model_id not in _model_instances:
            init_result = initialize_model(platform=platform, model_id=model_id)
            if not init_result["success"]:
                return init_result
        
        model = _model_instances[model_id]
        
        # Parse and execute simulation code
        # We provide 'm' as the bi instance and 'jnp' for convenience
        import jax.numpy as jnp
        local_vars = {"m": model, "jnp": jnp, "bi": bi}
        
        # Execute the code
        exec(code, {"m": model, "jnp": jnp, "bi": bi}, local_vars)
        
        # Filter out internal variables and the bi instance itself
        results = {}
        for k, v in local_vars.items():
            if k not in ["m", "jnp", "bi", "__builtins__"]:
                results[k] = safe_serialize(v)
                
        return {
            "success": True,
            "model_id": model_id,
            "data": results,
            "message": "Data simulated successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def fit_model(
    model_code: str,
    data: Optional[Dict[str, Any]] = None,
    model_id: str = "default",
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    platform: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fit a Bayesian model using MCMC sampling.
    
    Args:
        model_code: Python code defining the model function
        data: Dictionary of data to pass to the model
        model_id: Model instance identifier
        num_warmup: Number of warmup iterations
        num_samples: Number of sampling iterations
        num_chains: Number of MCMC chains
        platform: Platform override (if not using existing instance)
        
    Returns:
        Fitting results and diagnostics
    """
    try:
        # Get or create model instance
        if model_id not in _model_instances:
            if platform is None:
                platform = "cpu"
            init_result = initialize_model(platform=platform, model_id=model_id)
            if not init_result["success"]:
                return init_result
        
        model = _model_instances[model_id]
        
        # Parse and execute model code
        local_vars = {"m": model}
        exec(model_code, {"m": model, "bi": bi}, local_vars)
        
        # Get the model function
        if "model" not in local_vars:
            return {
                "success": False,
                "error": "Model code must define a function named 'model'"
            }
        
        model_func = local_vars["model"]
        
        # Fit the model
        model.fit(
            model=model_func,
            obs=data,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model fitted successfully",
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def get_summary(
    model_id: str = "default",
    round_to: int = 2,
    hdi_prob: float = 0.89
) -> Dict[str, Any]:
    """
    Get posterior summary statistics.
    
    Args:
        model_id: Model instance identifier
        round_to: Number of decimal places
        hdi_prob: HDI probability interval
        
    Returns:
        Summary statistics table
    """
    try:
        if model_id not in _model_instances:
            return {
                "success": False,
                "error": f"Model '{model_id}' not found. Initialize it first."
            }
        
        model = _model_instances[model_id]
        
        # Get summary
        summary = model.summary(round_to=round_to, hdi_prob=hdi_prob)
        
        return {
            "success": True,
            "model_id": model_id,
            "summary": summary.to_string(),
            "summary_dict": summary.to_dict()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def sample_posterior(
    model_id: str = "default",
    num_samples: int = 1,
    remove_obs: bool = True,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Generate posterior predictive samples.
    
    Args:
        model_id: Model instance identifier
        num_samples: Number of samples to generate
        remove_obs: Whether to remove observed data
        seed: Random seed
        
    Returns:
        Posterior predictive samples
    """
    try:
        if model_id not in _model_instances:
            return {
                "success": False,
                "error": f"Model '{model_id}' not found. Initialize it first."
            }
        
        model = _model_instances[model_id]
        
        # Generate samples
        samples = model.sample(
            samples=num_samples,
            remove_obs=remove_obs,
            seed=seed
        )
        
        # Serialize samples
        serialized = safe_serialize(samples)
        
        return {
            "success": True,
            "model_id": model_id,
            "num_samples": num_samples,
            "samples": serialized
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def get_diagnostics(
    model_id: str = "default"
) -> Dict[str, Any]:
    """
    Get MCMC diagnostics (R-hat, ESS, etc.).
    
    Args:
        model_id: Model instance identifier
        
    Returns:
        Diagnostic information
    """
    try:
        if model_id not in _model_instances:
            return {
                "success": False,
                "error": f"Model '{model_id}' not found. Initialize it first."
            }
        
        model = _model_instances[model_id]
        
        # Get summary which includes diagnostics
        summary = model.summary()
        
        # Extract diagnostic columns
        diagnostics = {}
        if 'r_hat' in summary.columns:
            diagnostics['r_hat'] = summary['r_hat'].to_dict()
        if 'ess_bulk' in summary.columns:
            diagnostics['ess_bulk'] = summary['ess_bulk'].to_dict()
        if 'ess_tail' in summary.columns:
            diagnostics['ess_tail'] = summary['ess_tail'].to_dict()
        
        return {
            "success": True,
            "model_id": model_id,
            "diagnostics": diagnostics,
            "full_summary": summary.to_string()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def create_simple_linear_model(
    x_data: List[float],
    y_data: List[float],
    model_id: str = "default",
    num_warmup: int = 500,
    num_samples: int = 500,
    platform: str = "cpu"
) -> Dict[str, Any]:
    """
    Convenience tool to create and fit a simple linear regression model.
    
    Args:
        x_data: Predictor variable data
        y_data: Response variable data
        model_id: Model instance identifier
        num_warmup: Number of warmup iterations
        num_samples: Number of sampling iterations
        platform: Platform to use
        
    Returns:
        Complete results including summary
    """
    try:
        # Initialize model if needed
        if model_id not in _model_instances:
            init_result = initialize_model(platform=platform, model_id=model_id)
            if not init_result["success"]:
                return init_result
        
        model = _model_instances[model_id]
        
        # Convert data to JAX arrays
        import jax.numpy as jnp
        x = jnp.array(x_data)
        y = jnp.array(y_data)
        
        # Define model
        def linear_model(x, y):
            alpha = model.dist.normal(loc=0, scale=10, name="alpha")
            beta = model.dist.normal(loc=0, scale=10, name="beta")
            sigma = model.dist.exponential(1, name="sigma")
            mu = alpha + beta * x
            model.dist.normal(mu, sigma, obs=y)
        
        # Fit model
        model.fit(
            model=linear_model,
            obs={"x": x, "y": y},
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1
        )
        
        # Get summary
        summary = model.summary()
        
        return {
            "success": True,
            "model_id": model_id,
            "model_type": "linear_regression",
            "message": "Linear regression model fitted successfully",
            "summary": summary.to_string(),
            "summary_dict": summary.to_dict()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def convert_stan_to_bi(stan_code: str) -> Dict[str, Any]:
    """
    Parses a Stan model and applies semantic mapping rules to generate an equivalent BI Python model.
    Note: This is a best-effort structural mapping.
    
    Args:
        stan_code: The raw Stan model code to convert.
        
    Returns:
        Conversion result containing the BI code, explanation, and confidence.
    """
    try:
        explanation = []
        assumptions = []
        confidence = 0.8  # Start high, lower if we hit unknown constructs
        
        # Very basic parsing to find blocks
        import re
        
        data_block_match = re.search(r'data\s*\{([^}]*)\}', stan_code, re.DOTALL)
        params_block_match = re.search(r'parameters\s*\{([^}]*)\}', stan_code, re.DOTALL)
        model_block_match = re.search(r'model\s*\{([^}]*)\}', stan_code, re.DOTALL)
        
        bi_code_lines = ["def model(data):"]
        explanation.append("Created a Python 'model' function that takes 'data' as argument.")
        
        # Data block
        if data_block_match:
            explanation.append("Found Stan 'data' block.")
            assumptions.append("Assuming data variables are passed inside the 'data' dictionary or available in scope.")
            data_vars = []
            for line in data_block_match.group(1).split('\n'):
                line = line.strip()
                if not line or line.startswith('//'): continue
                # Match e.g., 'int N;', 'array[10] int T;', 'vector[N] K;'
                m = re.search(r'\\b([a-zA-Z0-9_]+)\s*;', line)
                if m:
                    var_name = m.group(1)
                    bi_code_lines.append(f"    {var_name} = data.get('{var_name}')")
                    data_vars.append(var_name)
        
        # Params block
        if params_block_match:
            explanation.append("Found Stan 'parameters' block.")
            bi_code_lines.append("")
            bi_code_lines.append("    # Parameters (Priors must be defined below based on model block)")
            for line in params_block_match.group(1).split('\n'):
                line = line.strip()
                if not line or line.startswith('//'): continue
                m = re.search(r'\\b([a-zA-Z0-9_]+)\s*;', line)
                if m:
                    var_name = m.group(1)
                    bi_code_lines.append(f"    # TODO: Define prior for '{var_name}' (e.g., {var_name} = m.dist.normal(..., name='{var_name}'))")
        
        # Model block
        if model_block_match:
            explanation.append("Found Stan 'model' block.")
            bi_code_lines.append("")
            bi_code_lines.append("    # Model Likelihood")
            for line in model_block_match.group(1).split('\n'):
                line = line.strip()
                if not line or line.startswith('//'): continue
                
                # Check for sampling statements
                sample_match = re.search(r'([a-zA-Z0-9_\[\]]+)\s*~\s*([a-zA-Z0-9_]+)\s*\(([^;]+)\)\s*;', line)
                if sample_match:
                    lhs = sample_match.group(1).strip()
                    dist = sample_match.group(2).strip()
                    args = sample_match.group(3).strip()
                    
                    if dist == 'normal':
                        bi_dist = 'normal'
                    elif dist == 'poisson':
                        bi_dist = 'poisson'
                    elif dist == 'binomial':
                        bi_dist = 'binomial'
                    elif dist == 'exponential':
                        bi_dist = 'exponential'
                    else:
                        bi_dist = f"TODO_mapped_{dist}"
                        confidence -= 0.1
                        
                    # If it's a known parameter vs observed data
                    # This requires deeper semantic analysis, we do a basic heuristic
                    if '[' in lhs or lhs in (data_vars if data_block_match else []):
                        bi_code_lines.append(f"    m.dist.{bi_dist}({args}, obs={lhs})")
                    else:
                        bi_code_lines.append(f"    {lhs} = m.dist.{bi_dist}({args}, name='{dist}_{lhs}')")
                
                # Check for assignments
                assign_match = re.search(r'([a-zA-Z0-9_\[\]]+)\s*=\s*([^;]+);', line)
                if assign_match:
                    lhs = assign_match.group(1).strip()
                    rhs = assign_match.group(2).strip()
                    rhs = rhs.replace('inv_logit', 'jax.scipy.special.expit')
                    rhs = rhs.replace('exp', 'jnp.exp')
                    bi_code_lines.append(f"    {lhs} = {rhs}")
                    
        if not (data_block_match or params_block_match or model_block_match):
            assumptions.append("Stan code didn't perfectly match standard block structures, mapping might be incomplete.")
            confidence -= 0.3
            
        return {
            "success": True,
            "bi_code": "\n".join(bi_code_lines),
            "explanation": explanation,
            "assumptions": assumptions,
            "confidence": max(0.0, min(1.0, confidence))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def validate_bi_model(bi_code: str) -> Dict[str, Any]:
    """
    Validates BI Python code syntax and basic API usage using abstract syntax trees.
    
    Args:
        bi_code: The BI Python code block to validate.
        
    Returns:
        Validation results containing syntax check and API usage warnings.
    """
    import ast
    try:
        # Check basic Python Syntax
        tree = ast.parse(bi_code)
        
        errors = []
        warnings = []
        
        # Check if a function named 'model' is defined
        has_model_func = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'model':
                has_model_func = True
                
            # Check for m.dist usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'dist':
                        # Validating m.dist.<dist> call
                        has_name_or_obs = False
                        for kw in node.keywords:
                            if kw.arg in ['name', 'obs']:
                                has_name_or_obs = True
                        if not has_name_or_obs:
                            warnings.append(f"Line {node.lineno}: m.dist call missing 'name=' or 'obs=' keyword argument.")

        if not has_model_func:
            warnings.append("No function named 'model' found. BI expects a 'model' method to fit.")
            
        return {
            "success": True,
            "is_valid_python": True,
            "errors": errors,
            "warnings": warnings,
            "message": "Validation complete."
        }
    except SyntaxError as e:
        return {
            "success": False,
            "is_valid_python": False,
            "error": f"SyntaxError: {str(e)}",
            "line": e.lineno,
            "offset": e.offset,
            "text": e.text
        }
    except Exception as e:
        return {
            "success": False,
            "is_valid_python": False,
            "error": f"Validation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

