"""
Utility functions for the MCP server.
"""

import json
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict


def serialize_array(arr: Any) -> Dict[str, Any]:
    """
    Serialize JAX or NumPy arrays to JSON-compatible format.
    
    Args:
        arr: Array to serialize (JAX or NumPy)
        
    Returns:
        Dictionary with array data and metadata
    """
    if isinstance(arr, (jnp.ndarray, np.ndarray)):
        return {
            "type": "array",
            "data": arr.tolist(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype)
        }
    return arr


def serialize_posterior(posterior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize posterior samples to JSON-compatible format.
    
    Args:
        posterior: Dictionary of posterior samples
        
    Returns:
        Serialized posterior data
    """
    serialized = {}
    for key, value in posterior.items():
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            serialized[key] = serialize_array(value)
        else:
            serialized[key] = value
    return serialized


def format_summary_table(summary_df) -> str:
    """
    Format summary DataFrame as a readable string.
    
    Args:
        summary_df: ArviZ summary DataFrame
        
    Returns:
        Formatted string representation
    """
    return summary_df.to_string()


def safe_serialize(obj: Any) -> Any:
    """
    Safely serialize objects for JSON output.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return safe_serialize(obj.__dict__)
    else:
        return obj
