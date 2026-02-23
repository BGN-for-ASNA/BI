"""
Main MCP Server implementation for BayesInference.
"""

import asyncio
import sys
from typing import Any, Optional
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server
import mcp.server.stdio
import json

from . import resources, tools


# Create MCP server instance
app = Server("bayesinference-server")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources (datasets and documentation)."""
    resource_list = []
    
    # Add dataset resources
    for dataset_name in resources.list_available_datasets():
        dataset_info = resources.DATASETS[dataset_name]
        resource_list.append(
            Resource(
                uri=f"dataset://{dataset_name}",
                name=f"Dataset: {dataset_info['name']}",
                mimeType="application/json",
                description=dataset_info['description']
            )
        )
    
    # Add documentation resources organized by category
    docs_by_category = resources.get_docs_by_category()
    
    for category, doc_names in docs_by_category.items():
        for doc_name in doc_names:
            # Create descriptive names from doc_name
            display_name = doc_name.replace('_', ' ').title()
            resource_list.append(
                Resource(
                    uri=f"docs://{doc_name}",
                    name=f"[{category}] {display_name}",
                    mimeType="text/markdown",
                    description=f"BayesInference {category}: {display_name}"
                )
            )
    
    # Add Stan/BI conversion resources
    resource_list.append(
        Resource(
            uri="bi://stan_conversion_examples",
            name="Stan to BI Conversion Examples",
            mimeType="application/json",
            description="A collection of paired Stan and BI models demonstrating semantic equivalence."
        )
    )
    resource_list.append(
        Resource(
            uri="bi://stan_semantics",
            name="Stan to BI Semantic Mapping",
            mimeType="text/yaml",
            description="Formal specification defining how Stan blocks map to BI API calls."
        )
    )
    
    return resource_list


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific resource by URI."""
    
    if uri.startswith("dataset://"):
        dataset_name = uri.replace("dataset://", "")
        return resources.get_dataset_resource(dataset_name)
    
    elif uri.startswith("docs://"):
        doc_name = uri.replace("docs://", "")
        return resources.get_docs_resource(doc_name)
    
    elif uri == "bi://stan_conversion_examples":
        return resources.get_stan_conversion_examples()
        
    elif uri.startswith("bi://stan_conversion_examples/"):
        example_id = uri.replace("bi://stan_conversion_examples/", "")
        return resources.get_stan_conversion_example(example_id)
        
    elif uri == "bi://stan_semantics":
        return resources.get_stan_semantics()
        
    else:
        raise ValueError(f"Unknown resource URI: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="initialize_model",
            description="Initialize a new BayesInference model instance with specified platform (cpu/gpu/tpu)",
            inputSchema={
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["cpu", "gpu", "tpu"],
                        "description": "Computing platform to use",
                        "default": "cpu"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Identifier for this model instance",
                        "default": "default"
                    },
                    "rand_seed": {
                        "type": "boolean",
                        "description": "Whether to use random seed",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="load_dataset",
            description="Load a built-in dataset (howell1, milk, iris, chimpanzees, reedfrogs, tulips, ucbadmit, trolley, elephants, waffle_divorce)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to load"
                    },
                    "as_dict": {
                        "type": "boolean",
                        "description": "Return as dictionary instead of string representation",
                        "default": False
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="simulate_data",
            description="Simulate data using BI distributions. Code has access to 'm' (bi instance) and 'jnp' (jax.numpy). Use m.dist.<name>(..., sample=True) to generate data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to generate data"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Model instance identifier",
                        "default": "default"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed",
                        "default": 0
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["cpu", "gpu", "tpu"],
                        "description": "Platform to use if model not initialized",
                        "default": "cpu"
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="fit_model",
            description="Fit a Bayesian model using MCMC sampling. Model code should define a function named 'model'. Use the provided 'm' object (bi instance) for distributions (e.g., m.dist.normal).",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_code": {
                        "type": "string",
                        "description": "Python code defining the model function (must define 'model' function)"
                    },
                    "data": {
                        "type": "object",
                        "description": "Dictionary of data to pass to the model. Optional if data is pre-loaded into m.data_on_model."
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Model instance identifier",
                        "default": "default"
                    },
                    "num_warmup": {
                        "type": "integer",
                        "description": "Number of warmup iterations",
                        "default": 500
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of sampling iterations",
                        "default": 500
                    },
                    "num_chains": {
                        "type": "integer",
                        "description": "Number of MCMC chains",
                        "default": 1
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["cpu", "gpu", "tpu"],
                        "description": "Platform override if model not initialized"
                    }
                },
                "required": ["model_code"]
            }
        ),
        Tool(
            name="get_summary",
            description="Get posterior summary statistics for a fitted model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model instance identifier",
                        "default": "default"
                    },
                    "round_to": {
                        "type": "integer",
                        "description": "Number of decimal places",
                        "default": 2
                    },
                    "hdi_prob": {
                        "type": "number",
                        "description": "HDI probability interval",
                        "default": 0.89
                    }
                }
            }
        ),
        Tool(
            name="sample_posterior",
            description="Generate posterior predictive samples from a fitted model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model instance identifier",
                        "default": "default"
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of samples to generate",
                        "default": 1
                    },
                    "remove_obs": {
                        "type": "boolean",
                        "description": "Whether to remove observed data",
                        "default": True
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed",
                        "default": 0
                    }
                }
            }
        ),
        Tool(
            name="get_diagnostics",
            description="Get MCMC diagnostics (R-hat, ESS, etc.) for a fitted model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model instance identifier",
                        "default": "default"
                    }
                }
            }
        ),
        Tool(
            name="create_simple_linear_model",
            description="Convenience tool to create and fit a simple linear regression model",
            inputSchema={
                "type": "object",
                "properties": {
                    "x_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Predictor variable data"
                    },
                    "y_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Response variable data"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Model instance identifier",
                        "default": "default"
                    },
                    "num_warmup": {
                        "type": "integer",
                        "description": "Number of warmup iterations",
                        "default": 500
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of sampling iterations",
                        "default": 500
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["cpu", "gpu", "tpu"],
                        "description": "Platform to use",
                        "default": "cpu"
                    }
                },
                "required": ["x_data", "y_data"]
            }
        ),
        Tool(
            name="convert_stan_to_bi",
            description="Parses a Stan model and applies semantic mapping rules to generate an equivalent BI Python model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "stan_code": {
                        "type": "string",
                        "description": "The raw Stan model code to convert."
                    }
                },
                "required": ["stan_code"]
            }
        ),
        Tool(
            name="validate_bi_model",
            description="Validates BI Python code syntax and basic API usage using abstract syntax trees.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bi_code": {
                        "type": "string",
                        "description": "The BI Python code block to validate."
                    }
                },
                "required": ["bi_code"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool execution."""
    
    try:
        # Route to appropriate tool function
        if name == "initialize_model":
            result = tools.initialize_model(**arguments)
        elif name == "load_dataset":
            result = tools.load_dataset(**arguments)
        elif name == "simulate_data":
            result = tools.simulate_data(**arguments)
        elif name == "fit_model":
            result = tools.fit_model(**arguments)
        elif name == "get_summary":
            result = tools.get_summary(**arguments)
        elif name == "sample_posterior":
            result = tools.sample_posterior(**arguments)
        elif name == "get_diagnostics":
            result = tools.get_diagnostics(**arguments)
        elif name == "create_simple_linear_model":
            result = tools.create_simple_linear_model(**arguments)
        elif name == "convert_stan_to_bi":
            result = tools.convert_stan_to_bi(**arguments)
        elif name == "validate_bi_model":
            result = tools.validate_bi_model(**arguments)
        else:
            result = {
                "success": False,
                "error": f"Unknown tool: {name}"
            }
        
        # Return result as text content
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        import traceback
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }, indent=2)
        )]


async def main():
    """Run the MCP server."""
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
