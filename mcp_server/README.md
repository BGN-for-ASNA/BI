# BayesInference MCP Server

Model Context Protocol (MCP) server for the BayesInference Bayesian modeling package. This server exposes BayesInference functionality to AI assistants and LLMs through the MCP protocol.

## Features

- **Resources**: Access to 10 built-in datasets and **34+ model examples** from Quarto documentation
- **Tools**: Model initialization, fitting, sampling, and diagnostics
- **Platform Support**: CPU, GPU, and TPU acceleration via JAX
- **Documentation**: Complete model examples including linear regression, GLMs, hierarchical models, Gaussian processes, network models, and more

## Installation

### Prerequisites

1. Install the BayesInference package:
```bash
pip install BayesInference
```

2. Install the MCP server:
```bash
cd mcp_server
pip install -e .
```

## Usage

### Running the Server

The server can be run in stdio mode for integration with MCP clients:

```bash
python -m mcp_server
```

### Configuration for Claude Desktop

Add the following to your Claude Desktop configuration file:

**On macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Windows**: `%APPDATA%/Claude/claude_desktop_config.json`
**On Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/BI/mcp_server"
    }
  }
}
```

Replace `/path/to/BI/mcp_server` with the actual path to your mcp_server directory.

## Available Resources

The server provides the following resources:

### Datasets
- `dataset://howell1` - Demographic data from !Kung San people
- `dataset://milk` - Primate milk composition data
- `dataset://iris` - Classic iris dataset
- `dataset://chimpanzees` - Chimpanzee behavioral data
- `dataset://reedfrogs` - Reed frog survival experiments
- `dataset://tulips` - Tulip growth experiments
- `dataset://ucbadmit` - UC Berkeley admissions (Simpson's paradox)
- `dataset://trolley` - Trolley problem moral judgment data
- `dataset://elephants` - Elephant matriarch data
- `dataset://waffle_divorce` - Waffle House and divorce rates

### Documentation (34 Resources)

The server provides access to your complete Quarto documentation, organized by category:

#### Getting Started (2 docs)
- `docs://getting_started` - Quick start guide
- `docs://introduction` - Package introduction

#### Basic Regression (4 docs)
- `docs://linear_regression` - Simple linear models
- `docs://multiple_regression` - Multiple predictors
- `docs://interactions` - Interaction terms
- `docs://categorical_predictors` - Categorical variables

#### Generalized Linear Models (9 docs)
- `docs://binomial_model`, `docs://beta_binomial`, `docs://poisson_model`
- `docs://gamma_poisson`, `docs://poisson_offset`, `docs://categorical_outcomes`
- `docs://dirichlet_model`, `docs://multinomial_model`, `docs://zero_inflated`

#### Advanced Models (6 docs)
- `docs://survival_analysis`, `docs://varying_intercepts`, `docs://varying_slopes`
- `docs://gaussian_processes`, `docs://measurement_error`, `docs://missing_data`

#### Machine Learning (4 docs)
- `docs://pca`, `docs://gmm`, `docs://dpmm`, `docs://bnn`

#### Network Models (5 docs)
- `docs://network_model`, `docs://network_block_model`, `docs://network_biases`
- `docs://network_metrics`, `docs://nbda`

#### API Reference (3 docs)
- `docs://api_distributions`, `docs://api_diagnostics`, `docs://api_manipulation`

> **Note**: AI assistants can read and **combine** these examples to solve complex problems. No need for hardcoded tools for every model type!

## Available Tools

### 1. initialize_model
Initialize a new BayesInference model instance.

**Parameters:**
- `platform` (string): "cpu", "gpu", or "tpu" (default: "cpu")
- `model_id` (string): Identifier for this model (default: "default")
- `rand_seed` (boolean): Use random seed (default: true)

### 2. load_dataset
Load a built-in dataset.

**Parameters:**
- `dataset_name` (string, required): Name of the dataset
- `as_dict` (boolean): Return as dictionary (default: false)

### 3. fit_model
Fit a Bayesian model using MCMC sampling.

**Parameters:**
- `model_code` (string, required): Python code defining the model function
- `data` (object, required): Dictionary of data for the model
- `model_id` (string): Model instance identifier (default: "default")
- `num_warmup` (integer): Warmup iterations (default: 500)
- `num_samples` (integer): Sampling iterations (default: 500)
- `num_chains` (integer): Number of chains (default: 1)
- `platform` (string): Platform override

### 4. get_summary
Get posterior summary statistics.

**Parameters:**
- `model_id` (string): Model instance identifier (default: "default")
- `round_to` (integer): Decimal places (default: 2)
- `hdi_prob` (number): HDI probability (default: 0.89)

### 5. sample_posterior
Generate posterior predictive samples.

**Parameters:**
- `model_id` (string): Model instance identifier (default: "default")
- `num_samples` (integer): Number of samples (default: 1)
- `remove_obs` (boolean): Remove observed data (default: true)
- `seed` (integer): Random seed (default: 0)

### 6. get_diagnostics
Get MCMC diagnostics (R-hat, ESS).

**Parameters:**
- `model_id` (string): Model instance identifier (default: "default")

### 7. create_simple_linear_model
Convenience tool to create and fit a simple linear regression.

**Parameters:**
- `x_data` (array, required): Predictor data
- `y_data` (array, required): Response data
- `model_id` (string): Model identifier (default: "default")
- `num_warmup` (integer): Warmup iterations (default: 500)
- `num_samples` (integer): Sampling iterations (default: 500)
- `platform` (string): Platform to use (default: "cpu")

## Example Usage with AI Assistant

Once configured, you can ask your AI assistant questions like:

- "Load the howell1 dataset and show me the first few rows"
- "Create a simple linear regression model with x=[1,2,3,4,5] and y=[2,4,6,8,10]"
- "Fit a Bayesian model to predict height from weight using the howell1 dataset"
- "Show me the posterior summary for the fitted model"
- "Generate posterior predictive samples"

## Development

### Running Tests

```bash
cd mcp_server
pytest test_server.py -v
```

## License

GPL-3.0-or-later (same as BayesInference package)

## Links

- [BayesInference GitHub](https://github.com/BGN-for-ASNA/BI)
- [Model Context Protocol](https://modelcontextprotocol.io)
