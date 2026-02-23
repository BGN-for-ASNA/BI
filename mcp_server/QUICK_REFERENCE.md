# MCP Server with Quarto Documentation - Quick Reference

## Summary

Your MCP server now exposes **34 documentation resources** from your Quarto files, organized into 7 categories. AI assistants can read these on-demand and combine concepts to solve complex problems.

## Key Changes Made

### 1. Enhanced `resources.py`
- Maps 30+ doc names to Quarto files
- Loads `.qmd` files from `Documentation/` folder
- Falls back to basic docs if files not found
- Organizes docs into 7 categories

### 2. Updated `server.py`
- Lists all 34 documentation resources
- Shows category in resource names: `[Category] Doc Name`
- Better descriptions for discoverability

### 3. Added `INTEGRATION_GUIDE.md`
- Explains how AI uses the documentation
- Shows why you don't need hardcoded examples
- Provides example AI conversations
- Lists all available resources

## Available Documentation (34 Resources)

**Getting Started** (2)
- getting_started, introduction

**Basic Regression** (4)
- linear_regression, multiple_regression, interactions, categorical_predictors

**Generalized Linear Models** (9)
- binomial_model, beta_binomial, poisson_model, gamma_poisson, poisson_offset, categorical_outcomes, dirichlet_model, multinomial_model, zero_inflated

**Advanced Models** (6)
- survival_analysis, varying_intercepts, varying_slopes, gaussian_processes, measurement_error, missing_data

**Machine Learning** (4)
- pca, gmm, dpmm, bnn

**Network Models** (5)
- network_model, network_block_model, network_biases, network_metrics, nbda

**API Reference** (3)
- api_distributions, api_diagnostics, api_manipulation

## How to Use

### Configuration
Add to Claude Desktop config:
```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "python3",
      "args": ["-m", "mcp_server"],
      "cwd": "/home/sosa/work/BI/mcp_server"
    }
  }
}
```

### Example Queries

**Simple:**
> "Show me how to fit a linear regression model"
→ AI reads `docs://linear_regression`

**Combined:**
> "I need a Poisson model with varying intercepts"
→ AI reads `docs://poisson_model` + `docs://varying_intercepts`

**Advanced:**
> "Explain Gaussian Processes and show me an example"
→ AI reads `docs://gaussian_processes` (your full Quarto file with math!)

## Why This Works

✅ **No hardcoding needed** - AI reads and combines your docs  
✅ **Future-proof** - Add new Quarto files → AI gains new capabilities  
✅ **Flexible** - AI adapts examples to specific needs  
✅ **Educational** - Full explanations + code + math from your docs  

## Testing

```bash
# Test resource loading
python3 -c "from mcp_server import resources; print(len(resources.list_available_docs()))"
# Output: 34

# Test Quarto file loading
python3 -c "from mcp_server import resources; print('OK' if len(resources.get_docs_resource('linear_regression')) > 1000 else 'FAIL')"
# Output: OK
```

## Files Modified/Created

1. `mcp_server/resources.py` - Enhanced with Quarto integration
2. `mcp_server/server.py` - Updated resource listing
3. `mcp_server/INTEGRATION_GUIDE.md` - Detailed explanation (NEW)
4. `mcp_server/QUICK_REFERENCE.md` - This file (NEW)

For detailed information, see `INTEGRATION_GUIDE.md`.
