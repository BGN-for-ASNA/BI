# How AI Assistants Use Your MCP Server

## The Power of MCP: No Need for Hardcoded Examples!

Your MCP server now exposes **30+ complete model examples** from your Quarto documentation. AI assistants can read these examples on-demand and **combine concepts** to solve new problems.

## How It Works

### 1. **Resources = Documentation Library**
Your Quarto files are exposed as MCP resources:
- `docs://linear_regression` → Reads `1. Linear Regression for continuous variable.qmd`
- `docs://poisson_model` → Reads `7. Poisson model.qmd`
- `docs://varying_intercepts` → Reads `13. Varying intercepts.qmd`
- ... and 27+ more examples!

### 2. **AI Reads Documentation When Needed**
When you ask:
> "Fit a hierarchical Poisson model with varying intercepts by group"

The AI:
1. **Reads** `docs://poisson_model` to learn Poisson syntax
2. **Reads** `docs://varying_intercepts` to learn hierarchical structure
3. **Combines** both patterns to create your specific model
4. **Uses** the `fit_model` tool to run it

### 3. **No Hardcoding Required!**
You don't need to create separate tools for:
- ❌ `fit_poisson_model`
- ❌ `fit_hierarchical_model`
- ❌ `fit_poisson_with_varying_intercepts`
- ❌ `fit_network_model_with_blocks`

Instead, you have:
- ✅ One flexible `fit_model` tool
- ✅ 30+ documentation examples as resources
- ✅ AI reads and combines as needed

## Available Documentation Resources

Your server now exposes these categories:

### **Getting Started** (2 docs)
- `getting_started` - Quick start guide
- `introduction` - Package introduction

### **Basic Regression** (4 docs)
- `linear_regression` - Simple linear models
- `multiple_regression` - Multiple predictors
- `interactions` - Interaction terms
- `categorical_predictors` - Categorical variables

### **Generalized Linear Models** (9 docs)
- `binomial_model` - Binary outcomes
- `beta_binomial` - Overdispersed binomial
- `poisson_model` - Count data
- `gamma_poisson` - Negative binomial
- `poisson_offset` - Poisson with exposure
- `categorical_outcomes` - Multinomial response
- `dirichlet_model` - Dirichlet outcomes
- `multinomial_model` - Categorical choice
- `zero_inflated` - Zero-inflated models

### **Advanced Models** (6 docs)
- `survival_analysis` - Time-to-event models
- `varying_intercepts` - Hierarchical intercepts
- `varying_slopes` - Hierarchical slopes
- `gaussian_processes` - GP regression
- `measurement_error` - Errors-in-variables
- `missing_data` - Missing data handling

### **Machine Learning** (4 docs)
- `pca` - Principal Component Analysis
- `gmm` - Gaussian Mixture Models
- `dpmm` - Dirichlet Process Mixtures
- `bnn` - Bayesian Neural Networks

### **Network Models** (5 docs)
- `network_model` - Basic network models
- `network_block_model` - Stochastic block models
- `network_biases` - Network biases
- `network_metrics` - Network analysis
- `nbda` - Network-based diffusion

### **API Reference** (3 docs)
- `api_distributions` - All distributions
- `api_diagnostics` - Diagnostic functions
- `api_manipulation` - Data manipulation

## Example AI Conversations

### Example 1: Combining Models
**User**: "I need a zero-inflated Poisson model with varying intercepts by site"

**AI Process**:
1. Reads `docs://zero_inflated` for ZI structure
2. Reads `docs://poisson_model` for Poisson syntax  
3. Reads `docs://varying_intercepts` for hierarchical structure
4. Combines all three patterns
5. Uses `fit_model` tool with combined model code

### Example 2: Adapting Examples
**User**: "How do I model survival data with hierarchical effects?"

**AI Process**:
1. Reads `docs://survival_analysis` for survival model
2. Reads `docs://varying_intercepts` for hierarchy
3. Adapts survival model to include varying effects
4. Provides complete code example

### Example 3: Learning New Concepts
**User**: "What's a Gaussian Process and how do I use it?"

**AI Process**:
1. Reads `docs://gaussian_processes` (your full Quarto file!)
2. Sees your explanation, code examples, and mathematical details
3. Explains concept to you
4. Can help you adapt the example to your data

## Why This Approach is Powerful

### ✅ **Future-Proof**
- Add new Quarto examples → AI automatically gains new capabilities
- Update documentation → AI uses latest patterns
- No need to update MCP server tools

### ✅ **Flexible**
- AI can combine ANY documentation examples
- Handles edge cases you didn't anticipate
- Adapts to user's specific needs

### ✅ **Maintainable**
- One source of truth: your Quarto docs
- Documentation AND MCP capabilities stay in sync
- Easy to add new model types

### ✅ **Educational**
- AI can explain concepts from your docs
- Provides context and mathematical details
- Helps users learn Bayesian modeling

## Tips for Best Results

### 1. **Ask for Examples**
Instead of:
> "Fit a model"

Try:
> "Show me an example of a Poisson model with offset term"

The AI will read your docs and provide a complete working example.

### 2. **Request Combinations**
> "Combine varying intercepts with a binomial model for my grouped binary data"

The AI reads both docs and merges the patterns.

### 3. **Reference Documentation**
> "Following the gaussian_processes example, help me model this time series"

The AI will use that specific doc as the template.

### 4. **Leverage API Docs**
> "What distributions are available for count data?"

The AI reads `docs://api_distributions` and lists appropriate options.

## Technical Details

### Resource URIs
All documentation uses the pattern:
```
docs://{doc_name}
```

For example:
- `docs://linear_regression`
- `docs://poisson_model`
- `docs://bnn`

### Loading Strategy
1. **First**: Try to load from Quarto file in `Documentation/`
2. **Fallback**: Use hardcoded documentation if file not found
3. **Result**: AI always gets content, even if Quarto files move

### File Mapping
The server maps doc names to Quarto files:
```python
"linear_regression" → "1. Linear Regression for continuous variable.qmd"
"poisson_model" → "7. Poisson model.qmd"
"varying_intercepts" → "13. Varying intercepts.qmd"
```

## Conclusion

Your MCP server is now a **comprehensive Bayesian modeling assistant**. The AI has access to:
- 30+ complete model examples
- Full explanations and mathematical details
- Code in Python, R, and Julia
- Your expertise encoded in the Quarto docs

**No hardcoding needed** - the AI reads, learns, and adapts your documentation to solve new problems!
