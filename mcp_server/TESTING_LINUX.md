# MCP Server Testing Without Claude Desktop

Since Claude Desktop is not available on Linux yet, here are your testing options:

## ✅ **Automated Testing** (Works Now!)

```bash
cd /home/sosa/work/BI

# Quick test (already passed!)
python3 mcp_server/test_quick.py

# Full unit tests
python3 -m pytest mcp_server/test_server.py -v
```

## 🌐 **MCP Inspector** (Recommended for Linux)

Interactive web-based testing tool:

### Install:
```bash
# Requires Node.js
npm install -g @modelcontextprotocol/inspector
```

### Run:
```bash
cd /home/sosa/work/BI/mcp_server
mcp-inspector python3 -m mcp_server
```

Opens a web interface where you can:
- Browse all 34 documentation resources
- List all 10 datasets
- Call tools interactively
- See JSON request/response

## 🔧 **Manual Testing**

### Test Resources:
```bash
python3 -c "
from mcp_server import resources
print('Available docs:', resources.list_available_docs())
print()
print('Loading Poisson model example:')
print(resources.get_docs_resource('poisson_model')[:500])
"
```

### Test Tools:
```bash
python3 -c "
from mcp_server import tools
result = tools.load_dataset('howell1')
print('Dataset loaded:', result['success'])
print('Shape:', result['shape'])
print('Columns:', result['columns'])
"
```

### Test Model Fitting (Full Example):
```bash
python3 -c "
from mcp_server import tools
import json

# Load dataset
dataset = tools.load_dataset('howell1', as_dict=True)
print('Dataset loaded')

# Initialize model
init = tools.initialize_model(platform='cpu', model_id='test')
print('Model initialized:', init['success'])
"
```

## 📱 **Other MCP-Compatible Clients**

### Cursor IDE
If you have Cursor installed, add to settings:
```json
{
  "mcp": {
    "servers": {
      "bayesinference": {
        "command": "python3",
        "args": ["-m", "mcp_server"],
        "cwd": "/home/sosa/work/BI/mcp_server"
      }
    }
  }
}
```

### Windsurf IDE
Similar MCP support - check their docs for config location.

## 🐍 **Python Script Testing**

Create a test script that simulates AI usage:

```python
#!/usr/bin/env python3
from mcp_server import resources, tools

# Simulate AI reading documentation
print("AI: User asked about Poisson models")
print("AI: Reading documentation...")
doc = resources.get_docs_resource('poisson_model')
print(f"AI: Loaded Poisson documentation ({len(doc)} chars)")
print(f"AI: Found Python example: {'```python' in doc.lower() or '```{python}' in doc}")
print()

# Simulate AI loading dataset
print("AI: Loading example dataset...")
result = tools.load_dataset('kline')
if result['success']:
    print(f"AI: Dataset has {result['shape'][0]} rows and {result['shape'][1]} columns")
    print(f"AI: Columns: {result['columns']}")
print()

# Simulate AI combining docs
print("AI: User asked for hierarchical Poisson model")
print("AI: Reading multiple docs...")
poisson_doc = resources.get_docs_resource('poisson_model')
hierarchical_doc = resources.get_docs_resource('varying_intercepts')
print(f"AI: Loaded Poisson ({len(poisson_doc)} chars) + Hierarchical ({len(hierarchical_doc)} chars)")
print("AI: Can now combine these patterns!")
```

## 🌍 **Web-Based Alternative**

Use Claude on https://claude.ai (web version) and:
1. Manually copy documentation when needed
2. Use the Python package directly
3. Wait for Claude Desktop Linux support

## 📊 **Your Server Status**

✅ **Working perfectly:**
- 34 documentation resources from Quarto
- 10 datasets available
- All tools functional
- Tests passing

🔜 **When Claude Desktop for Linux is available:**
The config would go in: `~/.config/Claude/claude_desktop_config.json`

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

## 💡 **Recommendation**

For now, use:
1. **Automated tests** - Verify everything works
2. **MCP Inspector** - Interactive testing if you have Node.js
3. **Direct Python usage** - Your BayesInference package works standalone!

Your MCP server is ready and waiting for when you get an MCP-compatible client! 🎉
