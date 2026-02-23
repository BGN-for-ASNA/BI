# Testing Your MCP Server - Complete Guide

## Option 1: Quick Local Test ✅ (Already Done!)

```bash
# Run the quick test script
cd /home/sosa/work/BI
python3 mcp_server/test_quick.py
```

**Result**: All tests passed! Server can load 34 docs + 10 datasets.

---

## Option 2: MCP Inspector (Interactive Testing)

The MCP Inspector provides a web UI to test your server interactively.

### Install:
```bash
npm install -g @modelcontextprotocol/inspector
```

### Run:
```bash
cd /home/sosa/work/BI/mcp_server
mcp-inspector python3 -m mcp_server
```

This opens a web interface where you can:
- Browse all resources
- Call tools interactively
- See request/response JSON

---

## Option 3: Unit Tests (Automated)

```bash
# Run all unit tests
cd /home/sosa/work/BI
python3 -m pytest mcp_server/test_server.py -v

# Run specific test categories
python3 -m pytest mcp_server/test_server.py::TestResources -v
python3 -m pytest mcp_server/test_server.py::TestTools -v
```

---

## Option 4: Claude Desktop Integration (Real Usage)

### Setup:

1. **Find Claude Desktop config file:**
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add this configuration:**
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

3. **Restart Claude Desktop**

4. **Look for the 🔌 icon** indicating MCP connection

### Test Queries:

Once connected, try asking Claude:

**Basic:**
- "List the available datasets"
- "Show me the howell1 dataset"
- "What documentation is available?"

**Reading Documentation:**
- "Show me how to fit a linear regression model"
- "Explain how to use varying intercepts"
- "What's a Gaussian Process? Show me an example"

**Combining Models:**
- "I have count data with grouping. Show me how to fit a hierarchical Poisson model"
- "How do I fit a binomial model with varying slopes?"
- "Create a zero-inflated Poisson model with an offset term"

**Using Tools:**
- "Load the milk dataset and fit a linear model predicting kcal.per.g from mass"
- "Initialize a model on CPU and show me the summary"

---

## Option 5: Command-Line Testing (stdio mode)

Test the server directly in stdio mode:

```bash
cd /home/sosa/work/BI/mcp_server

# This runs the server (it waits for JSON-RPC input)
python3 -m mcp_server
```

Then send MCP protocol messages (advanced):
```json
{"jsonrpc":"2.0","id":1,"method":"resources/list"}
```

Press `Ctrl+C` to exit.

---

## Option 6: Test Individual Components

### Test resource loading:
```bash
python3 -c "
from mcp_server import resources
print('Docs:', len(resources.list_available_docs()))
print('First doc:', resources.get_docs_resource('linear_regression')[:200])
"
```

### Test tools:
```bash
python3 -c "
from mcp_server import tools
result = tools.load_dataset('howell1')
print('Success:', result.get('success'))
print('Shape:', result.get('shape'))
"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mcp'"
**Fix:**
```bash
pip3 install --user mcp pydantic
```

### Issue: "BayesInference not installed"
**Fix:**
```bash
cd /home/sosa/work/BI
pip3 install -e .
```

### Issue: "Quarto files not loading"
**Check:**
```bash
python3 -c "
from pathlib import Path
docs_path = Path('/home/sosa/work/BI/Documentation')
print('Docs folder exists:', docs_path.exists())
print('Linear regression exists:', (docs_path / '1. Linear Regression for continuous variable.qmd').exists())
"
```

---

## Recommended Testing Flow

1. ✅ **Quick test** (already passed!)
   ```bash
   python3 mcp_server/test_quick.py
   ```

2. **Unit tests**
   ```bash
   python3 -m pytest mcp_server/test_server.py -v
   ```

3. **MCP Inspector** (if you have Node.js)
   ```bash
   npm install -g @modelcontextprotocol/inspector
   mcp-inspector python3 -m mcp_server
   ```

4. **Claude Desktop** (for real usage)
   - Configure as shown above
   - Ask complex Bayesian modeling questions
   - Verify AI can read and combine your docs

---

## Success Criteria

Your server is working correctly if:
- ✅ Quick test passes (DONE!)
- ✅ Unit tests pass (DONE!)
- ✅ Claude Desktop shows 🔌 connection icon
- ✅ AI can list resources
- ✅ AI can read your Quarto documentation
- ✅ AI can combine multiple docs to solve problems

---

## Next Steps

1. **Try Claude Desktop integration** - The real test!
2. **Ask complex questions** - Test if AI combines docs
3. **Add more Quarto examples** - Server automatically uses them
4. **Share feedback** - Iterate on documentation quality
