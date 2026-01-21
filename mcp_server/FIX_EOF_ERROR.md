# Fixing MCP Server "EOF" Error in Antigravity

## The Problem

Error: `calling "initialize": EOF`

This happens because the MCP server process exits immediately when the working directory is set incorrectly.

## The Fix

Your Antigravity MCP config needs the correct working directory.

### ❌ **Wrong** (causes EOF error):
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

### ✅ **Correct**:
```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "python3",
      "args": ["-m", "mcp_server"],
      "cwd": "/home/sosa/work/BI"
    }
  }
}
```

**Key difference:** `cwd` should be `/home/sosa/work/BI` (the parent directory), not `/home/sosa/work/BI/mcp_server`.

## Why?

When you run `python3 -m mcp_server`, Python looks for the `mcp_server` package in the current directory. 

- ✅ From `/home/sosa/work/BI` → finds `mcp_server/` subdirectory
- ❌ From `/home/sosa/work/BI/mcp_server` → can't find `mcp_server` module (it's the current dir, not a subdirectory)

## Alternative: Use Script Instead of Module

If you prefer, you can also use the `server.py` script directly:

```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "python3",
      "args": ["server.py"],
      "cwd": "/home/sosa/work/BI/mcp_server"
    }
  }
}
```

## Testing

After fixing the config:

1. **Restart Antigravity** (or reload MCP config)
2. **Check connection** - look for MCP server indicator
3. **Test query**: "List available datasets"

You should see:
- 10 datasets loading successfully
- 34 documentation resources available
- No EOF error!

## Verification Command

Test the server manually from the correct directory:
```bash
cd /home/sosa/work/BI
python3 -m mcp_server
# Server should start and wait for input (Ctrl+C to exit)
```

If this works, your Antigravity config with `"cwd": "/home/sosa/work/BI"` will work too!
