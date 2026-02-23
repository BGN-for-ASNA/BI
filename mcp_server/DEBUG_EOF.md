# Debugging MCP Server EOF Error

## Try This Configuration

Use the wrapper script with error logging:

```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "/home/sosa/work/BI/mcp_server/run_server.py",
      "cwd": "/home/sosa/work/BI"
    }
  }
}
```

After trying to connect, check the error log:
```bash
cat /tmp/mcp_server_error.log
```

## Alternative: Simpler Approach

If the wrapper doesn't work, try using `python` directly with explicit module path:

```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "/home/sosa/work/3.12venv/bin/python",
      "args": ["-u", "-m", "mcp_server"],
      "cwd": "/home/sosa/work/BI",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

The `-u` flag makes Python unbuffered, which can help with stdio communication.

## Check for BI Package

The EOF might happen if BI package isn't installed in the venv:

```bash
source /home/sosa/work/3.12venv/bin/activate
cd /home/sosa/work/BI
pip install -e .
```

This ensures the BI package is available when the server tries to import it.

## Common Issues

1. **BI not installed in venv** - Run `pip install -e .` in the BI directory
2. **Buffering issues** - Add `-u` flag and `PYTHONUNBUFFERED=1`
3. **Import errors** - Check `/tmp/mcp_server_error.log` after connection attempt
4. **Path issues** - Ensure `cwd` is `/home/sosa/work/BI` not `/home/sosa/work/BI/mcp_server`

Try the configurations above and check the error log to see what's happening!
