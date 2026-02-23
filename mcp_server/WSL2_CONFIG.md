# MCP Server Configuration for WSL2 (Antigravity)

## ✅ Correct Configuration for WSL2

Use this in your Antigravity MCP config:

```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "wsl",
      "args": [
        "-d", "Ubuntu-22.04",
        "--cd", "/home/sosa/work/BI",
        "python3", "-m", "mcp_server"
      ]
    }
  }
}
```

## Alternative (If above doesn't work)

```json
{
  "mcpServers": {
    "bayesinference": {
      "command": "wsl",
      "args": ["python3", "-m", "mcp_server"],
      "cwd": "\\\\wsl.localhost\\Ubuntu-22.04\\home\\sosa\\work\\BI"
    }
  }
}
```

## Explanation

Since you're on WSL2 and Antigravity runs on the Windows side:
- **Command**: `wsl` (not `python3`) - this calls into WSL from Windows
- **Distribution**: `Ubuntu-22.04` - your WSL distro name
- **Working Directory**: `/home/sosa/work/BI` (Linux path inside WSL)
- **Windows Path**: `\\wsl.localhost\Ubuntu-22.04\home\sosa\work\BI`

## Testing

After updating the config:

### 1. Test WSL command from Windows PowerShell:
```powershell
wsl -d Ubuntu-22.04 --cd /home/sosa/work/BI python3 -m mcp_server
```

Should start the server (press Ctrl+C to exit).

### 2. Reload Antigravity config
### 3. Test connection - EOF error should be gone!

## Troubleshooting

### If you get "distribution not found":
Check your WSL distro name:
```bash
wsl -l -v
```

The exact name might be:
- `Ubuntu-22.04` (most likely)
- `Ubuntu`
- `Ubuntu-20.04`

Replace in the config accordingly.

### If you get permission errors:
Make sure Python modules are accessible:
```bash
wsl -d Ubuntu-22.04 python3 -c "import mcp_server; print('OK')"
```

### If still getting EOF:
Try the alternative config with Windows path in `cwd`.

## Your Setup Details

- **WSL Version**: WSL2
- **Distribution**: Ubuntu 22.04
- **Linux Path**: `/home/sosa/work/BI`
- **Windows Path**: `\\wsl.localhost\Ubuntu-22.04\home\sosa\work\BI`
- **Python**: `python3` (inside WSL)

Copy the first configuration into your Antigravity MCP config and it should work!
