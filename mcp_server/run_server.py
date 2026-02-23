#!/home/sosa/work/3.12venv/bin/python
"""
Wrapper script for MCP server with error logging.
This helps debug why the server might exit immediately.
"""
import sys
import traceback

# Add error logging
error_log = "/tmp/mcp_server_error.log"

try:
    # Import and run the server
    from mcp_server.server import main
    import asyncio
    
    # Log startup
    with open(error_log, "w") as f:
        f.write("Server starting...\n")
        f.write(f"Python: {sys.executable}\n")
        f.write(f"CWD: {sys.path}\n")
    
    # Run server
    asyncio.run(main())
    
except Exception as e:
    # Log any errors
    with open(error_log, "a") as f:
        f.write(f"\nError: {e}\n")
        f.write(traceback.format_exc())
    raise
