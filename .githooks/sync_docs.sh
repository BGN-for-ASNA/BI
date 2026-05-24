#!/bin/bash

# Dynamically resolve paths relative to the script location
# Since this script lives in BI/.githooks/, its parent directory is the BI root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BI_MCP_DIR="$(cd "$BI_DIR/../BI_mcp" 2>/dev/null && pwd)"

SRC_DOCS="$BI_DIR/Documentation/"
DST_DOCS="$BI_MCP_DIR/mcp_server/Documentation/"
CACHE_FILE="$BI_MCP_DIR/_rag_cache.pkl"

# Colors for nice terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}[BI-Sync] Running documentation synchronization helper...${NC}"

# Verify BI_mcp exists
if [ -z "$BI_MCP_DIR" ] || [ ! -d "$BI_MCP_DIR" ]; then
    echo -e "${RED}[BI-Sync] Error: Could not locate BI_mcp repository sibling to $BI_DIR${NC}"
    echo -e "${YELLOW}[BI-Sync] Expected to find it at: $(dirname "$BI_DIR")/BI_mcp${NC}"
    exit 1
fi

# Check the mode (commit vs merge)
MODE=$1
SHOULD_SYNC=false

if [ "$MODE" = "--commit" ]; then
    # Check if Documentation/ changed in the last commit
    if git -C "$BI_DIR" diff-tree --no-commit-id --name-only -r HEAD | grep -E '^Documentation/' > /dev/null; then
        echo -e "${YELLOW}[BI-Sync] Documentation changes detected in local commit.${NC}"
        SHOULD_SYNC=true
    fi
elif [ "$MODE" = "--merge" ]; then
    # Check if Documentation/ changed in the merge/pull
    if git -C "$BI_DIR" diff-tree --no-commit-id --name-only -r ORIG_HEAD HEAD | grep -E '^Documentation/' > /dev/null; then
        echo -e "${YELLOW}[BI-Sync] Documentation changes detected in merge/pull.${NC}"
        SHOULD_SYNC=true
    fi
else
    # Default: sync unconditionally
    echo -e "${YELLOW}[BI-Sync] Running in manual synchronization mode.${NC}"
    SHOULD_SYNC=true
fi

if [ "$SHOULD_SYNC" = true ]; then
    if [ ! -d "$SRC_DOCS" ]; then
        echo -e "${RED}[BI-Sync] Error: Source documentation directory $SRC_DOCS does not exist!${NC}"
        exit 1
    fi

    if [ ! -d "$DST_DOCS" ]; then
        echo -e "${YELLOW}[BI-Sync] Target directory does not exist, creating it: $DST_DOCS${NC}"
        mkdir -p "$DST_DOCS"
    fi

    echo -e "${BLUE}[BI-Sync] Syncing files via rsync...${NC}"
    rsync -av --delete \
        --exclude="/.quarto/" \
        --exclude="**/*.quarto_ipynb" \
        --exclude="*.pkl" \
        --exclude=".git/" \
        "$SRC_DOCS" "$DST_DOCS"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[BI-Sync] Files synchronized successfully!${NC}"
    else
        echo -e "${RED}[BI-Sync] Error: rsync failed!${NC}"
        exit 1
    fi

    # Invalidate RAG cache so it rebuilds on next access
    if [ -f "$CACHE_FILE" ]; then
        echo -e "${BLUE}[BI-Sync] Invalidating RAG cache by deleting $CACHE_FILE...${NC}"
        rm -f "$CACHE_FILE"
        echo -e "${GREEN}[BI-Sync] RAG cache successfully invalidated.${NC}"
    fi

    # Commit changes in BI_mcp if it has a .git repo
    if [ -d "$BI_MCP_DIR/.git" ]; then
        echo -e "${BLUE}[BI-Sync] Checking for staging changes in BI_mcp...${NC}"
        cd "$BI_MCP_DIR" || exit 1
        git add mcp_server/Documentation/

        if ! git diff --cached --quiet; then
            BI_COMMIT_INFO=$(git -C "$BI_DIR" log -1 --format="%h: %s")
            echo -e "${YELLOW}[BI-Sync] Committing changes in BI_mcp: Sync documentation from BI commit ($BI_COMMIT_INFO)${NC}"
            git commit -m "Sync documentation from BI commit ($BI_COMMIT_INFO)"
            echo -e "${GREEN}[BI-Sync] Successfully committed documentation changes in BI_mcp.${NC}"
        else
            echo -e "${GREEN}[BI-Sync] No new documentation changes to commit in BI_mcp.${NC}"
        fi
    fi
else
    echo -e "${GREEN}[BI-Sync] No documentation changes detected. Skipping synchronization. All clear!${NC}"
fi
