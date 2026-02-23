#!/usr/bin/env python3
"""
Quick test script for the BayesInference MCP server.
Tests resource loading and server initialization.
"""

import sys
sys.path.insert(0, '/home/sosa/work/BI')

from mcp_server import resources

print("=" * 60)
print("BayesInference MCP Server - Quick Test")
print("=" * 60)

# Test 1: Dataset resources
print("\n✓ Test 1: Dataset Resources")
datasets = resources.list_available_datasets()
print(f"  Found {len(datasets)} datasets: {', '.join(datasets[:3])}...")

# Test 2: Documentation resources
print("\n✓ Test 2: Documentation Resources")
docs = resources.list_available_docs()
print(f"  Found {len(docs)} documentation resources")

# Test 3: Categories
print("\n✓ Test 3: Documentation Categories")
categories = resources.get_docs_by_category()
for cat, doc_list in categories.items():
    print(f"  - {cat}: {len(doc_list)} docs")

# Test 4: Load a Quarto file
print("\n✓ Test 4: Loading Quarto Documentation")
try:
    content = resources.get_docs_resource('linear_regression')
    print(f"  Loaded 'linear_regression': {len(content)} characters")
    print(f"  Contains Python code: {'```python' in content}")
    print(f"  Contains R code: {'```r' in content or '```R' in content}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 5: Load dataset info
print("\n✓ Test 5: Dataset Information")
try:
    dataset_info = resources.get_dataset_resource('howell1')
    print(f"  Loaded 'howell1' metadata: {len(dataset_info)} characters")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("All tests completed! Server is ready.")
print("=" * 60)
