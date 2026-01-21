"""
Tests for the BayesInference MCP server.
"""

import pytest
import asyncio
from mcp_server import server, resources, tools


class TestResources:
    """Test resource functionality."""
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = resources.list_available_datasets()
        assert isinstance(datasets, list)
        assert "howell1" in datasets
        assert "milk" in datasets
        assert "iris" in datasets
    
    def test_list_docs(self):
        """Test listing available documentation."""
        docs = resources.list_available_docs()
        assert isinstance(docs, list)
        assert "getting_started" in docs
        assert "distributions" in docs
    
    def test_get_dataset_resource(self):
        """Test getting dataset resource."""
        dataset_info = resources.get_dataset_resource("howell1")
        assert dataset_info is not None
        assert "Howell1" in dataset_info
    
    def test_get_docs_resource(self):
        """Test getting documentation resource."""
        docs = resources.get_docs_resource("getting_started")
        assert docs is not None
        assert "BayesInference" in docs


class TestTools:
    """Test tool functionality."""
    
    def test_initialize_model(self):
        """Test model initialization."""
        result = tools.initialize_model(platform="cpu", model_id="test")
        assert "success" in result
        # Note: This will fail if BI is not installed, which is expected
    
    def test_load_dataset(self):
        """Test dataset loading."""
        result = tools.load_dataset(dataset_name="howell1", as_dict=False)
        assert "success" in result
        # Note: This will fail if BI is not installed, which is expected


@pytest.mark.asyncio
class TestServer:
    """Test server functionality."""
    
    async def test_list_resources(self):
        """Test server resource listing."""
        resource_list = await server.list_resources()
        assert isinstance(resource_list, list)
        assert len(resource_list) > 0
        
        # Check for dataset resources
        dataset_uris = [str(r.uri) for r in resource_list if str(r.uri).startswith("dataset://")]
        assert len(dataset_uris) > 0
        
        # Check for docs resources
        doc_uris = [str(r.uri) for r in resource_list if str(r.uri).startswith("docs://")]
        assert len(doc_uris) > 0
    
    async def test_read_resource_dataset(self):
        """Test reading dataset resource."""
        content = await server.read_resource("dataset://howell1")
        assert content is not None
        assert "Howell1" in content
    
    async def test_read_resource_docs(self):
        """Test reading documentation resource."""
        content = await server.read_resource("docs://getting_started")
        assert content is not None
        assert "BayesInference" in content
    
    async def test_list_tools(self):
        """Test server tool listing."""
        tool_list = await server.list_tools()
        assert isinstance(tool_list, list)
        assert len(tool_list) > 0
        
        # Check for expected tools
        tool_names = [t.name for t in tool_list]
        assert "initialize_model" in tool_names
        assert "load_dataset" in tool_names
        assert "simulate_data" in tool_names
        assert "fit_model" in tool_names
        assert "get_summary" in tool_names
        assert "sample_posterior" in tool_names
        assert "get_diagnostics" in tool_names
        assert "create_simple_linear_model" in tool_names


def test_import():
    """Test that the module can be imported."""
    import mcp_server
    assert mcp_server is not None
    assert hasattr(mcp_server, '__version__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
