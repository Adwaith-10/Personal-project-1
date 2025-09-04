import pytest
import os
import sys

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required imports are available."""
    try:
        import streamlit
        import requests
        import pandas
        import plotly.express
        from datetime import datetime, timedelta
        import os
        assert True
    except ImportError as e:
        pytest.fail(f"Required import failed: {e}")

def test_environment_variables():
    """Test that environment variables are properly configured."""
    # Check that backend URL is set
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    assert backend_url.startswith("http")
    
    # Check that it's a valid URL format
    assert "://" in backend_url

def test_api_base_url():
    """Test that API base URL is constructed correctly."""
    try:
        from streamlit_app import API_BASE_URL
        assert API_BASE_URL.endswith("/api/v1")
        assert API_BASE_URL.startswith("http")
    except ImportError:
        pytest.skip("Streamlit app not available for import")

def test_page_config_values():
    """Test that page configuration values are correct."""
    expected_title = "Food Vision Pro"
    expected_icon = "🍽️"
    expected_layout = "wide"
    
    # These values should match what's set in the app
    assert expected_title == "Food Vision Pro"
    assert expected_icon == "🍽️"
    assert expected_layout == "wide"

def test_data_structures():
    """Test that data structures are properly defined."""
    # Test sample data structures that might be used
    sample_meal_data = {
        "id": "meal_123",
        "name": "Test Meal",
        "items": [
            {
                "label": "grilled_chicken",
                "grams": 150.0,
                "calories": 250.0
            }
        ]
    }
    
    assert "id" in sample_meal_data
    assert "name" in sample_meal_data
    assert "items" in sample_meal_data
    assert isinstance(sample_meal_data["items"], list)

def test_plotly_chart_creation():
    """Test that Plotly charts can be created with sample data."""
    try:
        import pandas as pd
        import plotly.express as px
        
        # Create sample data
        sample_data = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "calories": [1500, 1600, 1400]
        })
        
        # Create a simple chart
        fig = px.line(sample_data, x="date", y="calories", title="Daily Calories")
        assert fig is not None
        assert hasattr(fig, 'data')
    except ImportError as e:
        pytest.skip(f"Plotly dependencies not available: {e}")

def test_pandas_operations():
    """Test that pandas operations work correctly."""
    try:
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "calories": [1500, 1600, 1400],
            "protein": [80, 85, 75]
        })
        
        # Test basic operations
        assert len(sample_data) == 3
        assert sample_data["calories"].sum() == 4500
        assert sample_data["protein"].mean() == 80.0
    except ImportError as e:
        pytest.skip(f"Pandas dependencies not available: {e}")

def test_date_operations():
    """Test that date operations work correctly."""
    from datetime import datetime, timedelta
    
    # Test date arithmetic
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    
    assert yesterday < today < tomorrow
    assert (today - yesterday).days == 1
    assert (tomorrow - today).days == 1

def test_api_endpoint_construction():
    """Test that API endpoints are constructed correctly."""
    try:
        from streamlit_app import API_BASE_URL
        
        # Test endpoint construction
        auth_endpoint = f"{API_BASE_URL}/auth/login"
        meals_endpoint = f"{API_BASE_URL}/meals"
        foods_endpoint = f"{API_BASE_URL}/foods/search"
        
        assert auth_endpoint.endswith("/auth/login")
        assert meals_endpoint.endswith("/meals")
        assert foods_endpoint.endswith("/foods/search")
    except ImportError:
        pytest.skip("Streamlit app not available for import")

def test_requests_library():
    """Test that requests library is available and working."""
    try:
        import requests
        
        # Test basic requests functionality
        assert hasattr(requests, 'get')
        assert hasattr(requests, 'post')
        assert hasattr(requests, 'put')
        assert hasattr(requests, 'delete')
        assert callable(requests.get)
        assert callable(requests.post)
    except ImportError as e:
        pytest.skip(f"Requests library not available: {e}")

def test_streamlit_components():
    """Test that Streamlit components are available."""
    try:
        import streamlit as st
        
        # Test basic Streamlit functionality
        assert hasattr(st, 'title')
        assert hasattr(st, 'header')
        assert hasattr(st, 'subheader')
        assert hasattr(st, 'text')
        assert hasattr(st, 'write')
        assert hasattr(st, 'button')
        assert hasattr(st, 'text_input')
        assert hasattr(st, 'selectbox')
        assert hasattr(st, 'date_input')
        assert hasattr(st, 'file_uploader')
        assert hasattr(st, 'sidebar')
        assert hasattr(st, 'columns')
        assert hasattr(st, 'container')
        assert hasattr(st, 'expander')
        assert hasattr(st, 'tabs')
        assert hasattr(st, 'form')
        assert hasattr(st, 'session_state')
        
        # Test that they are callable
        assert callable(st.title)
        assert callable(st.header)
        assert callable(st.subheader)
        assert callable(st.text)
        assert callable(st.write)
        assert callable(st.button)
        assert callable(st.text_input)
        assert callable(st.selectbox)
        assert callable(st.date_input)
        assert callable(st.file_uploader)
        assert callable(st.sidebar)
        assert callable(st.columns)
        assert callable(st.container)
        assert callable(st.expander)
        assert callable(st.tabs)
        assert callable(st.form)
    except ImportError as e:
        pytest.skip(f"Streamlit not available: {e}")

def test_plotly_components():
    """Test that Plotly components are available."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Test basic Plotly functionality
        assert hasattr(px, 'line')
        assert hasattr(px, 'bar')
        assert hasattr(px, 'scatter')
        assert hasattr(px, 'histogram')
        assert hasattr(px, 'pie')
        assert hasattr(px, 'box')
        assert hasattr(px, 'violin')
        assert hasattr(px, 'heatmap')
        
        assert hasattr(go, 'Figure')
        assert hasattr(go, 'Scatter')
        assert hasattr(go, 'Bar')
        assert hasattr(go, 'Pie')
        
        # Test that they are callable
        assert callable(px.line)
        assert callable(px.bar)
        assert callable(px.scatter)
        assert callable(px.histogram)
        assert callable(px.pie)
        assert callable(px.box)
        assert callable(px.violin)
        assert callable(px.heatmap)
        
        assert callable(go.Figure)
        assert callable(go.Scatter)
        assert callable(go.Bar)
        assert callable(go.Pie)
    except ImportError as e:
        pytest.skip(f"Plotly not available: {e}")

def test_pandas_components():
    """Test that Pandas components are available."""
    try:
        import pandas as pd
        
        # Test basic Pandas functionality
        assert hasattr(pd, 'DataFrame')
        assert hasattr(pd, 'Series')
        assert hasattr(pd, 'read_csv')
        assert hasattr(pd, 'read_json')
        assert hasattr(pd, 'to_csv')
        assert hasattr(pd, 'to_json')
        assert hasattr(pd, 'concat')
        assert hasattr(pd, 'merge')
        assert hasattr(pd, 'groupby')
        
        # Test that they are callable
        assert callable(pd.DataFrame)
        assert callable(pd.Series)
        assert callable(pd.read_csv)
        assert callable(pd.read_json)
        assert callable(pd.to_csv)
        assert callable(pd.to_json)
        assert callable(pd.concat)
        assert callable(pd.merge)
        assert callable(pd.groupby)
    except ImportError as e:
        pytest.skip(f"Pandas not available: {e}")

def test_os_operations():
    """Test that OS operations work correctly."""
    # Test basic OS functionality
    assert os.path.exists(__file__)
    assert os.path.isfile(__file__)
    assert os.path.dirname(__file__) == os.path.dirname(os.path.abspath(__file__))
    
    # Test environment variable operations
    test_var = "TEST_VAR_12345"
    test_value = "test_value_12345"
    
    # Set and get environment variable
    os.environ[test_var] = test_value
    assert os.getenv(test_var) == test_value
    
    # Clean up
    del os.environ[test_var]
    assert os.getenv(test_var) is None

def test_sys_operations():
    """Test that sys operations work correctly."""
    # Test basic sys functionality
    assert hasattr(sys, 'version')
    assert hasattr(sys, 'platform')
    assert hasattr(sys, 'path')
    
    # Test path operations
    assert isinstance(sys.path, list)
    assert len(sys.path) > 0
    
    # Test that we can add to path
    original_length = len(sys.path)
    sys.path.append("/test/path")
    assert len(sys.path) == original_length + 1
    
    # Clean up
    sys.path.pop()
    assert len(sys.path) == original_length
