"""Pytest configuration for TICE tests."""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure we're using test-safe settings
    os.environ["TICE_SECRET_KEY"] = "test-secret-key"
    os.environ["TESTING"] = "1"