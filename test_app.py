import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_search_valid_query(client):
    response = client.post("/search", json={"query": "neural networks"})
    assert response.status_code == 200
    data = response.get_json()
    assert "results" in data
    assert len(data["results"]) > 0

def test_search_missing_query(client):
    response = client.post("/search", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Missing query"
