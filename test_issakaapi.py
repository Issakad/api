import pytest
from flask import jsonify
from issakaapi import app

# Initialize the test client
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test the home endpoint
def test_home(client):
    response = client.get('/api/')
    assert response.status_code == 200
    assert b"API, model and data loaded" in response.data

# Test the sk_ids endpoint
def test_sk_ids(client):
    response = client.get('/api/sk_ids/')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'data' in data
    assert isinstance(data['data'], list)

# Test the scoring endpoint
def test_scoring(client):
    response = client.get('/api/scoring?SK_ID_CURR=137160', follow_redirects=True)
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'SK_ID_CURR' in data
    assert 'score' in data
    assert 'applicant_data' in data


# Test the neigh_cust endpoint
def test_neigh_cust(client):
    response = client.get('/api/neigh_cust/?SK_ID_CURR=191894')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'X_neigh' in data
    assert 'y_neigh' in data



# Test the features_desc endpoint
def test_features_desc(client):
    response = client.get('/api/features_desc/?SK_ID_CURR=399070')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'data' in data

# Test the features_imp endpoint
def test_features_imp(client):
    response = client.get('/api/features_imp/')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'data' in data

# Test the local_interpretation endpoint
def test_local_interpretation(client):
    response = client.get('/api/local_interpretation/?SK_ID_CURR=399070')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'prediction' in data
    assert 'contribs' in data

if __name__ == '__main__':
    pytest.main()
