import pytest
import io
from app import app
from unittest.mock import patch
from PIL import Image
import torch

@pytest.fixture
def client():
    """Flask test client setup"""
    app.config['TESTING'] = True
    client = app.test_client()
    return client


def create_test_image():
    """Create an in-memory test image"""
    image = Image.new('RGB', (64, 64), color='white')
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    img_io.seek(0)
    return img_io


@patch("app.model")
def test_predict_valid_image(mock_model, client):
    """Test valid image upload"""
    mock_model.return_value = torch.tensor([[0.1, 0.2, 0.6, 0.1]])  # Mock output

    data = {
        'file': (create_test_image(), 'test.png')
    }
    response = client.post('/predict', content_type='multipart/form-data', data=data)

    assert response.status_code == 200
    json_data = response.get_json()
    assert 'class' in json_data
    assert 'probabilities' in json_data
    assert isinstance(json_data['probabilities'], dict)


def test_predict_no_file(client):
    """Test API when no file is uploaded"""
    response = client.post('/predict', content_type='multipart/form-data', data={})

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == 'No file uploaded'


def test_predict_invalid_file(client):
    """Test uploading a non-image file"""
    data = {
        'file': (io.BytesIO(b"This is not an image"), 'test.txt')
    }
    response = client.post('/predict', content_type='multipart/form-data', data=data)

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == 'File must be an image'
