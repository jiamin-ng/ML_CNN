from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

# Load saved state_dict and set up the model
from cnn_fruits import SimpleCNN
try:
    model = SimpleCNN()
    model.load_state_dict(torch.load("cnn_fruits_model.pth"))   # Load the weights
    model.eval()    # Set model to evaluation mode
    print("Model loaded successfully!")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'cnn_fruits_model.pth' exists.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.debug("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    app.logger.debug(f"Received file: {file.filename}, MIME type: {file.mimetype}")

    if not file.mimetype.startswith('image/'):
        app.logger.debug("Uploaded file is not an image")
        return jsonify({'error': 'File must be an image'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)    # Add batch dimension

        # Perform prediction
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).tolist()[0]
        class_idx = torch.argmax(output, dim=1).item()

        # Map index to class name
        class_mapping = ['apple', 'banana', 'orange', 'mixed']
        predicted_class = class_mapping[class_idx]

        return jsonify({
            'class': predicted_class,
            'probabilities': {class_mapping[i]: prob for i, prob in enumerate(probabilities)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)