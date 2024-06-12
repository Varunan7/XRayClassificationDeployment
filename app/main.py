from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load models
cnn_model = load_model('models/cnn_model.h5')
dnn_model = load_model('models/dnn_model.h5')

def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert('L')
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    img = preprocess_image(image)

    model_type = request.form.get('model', 'cnn')
    if model_type == 'cnn':
        prediction = cnn_model.predict(img)
    else:
        prediction = dnn_model.predict(img)

    prediction = prediction.tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()
