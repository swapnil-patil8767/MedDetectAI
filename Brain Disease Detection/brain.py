from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = None

def load_tumor_model():
    global model
    try:
        model = load_model("brain_tumor_final2.h5")
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Define class names based on dataset
class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# Class descriptions for better user understanding
class_descriptions = {
    'Glioma': 'Glioma is a type of tumor that originates in the glial cells of the brain or the spine. Glial cells support nerve cells with energy and nutrients and help maintain the blood-brain barrier.',
    'Meningioma': 'Meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull. Most meningiomas are benign, though they can cause serious problems.',
    'No tumor': 'No tumor detected in the brain MRI scan.',
    'Pituitary': 'Pituitary tumors are abnormal growths that develop in the pituitary gland. Most are benign (not cancer) and are called pituitary adenomas. They usually don\'t spread to other parts of the body.'
}

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_prediction(image_path):
    try:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class]) * 100
        
        result = {
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'description': class_descriptions[class_names[predicted_class]],
            'probabilities': {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}
        }
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        # Resize the image to be displayed
        img.thumbnail((400, 400))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

@app.route('/')
def index():
    # Check if model is loaded
    global model
    if model is None:
        load_status = load_tumor_model()
        if not load_status:
            return render_template('error.html', message="Failed to load brain tumor model.")
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Get prediction
        result = get_prediction(file_path)
        
        if result:
            # Convert image to base64 for display
            img_base64 = image_to_base64(file_path)
            result['image'] = img_base64
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Load model on startup
    load_tumor_model()
    app.run(debug=True)