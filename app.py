import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf

# Flask App Setup
app = Flask(__name__)

# Configure upload folder
app.config["IMAGE_UPLOADS"] = "uploads/"
if not os.path.exists(app.config["IMAGE_UPLOADS"]):
    os.makedirs(app.config["IMAGE_UPLOADS"])

# Load the trained model once at startup
model = tf.keras.models.load_model('breast_cancer_model.h5')
classes = ['benign', 'malignant', 'normal']

# Root route
@app.route('/')
def home():
    return render_template('index.html')

# Flask API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file provided'), 400
    
    file = request.files['file']
    
    try:
        # Save the uploaded image
        file_path = os.path.join(app.config["IMAGE_UPLOADS"], file.filename)
        file.save(file_path)

        # Process the image for prediction
        img = Image.open(file_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        
        # Prepare results for rendering
        return render_template('result.html', 
                               uploaded_image=file_path,
                               prediction=classes[class_index], 
                               confidence=float(prediction[0][class_index]) * 100)

    except Exception as e:
        return render_template('index.html', error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
