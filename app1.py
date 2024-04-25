import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("model.h5")
display_labels1=['area','heatmap','horizontal_bar','horizontal_interval','line','manhattan','map','pie','scatter','scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    # Resize image to match model input shape
    resized_image = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.asarray(resized_image)
    # Normalize pixel values to range [0, 1]
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return "ffff"

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("asdfghgf")
        if 'file' not in request.files:
            return "no file"
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            # Create the uploads folder if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(filename)

            # Load the trained model

            # Preprocess the uploaded image
            img = Image.open(filepath)
            img_array = preprocess_image(img)

            # Perform inference using the model
            predictions = model.predict(img_array)
            predictions_list = predictions.tolist()
            # print(predictions_list)
            predictions_array = np.array(predictions_list)
            
            print(predictions_array)

            # Find the index of the maximum probability
            max_prob_index = np.argmax(predictions_array)
            predicted_label = display_labels1[max_prob_index]

            print("predicted: ", predicted_label)
            # print(predicted_label)
            
            # Print labels and probabilities
            for label, probability in zip(display_labels1, predictions_array[0]):
                print(f"{label}: {probability}")

            # Convert predictions to a human-readable format
            # (e.g., class labels, probabilities)
            # processed_output = process_predictions(predictions)

            # Return the processed output as JSON response
            return jsonify({'class': predicted_label})
        else:
            return 'Invalid file format'
    except Exception as e:
        print(e)
        return 'An error occurred'


if __name__ == '__main__':
    app.run(debug=True)
