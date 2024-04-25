import base64
from flask import Flask, request, jsonify
import subprocess
import os
import json
import shutil
import pytesseract
from PIL import Image
import fastwer
import jiwer
from jiwer import wer
from jiwer import cer
import json
import pandas as pd
import math
import cv2
import numpy as np
import os
import json
import pandas as pd
import cv2
import numpy as np
import string
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from collections import Counter
from PIL import Image
from itertools import groupby
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.config.experimental_run_functions_eagerly(True)
import tensorflow as tf
from tensorflow.keras.models import Model

import cv2
import numpy as np


app = Flask(__name__)

# Path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

characters_train = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¢', '£', '¥', '§', '®', '°', 'é', '—', '‘', '’', '“', '”', '€', '™', 'ﬁ', 'ﬂ']
characters = characters_train 



class CTCLayer(layers.Layer):

    def __init__(self, name=None):

        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def ctc_decoder(predictions):
    '''
    input: given batch of predictions from text rec model
    output: return lists of raw extracted text

    '''
    text_list = []
    
    pred_indcies = np.argmax(predictions, axis=2)
    
    for i in range(pred_indcies.shape[0]):
        ans = ""
        
        ## merge repeats
        merged_list = [k for k,_ in groupby(pred_indcies[i])]
        
        ## remove blanks
        for p in merged_list:
            if p != len(characters):
                ans +=characters[int(p)]
        
        text_list.append(ans)
        
    return text_list

def create_inference_model(training_model):
    # Extract the layers till softmax output from the training model
    inference_model = Model(inputs=training_model.get_layer(name="image").input,
                            outputs=training_model.get_layer(name="dense").output)
    
    return inference_model


custom_objects = {'CTCLayer': CTCLayer}
custom_objects_1 = {}

# Load the model with custom objects
with tf.keras.utils.custom_object_scope(custom_objects):
    model_new = load_model('text_recognition.h5')

print(model_new.summary())

chart_recog_model = load_model('model.h5')

inference_model = create_inference_model(model_new)



@app.route('/')
def index():
    return "yoyo"

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    


    # Save uploaded image to a temporary file
    image_path = './yolov5/data/images/temp_image.jpg'
    image_file.save(image_path)

    # Change directory to yolov5
    os.chdir('./yolov5')
    image_path = './data/images/temp_image.jpg'
    # Run YOLOv5 detection script
    yolo_command = ['python', 'detect.py', '--weights', 'runs/train/PROPOSED/weights/best.pt', '--source', image_path, '--save-txt', '--save-crop']
    subprocess.run(yolo_command)

    # Move back to the previous directory
    os.chdir('..')

    # Path to the output folder of YOLOv5
    output_folder = './yolov5/runs/detect'
    
    
    latest_out_folder = get_latest_exp_folder(output_folder)
    temp_path = os.path.join(latest_out_folder, 'crops/')
   

    # Check if Chart title folder is present
    chart_title_folder = os.path.join(temp_path, 'Chart')
    if os.path.exists(chart_title_folder):
        # Move images from Chart title folder to root's data/images/
        image_list = os.listdir(chart_title_folder)
        if image_list:
            image_list.sort(key=lambda x: os.path.getmtime(os.path.join(chart_title_folder, x)), reverse=True)
            latest_image = image_list[0]
            image_to_move = os.path.join(chart_title_folder, latest_image)
            shutil.copy(image_to_move, './data/images/temp_image.jpg')
            image_path = './data/images/temp_image.jpg'
    else:
        return jsonify({'output': "No chart found"})
    
    # Call YOLOv5 detection script with the uploaded image
    command = ['python', 'detect.py', '--weights', 'runs/train/Proposed/weights/best.pt', '--source', image_path, '--save-crop', '--save-txt']
    result = subprocess.run(command, capture_output=True, text=True)
    
    
    # Run OCR on the latest exp folder
    ocr_output = run_ocr_on_latest_exp_folder()
    
    exp_folder = './runs/detect'
    latest_exp_folder = get_latest_exp_folder(exp_folder)
    temp_image_path = os.path.join(latest_exp_folder, 'temp_image.jpg')
    with open(temp_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # return jsonify({'output': ocr_output, 'image_data': encoded_image})
    
    plabel = getchartcat(image_path)
    print("returning")

    return jsonify({'class': plabel,'output': ocr_output})


def process_single_sample(img_path):
    img_width = 128
    img_height = 32
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    return img

def predText(img): 
    # Read characters from the text file
    

    image = np.expand_dims(img, axis=0)  # Add a batch dimension

    # Run prediction
    preds = inference_model.predict(image)

    # Decode CTC output to text
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    # Uses greedy search. For complex tasks, you can use beam search
    decoded_preds, _ = K.ctc_decode(preds, input_length=input_len, greedy=True)
    decoded_preds = decoded_preds[0][0]  # only interested in the first result

    # Convert to string
    out = ''
    for i in range(decoded_preds.shape[0]):
        c = tf.keras.backend.get_value(decoded_preds[i])
        if c < len(characters_train):  # Ensure index doesn't exceed characters_train length
            out += characters_train[c]

    substr = 'ﬂ'
    loc = out.find(substr)
    out = out[:loc]

    return out

def correct_skew(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
   
    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    #print(lines)
    if(lines is None):
#        print('hi')
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)
       
    # Compute median angle
    median_angle = np.median(angles)
   # print(f"Median angle: {median_angle}")
    if(median_angle>30 or median_angle<-30):  
        (h, w) = image.shape[:2]
        # Rotate image to correct skew
        #(h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # Calculate new bounding dimensions
        alpha = np.abs(angle) * np.pi / 180.0
        bound_w = int(h * np.abs(np.sin(alpha)) + w * np.abs(np.cos(alpha)))
        bound_h = int(h * np.abs(np.cos(alpha)) + w * np.abs(np.sin(alpha)))

        # Adjust the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += (bound_w - w) // 2
        M[1, 2] += (bound_h - h) // 2

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC)

        return rotated
    return image

def run_ocr_on_latest_exp_folder():
    # Path to the folder where detect.py saves the exp folder
    exp_folder = './runs/detect'

    # Get the latest exp folder
    latest_exp_folder = get_latest_exp_folder(exp_folder)

    # Path to the crops folder in the latest exp folder
    crops_folder = os.path.join(latest_exp_folder, 'crops')

    # Initialize the dictionary to store OCR results
    ocr_results = {}
    


    # Iterate through each folder in crops folder
    for folder_name in os.listdir(crops_folder):
        folder_path = os.path.join(crops_folder, folder_name)
        if os.path.isdir(folder_path):
            texts = []
            # Iterate through each image in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    # Apply OCR to the image
                    text = apply_ocr(image_path)
                    texts.append(text)
            # Store the texts in the results dictionary
            ocr_results[folder_name] = texts

    return ocr_results

def obtain_recog(img):
    im1=correct_skew(img)
    cv2.imwrite('as.png',im1)
    img=process_single_sample('as.png')
    pred_str=predText(img)
    return pred_str


def get_latest_exp_folder(exp_folder):
    exp_folders = [os.path.join(exp_folder, d) for d in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, d))]
    latest_exp_folder = max(exp_folders, key=os.path.getmtime)
    return latest_exp_folder

def apply_ocr(image_path):
    img = cv2.imread(image_path)
    text=obtain_recog(img)
    return text


def getchartcat(filepath):
    import tensorflow as tf
    from tensorflow import keras
    print("in getch")
    mod = chart_recog_model
    display_labels1=['area','heatmap','horizontal_bar','horizontal_interval','line','manhattan','map','pie','scatter','scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']


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

    img = Image.open(filepath)
    img_array = preprocess_image(img)
    print("here")
    # Perform inference using the model
    predictions = mod.predict(img_array)
    predictions_list = predictions.tolist()
    # print(predictions_list)
    predictions_array = np.array(predictions_list)

    print(predictions_array)

    # Find the index of the maximum probability
    max_prob_index = np.argmax(predictions_array)
    predicted_label = display_labels1[max_prob_index]

    print("predicted: ", predicted_label)
    for label, probability in zip(display_labels1, predictions_array[0]):
        print(f"{label}: {probability}")
    return predicted_label



if __name__ == '__main__':
    app.run(debug=True)
