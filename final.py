import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
from flask_cors import CORS

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
       
from tensorflow import keras
        
    

import cv2
import numpy as np

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cclassmodel = tf.keras.models.load_model("./CHARTCLASS/model.h5")
display_labels1=['area','heatmap','horizontal_bar','horizontal_interval','line','manhattan','map','pie','scatter','scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']

label_classes = ['axis_title','chart_title','legend_label','legend_title','mark_label','other','tick_grouping','tick_label','value_label']
label_map = {
    0: 'axis_title',
    1: 'chart_title',
    2: 'legend_label',
    3: 'legend_title',
    4: 'mark_label',
    5: 'other',
    6: 'tick_grouping',
    7: 'tick_label',
    8: 'value_label'
}
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
   

# Load the model with custom objects
with tf.keras.utils.custom_object_scope(custom_objects):
        model_new = load_model('./OCR/text_recognition.h5')

inference_model = create_inference_model(model_new)

cclassmod = load_model('./CHARTCLASS/model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




def getcclass(image_path):

        def preprocess_image(image):
            resized_image = image.resize((224, 224))
            img_array = np.asarray(resized_image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array

        img = Image.open(image_path)
        img_array = preprocess_image(img)
        print("cclass")
        predictions = cclassmod.predict(img_array)
        predictions_list = predictions.tolist()
        predictions_array = np.array(predictions_list)

        max_prob_index = np.argmax(predictions_array)
        predicted_label = display_labels1[max_prob_index]

        print("predicted: ", predicted_label)
        for label, probability in zip(display_labels1, predictions_array[0]):
            print(f"{label}: {probability}")
        return predicted_label



def get_latest_exp_folder(exp_folder):
    exp_folders = [os.path.join(exp_folder, d) for d in os.listdir(exp_folder) if os.path.isdir(os.path.join(exp_folder, d))]
    latest_exp_folder = max(exp_folders, key=os.path.getmtime)
    return latest_exp_folder

    
def getocr(image):
    
    def process_single_sample(img_path):
        img_width = 128
        img_height = 32
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_height, img_width])
        return img
 
    def predText(img): 
        # Read characters from the text file
        image = np.expand_dims(img, axis=0)  # Add a batch dimension

        # Run prediction2
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
            return image
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
        
        # Compute median angle
        median_angle = np.median(angles)
    
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


    def obtain_recog(img):
        im1 = correct_skew(img)
        img = process_single_sample(im1)
        pred_str = predText(img)
        return pred_str

    return obtain_recog(image)

class DetectionObject:
    def __init__(self, xlt, ylt, xrb, yrb, role, text,mapped_tick=None,axis=None):
        self.xlt = xlt
        self.ylt = ylt
        self.xrb = xrb
        self.yrb = yrb
        self.role = role
        self.text = text
        self.mapped_tick = mapped_tick
        self.axis = axis
        
        

def func(x,y,w,h):
    x=float(x)
    y=float(y)
    w=float(w)
    h=float(h)
    x = 2 * x
    y = 2 * y
    x_max_pd = (x + w) / 2
    y_max_pd = (h + y) / 2
    x_min_pd = max(0, x_max_pd - w)
    y_min_pd = max(0, y_max_pd - h)
    return x_min_pd,y_min_pd,x_max_pd,y_max_pd



def read_single_txt_in_folder(folder_path):
    # Get list of files in the folder
    files = os.listdir(folder_path)
    
    # Filter only the txt files
    txt_files = [file for file in files if file.endswith('.txt')]
    
    if len(txt_files) == 1:  # If only one txt file is found
        txt_file_path = os.path.join(folder_path, txt_files[0])
        with open(txt_file_path, 'r') as f:
            label_lines = f.readlines()
        return label_lines
    else:
        raise ValueError("Expected exactly one txt file in the folder, but found {}.".format(len(txt_files)))


import math

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to find the nearest point in ticklabels for a given point in ticks
def find_nearest_yticklabel(tick, ticklabels):
    min_distance = float('inf')
    nearest_ticklabel = None
    
    for ticklabel in ticklabels:
        distance = calculate_distance(   (tick[0], (tick[1] + tick[3]) / 2)   ,  (ticklabel[2], (ticklabel[1] + ticklabel[3]) / 2)  )
        if distance < min_distance:
            min_distance = distance
            nearest_ticklabel = ticklabel
    
    return nearest_ticklabel

def find_nearest_xticklabel(tick, ticklabels):
    min_distance = float('inf')
    nearest_ticklabel = None
    
    for ticklabel in ticklabels:
        distance = calculate_distance(    ((tick[0]+tick[2])/2, tick[3])  , ((ticklabel[0]+ticklabel[2])/2, ticklabel[1])  )
        if distance < min_distance:
            min_distance = distance
            nearest_ticklabel = ticklabel
    
    return nearest_ticklabel

import cv2 as cv
           
            
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
            
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            print(image_path)

            predicted_label = getcclass(image_path)
            
            os.chdir('./TXTLBL')
            ycmd = ['python','detect.py','--weights','best.pt','--source',image_path,'--save-txt']
            subprocess.run(ycmd)
            exp_folder = './runs/detect'
            latest_exp_folder = get_latest_exp_folder(exp_folder)
            folderpath = os.path.join(latest_exp_folder, 'labels')
            label_lines = read_single_txt_in_folder(folderpath)
            os.chdir('..')
            
            os.chdir('./AABP')
            ycmd = ['python','detect.py','--weights','best.pt','--source',image_path,'--save-txt']
            subprocess.run(ycmd)
            exp_folder = './runs/detect'
            latest_exp_folder = get_latest_exp_folder(exp_folder)
            folderpath = os.path.join(latest_exp_folder, 'labels')
            tick_lines = read_single_txt_in_folder(folderpath)
            os.chdir('..')
            
            ticklabelcenters=[]
            chartlay=[]
            tickcenters=[]

            label_objects=[]
            
            img = cv.imread(image_path)

            for line in label_lines:
                line = line.rstrip()
                line = line.split(" ")
                if line[0] == '7':
                    ticklabelcenters.append(line[1]+' '+line[2]+' '+line[3]+' '+line[4])
                else:
                    a,b,c,d = func(line[1],line[2],line[3],line[4])
                    img_height, img_width, _ = img.shape
                    a,b,c,d = int(a * img_width), int(b * img_height), int(c * img_width), int(d * img_height)
                    cropped_image = img[b:d, a:c]
                    txt = getocr(cropped_image)
                    label_objects.append(DetectionObject(a, b, c, d, label_map[int(line[0])], txt))
        

            for line in tick_lines:
                line = line.rstrip()
                line = line.split(" ")
                if line[0] == '1':
                    a,b,c,d  = func(line[1],line[2],line[3],line[4])
                    str1 = str(a)+' '+str(b)+' '+str(c)+' '+str(d)
                    chartlay.append(str1)
                else:
                    tickcenters.append(line[1]+' '+line[2]+' '+line[3]+' '+line[4])

            x_ticks=[]
            x_ticklabels=[]
            y_ticks=[]
            y_ticklabels=[]

            xmn,ymn,xmx,ymx = list(map(float, chartlay[0].split()))

            for item in ticklabelcenters:
                values = list(map(float, item.split()))
                # is center's x coordinate left of threshold
                if values[0] < xmn and values[1]< ymx: 
                    y_ticklabels.append(func(values[0],values[1],values[2],values[3]))
                else :
                    x_ticklabels.append(func(values[0],values[1],values[2],values[3]))
                    
            for item in tickcenters:
                values = list(map(float, item.split()))
                # is center's x coordinate left of threshold
                if values[0] < xmn : 
                    y_ticks.append(func(values[0],values[1],values[2],values[3]))
                else :
                    x_ticks.append(func(values[0],values[1],values[2],values[3]))
        

            # Mapping each point in y_ticks to the nearest point in y_ticklabels
            mapped_points_y = []
            for tick in y_ticks:
                nearest_ticklabel = find_nearest_yticklabel(tick, y_ticklabels)
                a,b,c,d = nearest_ticklabel
                label_objects.append(DetectionObject(a, b, c, d, label_map[int(line[0])], txt,tick,'y'))
                mapped_points_y.append((tick, nearest_ticklabel))
                if(nearest_ticklabel in y_ticklabels):
                    y_ticklabels.remove(nearest_ticklabel)  # Remove the mapped point to avoid repetition

            # Print the mapped points
            for i, (tick, nearest_ticklabel) in enumerate(mapped_points_y, 1):
                print(f"Point {i} in y_ticks: {tick}, mapped to nearest point in y_ticklabels: {nearest_ticklabel}")
    

            # Mapping each point in x_ticks to the nearest point in x_ticklabels
            mapped_points_x = []
            for tick in x_ticks:
                nearest_ticklabel = find_nearest_xticklabel(tick, x_ticklabels)
                a,b,c,d = nearest_ticklabel
                label_objects.append(DetectionObject(a, b, c, d, label_map[int(line[0])], txt,tick,'x'))
                mapped_points_x.append((tick, nearest_ticklabel))
                if(nearest_ticklabel in x_ticklabels):
                    x_ticklabels.remove(nearest_ticklabel)  # Remove the mapped point to avoid repetition

            # Print the mapped points
            for i, (tick, nearest_ticklabel) in enumerate(mapped_points_x, 1):
                print(f"Point {i} in x_ticks: {tick}, mapped to nearest point in x_ticklabels: {nearest_ticklabel}")
    
            for obj in label_objects:
                print(obj.xlt,obj.ylt,obj.xrb,obj.yrb,obj.role,obj.text,obj.mapped_tick,obj.axis)
            
            detected_objects = []

            # Loop through label_objects and populate detected_objects list
            for obj in label_objects:
                detected_object = {
                    'xlt': obj.xlt,
                    'ylt': obj.ylt,
                    'xrb': obj.xrb,
                    'yrb': obj.yrb,
                    'role': obj.role,
                    'text': obj.text,
                    'mapped_tick': obj.mapped_tick,
                    'axis': obj.axis
                }
                detected_objects.append(detected_object)

            # Return JSON response
            return jsonify({
                'image': filename,
                'class': predicted_label,
                'data': detected_objects
            })


        else:
            return 'Invalid file format'
    
    except Exception as e:
        print(e)
        return 'An error occurred'

if __name__ == '__main__':
    app.run(debug=True)