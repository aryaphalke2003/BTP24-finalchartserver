# Open the label file and read its content
with open('ticklabel.txt', 'r') as f_pd:
    Lines = f_pd.readlines()

ticklabelcenters=[]
tickcenters=[]

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


for line in Lines:
    line = line.rstrip()
    line = line.split(" ")
    if line[0] == '7':
        ticklabelcenters.append(line[1]+' '+line[2]+' '+line[3]+' '+line[4])

with open('tick.txt', 'r') as f_pd:
    tLines = f_pd.readlines()

chartlay=[]

for line in tLines:
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

# Mapping each point in y_ticks to the nearest point in y_ticklabels
mapped_points_y = []
for tick in y_ticks:
    nearest_ticklabel = find_nearest_yticklabel(tick, y_ticklabels)
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
    mapped_points_x.append((tick, nearest_ticklabel))
    if(nearest_ticklabel in x_ticklabels):
        x_ticklabels.remove(nearest_ticklabel)  # Remove the mapped point to avoid repetition

# Print the mapped points
for i, (tick, nearest_ticklabel) in enumerate(mapped_points_x, 1):
    print(f"Point {i} in x_ticks: {tick}, mapped to nearest point in x_ticklabels: {nearest_ticklabel}")
    

import cv2 as cv
import os

# Create folders to store cropped images
cropped_x_folder = './croppedx/'
cropped_y_folder = './croppedy/'

# Create the directories if they don't exist
os.makedirs(cropped_x_folder, exist_ok=True)
os.makedirs(cropped_y_folder, exist_ok=True)

# Load the image
image = cv.imread('./data/images/PMC6165091___1.jpg')

# Example bounding box coordinates for x mappings
for i, (a, b) in enumerate(mapped_points_x, 1):
    xlt, ylt, xrb, yrb = b
    # Scale coordinates to image size
    img_height, img_width, _ = image.shape
    xlt, ylt, xrb, yrb = int(xlt * img_width), int(ylt * img_height), int(xrb * img_width), int(yrb * img_height)
    cropped_image = image[ylt:yrb, xlt:xrb]
    output_filename = f'{i}.jpg'  # Use i instead of constant 1 for unique filenames
    output_path = os.path.join(cropped_x_folder, output_filename)
    cv.imwrite(output_path, cropped_image)

# Example bounding box coordinates for y mappings
for i, (a, b) in enumerate(mapped_points_y, 1):
    xlt, ylt, xrb, yrb = b
    # Scale coordinates to image size
    img_height, img_width, _ = image.shape
    xlt, ylt, xrb, yrb = int(xlt * img_width), int(ylt * img_height), int(xrb * img_width), int(yrb * img_height)
    cropped_image = image[ylt:yrb, xlt:xrb]
    output_filename = f'{i}.jpg'  # Use i instead of constant 1 for unique filenames
    output_path = os.path.join(cropped_y_folder, output_filename)
    cv.imwrite(output_path, cropped_image)




