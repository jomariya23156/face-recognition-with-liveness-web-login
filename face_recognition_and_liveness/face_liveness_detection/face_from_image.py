import numpy as np
import argparse
import os
import cv2

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to input image')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Path to output directory of cropped face')
parser.add_argument('-d', '--detector', type=str, required=True,
                    help='Path to OpenCV\'s face detector')
parser.add_argument('-c', '--confidence', type=int, default=0.5,
                    help='Confidence of face detection')
args = vars(parser.parse_args())

print('[INFO] loading face detector')
proto_path = os.path.sep.join([args['detector'],'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'],
                               'res10_300x300_ssd_iter_140000.caffemodel'])

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# read the image
image = cv2.imread(args['input'])

# image name
if len(os.listdir(args['output'])) > 0:
    latest_file = 0
    for file in os.listdir(args['output']):
        latest_file = max(latest_file, int(file[:file.find('.')]))
    # +1 so it doesn't replace the latest image in the directory
    latest_file += 1
else:
    latest_file = 0
    
saved_name = latest_file

# construct a blob from the image (preprocess image)
# basically, it does mean subtraction and scaling
# (104.0, 177.0, 123.0) is the mean of image in FaceNet
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0,
                             (300,300), (104.0, 177.0, 123.0))

# pass the blob through the NN and obtain the detections
net.setInput(blob)
detections = net.forward()

# ensure atleast 1 face it detected
if len(detections) > 0:
    # we're making the assumption that each image has ONLY ONE face,
    # so find the bounding box with the largest probability
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]
    
    # ensure that the detection with the highest probability 
    # pass our minumum probability threshold (helping filter out some weak detections)
    if confidence > args['confidence']:
        # compute the (x,y) coordinates of the bounding box
        # for the face and extract face ROI
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        face = image[startY:endY, startX:endX]
        
        # write the image to disk
        p = os.path.sep.join([args['output'], f'{saved_name}.png'])
        cv2.imwrite(p, face)
        print(f'[INFO] saved {p} to disk')
        
