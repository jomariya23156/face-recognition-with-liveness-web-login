from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to trained model')
parser.add_argument('-l', '--le', type=str, required=True,
                    help='Path to Label Encoder')
parser.add_argument('-d', '--detector', type=str, required=True,
                    help='Path to OpenCV\'s deep learning face detector')
parser.add_argument('-c', '--confidence', type=float, default=0.5,
                    help='minimum probability to filter out weak detections')
args = vars(parser.parse_args())

# load our serialized face detector from disk
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# load the liveness detector model and label encoder from disk
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())

# initialize the video stream and allow camera to warmup
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2) # wait camera to warmup

# iterate over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    # and resize it to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    
    # grab the frame dimensions and convert it to a blob
    # blob is used to preprocess image to be easy to read for NN
    # basically, it does mean subtraction and scaling
    # (104.0, 177.0, 123.0) is the mean of image in FaceNet
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network 
    # and obtain the detections and predictions
    detector_net.setInput(blob)
    detections = detector_net.forward()
    
    # iterate over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e. probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections
        if confidence > args['confidence']:
            # compute the (x,y) coordinates of the bounding box
            # for the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            
            # ensure that the bounding box does not fall outside of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # extract the face ROI and then preprocess it
            # in the same manner as our training data
            face = frame[startY:endY, startX:endX]
            # some error occur here if my face is out of frame and comeback in the frame
            try:
                face = cv2.resize(face, (32,32))
            except:
                break
            face = face.astype('float') / 255.0 
            face = tf.keras.preprocessing.image.img_to_array(face)
            # tf model require batch of data to feed in
            # so if we need only one image at a time, we have to add one more dimension
            # in this case it's the same with [face]
            face = np.expand_dims(face, axis=0)
        
            # pass the face ROI through the trained liveness detection model
            # to determine if the face is 'real' or 'fake'
            # predict return 2 value for each example (because in the model we have 2 output classes)
            # the first value stores the prob of being real, the second value stores the prob of being fake
            # so argmax will pick the one with highest prob
            # we care only first output (since we have only 1 input)
            preds = liveness_model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j] # get label of predicted class
            
            # draw the label and bounding box on the frame
            label = f'{label}: {preds[j]:.4f}'
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
    # show the output fame and wait for a key press
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if 'q' is pressed, stop the loop
    if key == ord('q'):
        break
    
# cleanup
cv2.destroyAllWindows()
vs.stop()
        
    
            
    