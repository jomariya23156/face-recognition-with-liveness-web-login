import numpy as np
import argparse
import cv2
import os

# construct the argument parase and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help='Path to input video')
parser.add_argument('-o', '--output', type=str, required=True, 
                    help='Path to output directory of cropped face images')
parser.add_argument('-d', '--detector', type=str, required=True, 
                    help='Path to OpenCV\'s deep learning face detector')
parser.add_argument('-c', '--confidence', type=float, default=0.5, 
                    help='Confidence of face detection')
parser.add_argument('-s', '--skip', type=int, default=16,
                    help='# of frames to skip before applying face detection and crop')
args = vars(parser.parse_args())

# load our serialized face detector from disk
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'],
                               'res10_300x300_ssd_iter_140000.caffemodel'])
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# open a pointer to the video file stream
# initialize total number of frames read and saved
vs = cv2.VideoCapture(args['input'])
read = 0

# in case of, there are already some images in the folder
if len(os.listdir(args['output'])) > 0:
    latest_file = 0
    for file in os.listdir(args['output']):
        latest_file = max(latest_file,int(file[:file.find('.')]))
    # +1 so it doesn't replace the latest image
    latest_file += 1
else:
    latest_file = 0
saved = latest_file

# loop over frames from the video file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()
    
    # if the frame is not grabbed, then we have reached the end of the video
    if not grabbed:
        break
    
    # increase the number of read frame
    read += 1
    
    # check whether we should process this frame
    # since we want to skip some adjecant frame we have to do this
    if read % args['skip'] != 0:
        continue
    
    # grab the frame dimensions and construct a blob from the frame
    # basically, it does preprocessing image by mean subtraction and scaling
    # (104.0, 177.0, 123.0) is the mean of image in FaceNet
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                 (300,300), (104.0, 177.0, 123.0))
    
    # pass the blob through the NN and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    # ensure at least one face is found
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
            face = frame[startY:endY, startX:endX]

            # write the frame to disk
            p = os.path.sep.join([args['output'], f'{saved}.png'])
            cv2.imwrite(p, face)
            saved += 1
            print(f'[INFO] saved {p} to disk')
            
# clean up
vs.release()
cv2.destroyAllWindows()