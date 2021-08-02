from imutils.video import VideoStream
import face_recognition
import argparse
import pickle
import cv2
import time
import imutils

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--encodings', required=True,
                    help='Path to saved face encodings')
parser.add_argument('-d', '--detection-method', type=str, default='cnn',
                    help="face detection model to use: 'hog' or 'cnn'")
args = vars(parser.parse_args())

# load the encoded faces and names
print('[INFO] loading encodings...')
with open(args['encodings'], 'rb') as file:
    data = pickle.loads(file.read())

print('[INFO] starting video stream...')
vs = VideoStream(0).start()
time.sleep(2)

while True:
    # get the frame
    frame = vs.read()
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750) # scale down for faster process
    r = frame.shape[1] / float(rgb.shape[1]) # get the scale ratio for later use in puting text
    
    # detech face and get x,y coordinate of the bounding box
    # then embed/encode it
    print('[INFO] recognizing faces...')
    boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # initialize the list of names for detected faces
    names = list()
    
    # loop over the encoded faces
    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = 'Unknown'
        
        # check whether we found a matched face
        if True in matches:
            # find the indexes of all matched faces then initialize a dict
            # to count the total number of times each face was matched
            matchedIdxs = [i for i, b in enumerate(matches) if b]
            counts = {}
            
            # loop over matched indexes and count
            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1
                
            # get the name with the most count
            name = max(counts, key=counts.get)
            
        # append the list
        names.append(name)
        
    # iterate over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # scale back with the saved ratio
        top = int(top*r)
        right = int(right*r)
        bottom = int(bottom*r)
        left = int(left*r)
        
        cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
        # in case of text is off the top screen
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        
    # show the output
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord('q'):
        break
    
# clean up
cv2.destroyAllWindows()
vs.stop()