# set the matplotlib backend, so figures can be saved in the background
import matplotlib
matplotlib.use('Agg') # Agg is used for writing files

from livenessnet import LivenessNet # our model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True,
                    help='Path to input Dataset')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to output trained model')
parser.add_argument('-l', '--le', type=str, required=True,
                    help='Path to output Label Encoder')
parser.add_argument('-p', '--plot', type=str, default='plot.png',
                    help='Path to output loss/accuracy plot')
args = vars(parser.parse_args())

# get the list of images in out dataset directory
# then, initialize the list of data and class of images
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset'])) # get all image path from dataset path
data = list()
labels = list()

# iterate over all image paths
for imagePath in imagePaths:
    # extract the class label from file name, load and resize to 32x32
    # example for our path '/face liveness detection/dataset\\fake\\0.png'
    # so, if we split with os.path.sep we will get
    # ['/face liveness detection/dataset', 'fake', '0.png']
    # that means if we get the second last element we will get each image's label!
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32,32)) # we will use 32x32 input shape for our model (not too big)
    
    # append the image and label to the list
    data.append(image)
    labels.append(label)
    
# convert to ndarray and do feature scaling
# tf works best with ndarray and it's super fast and efficient!
data = np.array(data, dtype='float') / 255.0

# encode the labels from (fake, real) to  (0,1)
# and do one-hot encoding
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, 2) # one-hot encoding

# train/test split
# we go for traditional 80/20 split
# Theoretically, we have small dataset we need test set to be a bit bigger
# 75/25 or 70/30 split would be ideal, but from the trial and error
# 80/20 gives a better result, so we go for it
# Another thing to consider, since my dataset has only about 14 images of faces from card/solid image
# so 80/20 has a higher chance that those images will be in training set rather than test set (and none on training set)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

# construct the training image generator for data augmentation
# this method from TF will do augmentation at runtime. So, it's quite handy
aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20, 
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# build a model
# define hyperparameters
INIT_LR = 1e-4 # initial learning rate
BATCH_SIZE = 4
EPOCHS = 50

# we don't need early stopping here because we have a small dataset, there is no need to do so
# initialize the optimizer and model
print('[INFO] compiling model...')
optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# train the model
print(f'[INFO] training model for {EPOCHS} epochs...')
history = model.fit(x=aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    epochs=EPOCHS)

# evaluate the model
print('[INFO] evaluating network...')
predictions = model.predict(x=X_test, batch_size=BATCH_SIZE)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save model to disk
print(f"[INFO serializing network to '{args['model']}']")
model.save(args['model'], save_format='h5')

# save the label encoder to disk
with open(args['le'], 'wb') as file:
    file.write(pickle.dumps(le))
    
# plot training loss and accuract and save
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, EPOCHS), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, EPOCHS), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCHS), history.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, EPOCHS), history.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])