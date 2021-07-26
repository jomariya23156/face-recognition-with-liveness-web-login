# (full documentation is coming soon!)
# Face Recognition with Liveness Detection Login on Flask Web application
## Project Overview
We have implemented Flask web application login page including face verification (1-to-1 to verify whether the person who is logging in is really that person) for security purpose with liveness detection mechanism (to check whether the person detected on the camera is a REAL person or FAKE (eg. image, video, etc. of that person)) for Anti-Spoofting (Others pretending to be the person) built with Convolutional Neural Network.

## Result
* The face recognition works well detecting face and accurately recognizing the face.
* The liveness detection works well with classifying fake images and videos from smartphone spoofing.
* The liveness detection is also trained to be able to classify solid-printed images (image on papers and cards). But it's trained only with around 10 images, so it doesn't work well everytime (read "Training model with your own dataset" section, in case you want to train it yourself with bigger dataset)

## Packages and Tools
- OpenCV
- TensorFlow 2
- Scikit-learn
- Face_recognition
- Flask
- SQLite
- SQLAlchemy (for Flask)

## Project structure
(no need to care about "assets" folder, it's just images for GitHub)  
<img src = "./assets/project_structure.png" width=446 height=702>  

## Files explanation
In this section, we will show how to use/execute each file in this repo. The full workflow of the repo, from the data collection till running the web application, will be provided step-by-step in the next section.  
*Note: All files have code explanation in the source code themselves*
* In **main** folder (main repo)
  * `app.py`: This is the main Flask app. To use it, just normally run this file from your terminal/preferred IDE and the running port for your web app will be shown. **If you want to deploy this app, don't forget to change app.secret_key variable**
  * `database.sqlite`: This is the minimum example of database to store user's data and it is used to retrieve and verify while user is logging in.  
* In face_recognition_and_liveness/**face_recognition** folder
  * `encode_faces.py`: Detect the face from images, encode, and save encoded version and name (label) to pickle files. **The name/label coms from the name of folder of those input images.**  
    Command line argument:
    * `--dataset (or -i)` Path to input directory of images
    * `--encoding (or -e)` Path/Directory to save encoded images pickle files
    * `--detection-method (or -d)` Face detection model to use: 'hog' or 'cnn' (default is 'cnn')  
    **Example**: `python encode_faces.py -i dataset -e encoded_faces.pickle -d cnn`  
  * `recognize_faces.py`: Real-time face recognition on webcam with bounding boxes and names showing on top.
    Command line argument:
    * `--encoding (or -e)` Path to saved face encodings
    * `--detection-method (or -d)` Face detection model to use: 'hog' or 'cnn' (default is 'cnn')  
    **Example**: `python recognize_faces.py -e encoded_faces.pickle -d cnn`
  * `dataset` folder: Example folder and images to store dataset used to do face encoding. There are subfolders in this folder and each subfolder must be named after the owner of image in that subfolder because this subfolder name will be used as a label in face encoding process.
* In face_recognition_and_liveness/**face_liveness_detection** folder
  * `collect_dataset.py`: Collect face in each frame from a *video* dataset (real/fake) using face detector model (resnet-10 SSD in this case) and save to a directory (we provided a video example in videos folder, so you can collect the correct video dataset to train the model)
    Command line argument:
    * `--input (or -i)` Path to input input video
    * `--output (or -o)` Path/Directory to output directory of cropped face images
    * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
    * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)
    * `--skip (or -s)` Number of frames to skip before applying face detection and crop (dafault is 16). The main idea for this is the consequence frames usually give the same face to the dataset, so it can easily causes overfitting and is not a useful data for training.  
    **Example**: example for fake video dataset -> `python collect_dataset.py -i videos/fake_1.mp4 -o dataset/fake -d face_detector -c 0.5 -s 15` | example for real video dataset -> `python collect_dataset.py -i videos/real_1.mp4 -o dataset/real -d face_detector -c 0.5 -s 15`
  * `face_from_image.py`: Collect face in each frame from a *image* dataset (real/fake) using face detector model (resnet-10 SSD in this case) and save to a directory (we provided a video example in videos folder, so you can collect the correct video dataset to train the model)
    Command line argument:
    * `--input (or -i)` Path to input input image (A single image | Since we mainly collect dataset from videos, we use this code only to collect face from those solid-printed picture (picture from paper/card) and we didn't have many of them. So, we make the code just for collect face from 1 image. Feel free to adjust the code if you want to make it able to collect faces from all image in a folder/directory)
    * `--output (or -o)` Path/Directory to output directory of cropped face images
    * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
    * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)
    **Example**: example for fake image dataset -> `python face_from_image.py -i images/fakes/2.jpg -o dataset/fake -d face_detector -c 0.5` | example for real image dataset -> `python face_from_image.py -i images/reals/1.jpg -o dataset/real -d face_detector -c 0.5`
    

## Basic usage
1. Download/Clone this repo
2. Run app.py
3. That's it!  
  
**Note: Doing only these steps will allow only Liveness detection to work but not Recognition and Full login mechanism (full document for training your own model and using your own face to make everything work perfectly is coming soon! Stay Tuned!)**



## Problems we've found
All login mechanism work properly, but sometimes OpenCV camera doesn't show up when calling a function from app.py. But restarting app.py once or twice always solves the problem.
