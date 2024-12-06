import cv2
import os
import shutil

# Paths to input and output directories
# Change as needed
input_dir = 'D:\Data Downloads\Bing Image Scraped Results\Korean Clothes'
output_dir = 'D:\Data Downloads\Bing Image Scraped Results\Korean Clothes Face'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

#Obtain xml file path
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the Haar Cascade file
cascade_path = os.path.join(script_directory, 'haarcascade_frontalface_default.xml')

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # If faces are detected, copy the image to the output directory
        if len(faces) > 0:
            shutil.copy(img_path, os.path.join(output_dir, filename))
            print(f"Faces detected in {filename}. Image copied to output directory.")
        else:
            print(f"No faces detected in {filename}.")
