import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import PIL.Image as Image
import os
import sys
from torchvision.transforms import GaussianBlur
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

class FaceDetectorYunet():
    def __init__(self,
                  model_path='../models/face_detection_yunet_2023mar.onnx',
                  img_size=(300, 300),
                  threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.fd = cv2.FaceDetectorYN_create(str(model_path),"", img_size, score_threshold=threshold)

    def draw_faces(self,
                   image,
                   faces,
                   show_confidence=False):
        for face in faces:
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, (face['x1'], face['y1']), (face['x2'], face['y2']), color, thickness, cv2.LINE_AA)

            if show_confidence:
                confidence = face['confidence']
                confidence = "{:.2f}".format(confidence)
                position = (face['x1'], face['y1'] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
        return image

    def scale_coords(self, image, prediction):
        ih, iw = image.shape[:2]
        rw, rh = self.img_size
        a = np.array([
                (prediction['x1'], prediction['y1']),
                (prediction['x1'] + prediction['x2'], prediction['y1'] + prediction['y2'])
                    ])
        b = np.array([iw/rw, ih/rh])
        c = a * b
        prediction['img_width'] = iw
        prediction['img_height'] = ih
        prediction['x1'] = int(c[0,0].round())
        prediction['x2'] = int(c[1,0].round())
        prediction['y1'] = int(c[0,1].round())
        prediction['y2'] = int(c[1,1].round())
        prediction['face_width'] = (c[1,0] - c[0,0])
        prediction['face_height'] = (c[1,1] - c[0,1])
        # prediction['face_width'] = prediction['x2'] - prediction['x1']
        # prediction['face_height'] = prediction['y2'] - prediction['y1']
        prediction['area'] = prediction['face_width'] * prediction['face_height']
        prediction['pct_of_frame'] = prediction['area']/(prediction['img_width'] * prediction['img_height'])
        return prediction

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(str(image))
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, self.img_size)
        self.fd.setInputSize(self.img_size)
        _, faces = self.fd.detect(img)
        if faces is None:
            return None
        else:
            predictions = self.parse_predictions(image, faces)
            return predictions

    def parse_predictions(self,
                          image,
                          faces):
        data = []
        for num, face in enumerate(list(faces)):
            x1, y1, x2, y2 = list(map(int, face[:4]))
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            positions = ['left_eye', 'right_eye', 'nose', 'right_mouth', 'left_mouth']
            landmarks = {positions[num]: x.tolist() for num, x in enumerate(landmarks)}
            confidence = face[-1]
            datum = {'x1': x1,
                     'y1': y1,
                     'x2': x2,
                     'y2': y2,
                     'face_num': num,
                     'landmarks': landmarks,
                     'confidence': confidence,
                     'model': 'yunet'}
            d = self.scale_coords(image, datum)
            data.append(d)
        return data
    
    
def show_mask(mask, ax, random_color=False, display=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color to inpaint (white)
        color = np.array([255, 255, 255])

    # color to keep intact (black)
    background_color = np.array([0, 0, 0])
    h, w = mask.shape[-2:]

    # Reshape the mask to have the same number of color channels
    mask = mask.reshape(h, w, 1)

    # Apply color to the mask where mask is True, and color2 where mask is False
    mask_image = mask * color.reshape(1, 1, -1) + ~mask * background_color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def detect_faces(predictor, fd, input_dir, output_dir, save_boxes=False):
    # Create necessary directories if required
    if save_boxes: os.makedirs('images/box_images', exist_ok=True)

    # Loop through all images in the directory
    for image_path in tqdm(os.listdir(input_dir)):
        img = cv2.imread(os.path.join(input_dir, image_path))
        faces = fd.detect(img)

        # If no faces are detected, skip to the next image
        if not faces:
            print(f"\nNo face detected in {image_path}.")
            continue

        # Prepare face center coordinates and labels
        face_centers = [((face['x1'] + face['x2']) / 2, (face['y1'] + face['y2']) / 2) for face in faces]
        input_point = np.array(face_centers)
        input_label = np.ones(len(input_point))

        # Optionally save boxes around faces
        if save_boxes:
            box_file_name = os.path.splitext(os.path.basename(image_path))[0] + "_box.png"
            img_copy = img.copy()
            fd.draw_faces(img_copy, faces, show_confidence=True)
            cv2.imwrite(f'{output_dir}/box_images/{box_file_name}', img_copy)


        # Process image and make predictions
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Select the best mask
        mask_index = np.argmax(scores)
        mask_input = logits[mask_index, :, :]

        # Using the best mask, creates a more accurate mask
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

        # Create a mask image
        mask_file_name = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        plt.figure(figsize=(10, 10))
        show_mask(masks, plt.gca())
        plt.axis('off')

        # Save the mask with blur applied
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        blur = GaussianBlur(11, 20)
        blurred_mask = blur(Image.open(buf))
        blurred_mask.save(os.path.join(output_dir, mask_file_name))
        

def main():
    parser = argparse.ArgumentParser(description='Extract frames from YouTube videos based on visual similarity to a text prompt.')
    parser.add_argument('-i', '--input-dir', type=str, default='images/input_images', help='Directory containing input images')
    parser.add_argument('-o', '--output-dir', type=str, default='images/masks_images', help='Output directory for saved frames')
    args = parser.parse_args()
    
    IN_COLAB = 'google.colab' in sys.modules
    print("CUDA is available:", torch.cuda.is_available())
    
    ### download the models
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    face_detection_model = "face_detection_yunet_2023mar.onnx"

        
    ### load the SAM model and predictor
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=f'../models/{sam_checkpoint}')
    sam.to(device=device)

    predictor = SamPredictor(sam)

    ### detect faces
    fd = FaceDetectorYunet()
    detect_faces(predictor, fd, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()