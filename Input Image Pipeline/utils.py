import os
import cv2
import subprocess
import ffmpeg
import json
import shutil
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Tuple


class StreamingVideoFrameExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_video_stream(self, youtube_url):
        cmd = [
            'yt-dlp',
            '-f', 'best[height<=720]',
            '--get-url',
            '--print',
            '{"width": %(width)s, "height": %(height)s, "fps": %(fps)s}',
            youtube_url
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise Exception(f"Error getting video stream: {stderr.decode()}")

        lines = stdout.decode().strip().split('\n')
        metadata = json.loads(lines[0])
        stream_url = lines[1]

        return stream_url, metadata

    def create_ffmpeg_pipe(self, stream_url):
        process = (
            ffmpeg
            .input(stream_url)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        return process

    def update_top_frames(self, top_frames: Dict[int, Tuple[np.ndarray, float]],
                         frame: np.ndarray, score: float, frame_num: int,
                         max_frames: int = 3) -> Dict[int, Tuple[np.ndarray, float]]:
        """Update dictionary of top frames, maintaining only the highest scoring frames"""
        top_frames[frame_num] = (frame.copy(), score)
        sorted_frames = dict(sorted(top_frames.items(),
                                  key=lambda x: x[1][1],
                                  reverse=True)[:max_frames])
        return sorted_frames

    def extract_top_frames_from_stream(self, youtube_url, text_prompt, max_frames=3, sample_rate=30):
        try:
            stream_url, metadata = self.get_video_stream(youtube_url)
            width = metadata['width']
            height = metadata['height']
            
            if self.verbose:
                print(f"Resolution: {width}x{height}")

            process = self.create_ffmpeg_pipe(stream_url)

            text_inputs = self.processor(
                text=text_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)

            top_frames = {}
            frame_count = 0
            frame_size = width * height * 3

            try:
                while True:
                    frame_data = process.stdout.read(frame_size * sample_rate)
                    if not frame_data:
                        break

                    frame_bytes = frame_data[-frame_size:]
                    frame_count += sample_rate

                    frame = np.frombuffer(frame_bytes, np.uint8)
                    frame = frame.reshape([height, width, 3])

                    image = Image.fromarray(frame)
                    image_inputs = self.processor(
                        images=image,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    image_features = self.model.get_image_features(**image_inputs)

                    similarity = torch.nn.functional.cosine_similarity(
                        text_features, image_features
                    ).item()

                    top_frames = self.update_top_frames(top_frames, frame, similarity, frame_count, max_frames)

                    if self.verbose and frame_count % (sample_rate * 10) == 0:
                        print(f"Processed {frame_count} frames...")

            finally:
                process.stdout.close()
                process.stderr.close()
                process.wait()

            results = [(frame, score, frame_num)
                      for frame_num, (frame, score) in top_frames.items()]
            return sorted(results, key=lambda x: x[1], reverse=True)

        except Exception as e:
            raise Exception(f"Error processing video stream: {str(e)}")

    def save_frames(self, frames, base_output_path, video_index):
        """Save multiple frames to files"""
        saved_paths = []
        if self.verbose:
            print(f"Frames extracted successfully!")
        for i, (frame, score, frame_number) in enumerate(frames):
            output_path = f"{base_output_path}/video-{video_index}_top-{i}.jpg"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)
            saved_paths.append(output_path)
            if self.verbose:
                print(f"Frame number: {frame_number} | Confidence score: {score:.4f}")
        return saved_paths

def search_youtube_videos(query, api_key, max_results=5):
    """Search for YouTube videos matching a query and return their URLs."""
    youtube = build('youtube', 'v3', developerKey=api_key)

    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=max_results,
        type='video',
        order='relevance'
    ).execute()

    videos = []
    for item in search_response.get('items', []):
        if item['id']['kind'] == 'youtube#video':
            video_id = item['id']['videoId']
            video_info = {
                'title': item['snippet']['title'],
                'url': f'https://www.youtube.com/watch?v={video_id}',
            }
            videos.append(video_info)

    return videos

def filter_faces(input_dir, output_dir):
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

            # If no face is detected, remove the image
            if len(faces) == 0:
                os.remove(img_path)
