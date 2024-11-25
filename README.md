# ImageSynthPipeline

Image Synthesis Pipeline for Creation of dataset with cultural notions. (Step 4 for automatic VQA using BLIP model.)
**Refer Instructions, then Pipeline_Image_Synthesis.ipynb for step by step code.**

## Commit 21.

Added API compatibility for inpainting method
Follow instructions in notebook.

#### Python Ver.

Python 3.10.12

#### Pytorch Ver.

2.4.1+cu121

#### Inpainting Model Used

stabilityai/stable-diffusion-2-inpainting: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

# Instructions (Local VQA model Setup)

0. (optional) `conda create -n synthesis-pipeline python=3.10.12`
1. `pip install diffusers transformers accelerate scipy safetensors`
2. `pip install salesforce-lavis`

# Instructions (Input Image Generator Pipeline)

0. `pip install git+https://github.com/ostrolucky/Bulk-Bing-Image-downloader`
1. Check "/ImageSynthPipeline/Input Image Pipeline/Step1_Image Scrapers" for instructions and scrape web for images that contain certain query
2. `pip install opencv-python`
3. Check "/ImageSynthPipeline/Input Image Pipeline/Step2_Image Face Detector" to filter images that contain people
4. For extracting frames from YouTube, `pip install yt-dlp ffmpeg-python python-dotenv google-api-python-client`. If `ffmpeg` is still not found, try `sudo apt install ffmpeg`
   - Make sure to keep your own YouTube Data API Key in `.env` file, `YOUTUBE_API_KEY=<your_key>` or directly pass the key as an argument to the command (not recommended).

## Folder Structure

```
images
      |---input_images
            |---food
                  |---country
            |---clothes
                  |---country
      |---masks_images
      |---output_images
Input Image Pipeline
      |---extract_frame.py
      |---mask.py
      |---pipeline.ipynb   # uses mask.py and extract_frame.py
API_Access
UTKFace_SortedByRace
```

### Limitations:

1. Limited 512x512 resolution

### Todo:

1. Combine with automasker (segment-anything?)
2. Figure out checkpoint for inpainting. "x4-upscaling-ema.ckpt"
