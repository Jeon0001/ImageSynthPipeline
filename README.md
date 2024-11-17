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

## Instructions (Local VQA model Setup)
0. (optional) ```conda create -n synthesis-pipeline python=3.10.12```
1. ```pip install diffusers transformers accelerate scipy safetensors```
2. ```pip install salesforce-lavis```

## Instructions (Google Image Scraper Setup)
0. ```pip install bing-image-downloader```

### Limitations:
1. Limited 512x512 resolution


### Todo:
1. Combine with automasker (segment-anything?)
2. Figure out checkpoint for inpainting. "x4-upscaling-ema.ckpt"