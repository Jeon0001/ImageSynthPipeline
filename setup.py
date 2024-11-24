from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="extract_frame",
    version="1.0",
    author="MLLM",
    description="Script to extract frames from YouTube videos based on visual similarity to a text prompt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jeon0001/ImageSynthPipeline",
    
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    scripts=["extract_frame.py"],
    entry_points={
        "console_scripts": [
            "extract_frame=extract_frame:main",
        ],
    },

)