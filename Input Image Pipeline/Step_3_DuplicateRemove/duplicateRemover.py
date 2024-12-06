import os
import hashlib
from PIL import Image

def hash_image(image_path):
    """
    Generate a hash for the given image file.
    """
    with Image.open(image_path) as img:
        img = img.resize((8, 8), Image.ANTIALIAS).convert("L")  # Resize and convert to grayscale
        pixels = list(img.getdata())
        avg_pixel = sum(pixels) / len(pixels)
        bits = ''.join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
        hex_representation = f'{int(bits, 2):016x}'
        return hashlib.md5(hex_representation.encode('utf-8')).hexdigest()

def remove_duplicate_images(folder_path):
    """
    Remove duplicate images in the specified folder.
    """
    hashes = {}
    duplicate_count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_hash = hash_image(file_path)
                if file_hash in hashes:
                    # If hash already exists, remove the duplicate
                    os.remove(file_path)
                    duplicate_count += 1
                    print(f"Removed duplicate: {file_path}")
                else:
                    hashes[file_hash] = file_path
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Duplicate removal complete. {duplicate_count} duplicates removed.")

# Example usage:
folder_path = "D:\Data Downloads\Bing Image Scraped Results\Korean Clothes"  # Replace with your folder path
remove_duplicate_images(folder_path)
