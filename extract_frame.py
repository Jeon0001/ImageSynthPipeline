import argparse
import sys
import os
from utils import * 
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description='Extract frames from YouTube videos based on visual similarity to a text prompt.')
    parser.add_argument('-s', '--search-query', type=str, required=True, help='Name of the food to search for')
    parser.add_argument('-mr', '--max-results', type=int, default=10, help='Maximum number of videos to process')
    parser.add_argument('-mf', '--max-frames', type=int, default=3, help='Maximum number of frames to extract per video')
    parser.add_argument('-p', '--text-prompt', type=str, default='A person with the food', help='Text prompt for CLIP model')
    parser.add_argument('-o', '--output-dir', type=str, default='saved_images', help='Output directory for saved frames')
    parser.add_argument('-ya', '--youtube-api-key', type=str, default='', help='YouTube API key')    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()
    load_dotenv()

    youtube_api_key = os.getenv('OPENAI_API_KEY') if not args.youtube_api_key else args.youtube_api_key

    extractor = StreamingVideoFrameExtractor()
    extractor.verbose = args.verbose

    search_results = search_youtube_videos(args.search_query, youtube_api_key, args.max_results)
    if not search_results:
        print("No matching videos found")
        return

    for i, video in enumerate(search_results):

        print(f"Processing video {i+1}/{len(search_results)}")
        print(f"Title: {video['title']} | url: {video['url']}")
        
        try:
            results = extractor.extract_top_frames_from_stream(
                video['url'], 
                args.text_prompt, 
                args.max_frames
            )
            extractor.save_frames(results, args.output_dir, i)
        except Exception as e:
            print(f"Error processing video {i+1}: {str(e)}")
            continue

if __name__ == "__main__":
    main()