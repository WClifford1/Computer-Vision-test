"""
Download Sample Image Dataset
Downloads images from various sources for testing computer vision models
"""

import requests
from pathlib import Path
import argparse


# Sample image URLs from different categories
SAMPLE_DATASETS = {
    'coco_samples': [
        # People
        'http://images.cocodataset.org/val2017/000000039769.jpg',  # Cats on couch
        'http://images.cocodataset.org/val2017/000000397133.jpg',  # People with tennis rackets
        'http://images.cocodataset.org/val2017/000000037777.jpg',  # Giraffe
        'http://images.cocodataset.org/val2017/000000252219.jpg',  # Skateboarder
        'http://images.cocodataset.org/val2017/000000087038.jpg',  # Person with umbrella
        'http://images.cocodataset.org/val2017/000000174482.jpg',  # Dogs
        'http://images.cocodataset.org/val2017/000000403385.jpg',  # Baseball players
        'http://images.cocodataset.org/val2017/000000006818.jpg',  # Skier
        'http://images.cocodataset.org/val2017/000000360137.jpg',  # Street scene
        'http://images.cocodataset.org/val2017/000000001000.jpg',  # Tennis player
        # Vehicles
        'http://images.cocodataset.org/val2017/000000018150.jpg',  # Cars
        'http://images.cocodataset.org/val2017/000000017627.jpg',  # Motorcycle
        'http://images.cocodataset.org/val2017/000000025560.jpg',  # Bicycle
        # Animals
        'http://images.cocodataset.org/val2017/000000025393.jpg',  # Zebras
        'http://images.cocodataset.org/val2017/000000042296.jpg',  # Elephants
        'http://images.cocodataset.org/val2017/000000058636.jpg',  # Sheep
        # Mixed scenes
        'http://images.cocodataset.org/val2017/000000286994.jpg',  # Food/dining
        'http://images.cocodataset.org/val2017/000000118113.jpg',  # Living room
        'http://images.cocodataset.org/val2017/000000156071.jpg',  # Kitchen
        'http://images.cocodataset.org/val2017/000000001503.jpg',  # Sports
    ],
    
    'picsum': [
        # Random high-quality photos from Lorem Picsum
        'https://picsum.photos/id/10/1920/1080',  # Forest
        'https://picsum.photos/id/20/1920/1080',  # Beach
        'https://picsum.photos/id/30/1920/1080',  # City
        'https://picsum.photos/id/40/1920/1080',  # Nature
        'https://picsum.photos/id/50/1920/1080',  # Urban
    ],
}


def download_image(url, output_path, timeout=10):
    """Download a single image from URL"""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_dataset(dataset_name='coco_samples', output_dir='images/test_dataset'):
    """
    Download a dataset of sample images
    
    Args:
        dataset_name: Name of the dataset to download
        output_dir: Directory to save images
    """
    if dataset_name not in SAMPLE_DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(SAMPLE_DATASETS.keys())}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    urls = SAMPLE_DATASETS[dataset_name]
    
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name} dataset ({len(urls)} images)")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        # Extract filename from URL or create one
        if '/' in url:
            filename = url.split('/')[-1]
            # Add .jpg extension if not present
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                filename = f"{dataset_name}_{i:03d}.jpg"
        else:
            filename = f"{dataset_name}_{i:03d}.jpg"
        
        output_file = output_path / filename
        
        print(f"[{i}/{len(urls)}] Downloading {filename}...", end=' ')
        
        if download_image(url, output_file):
            print(f"✓")
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  ✓ Successful: {successful}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    print(f"\nImages saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Print usage example
    print("Usage examples:")
    print(f"  python yolo_demo.py {output_path}/000000039769.jpg --show --conf 0.4")
    print(f"  python yolo_cyberpunk.py {output_path} --show")
    print(f"  python pose_demo.py {output_path}/000000001000.jpg --show")


def main():
    parser = argparse.ArgumentParser(description='Download sample image datasets for testing')
    parser.add_argument('--dataset', type=str, default='coco_samples',
                        choices=list(SAMPLE_DATASETS.keys()),
                        help='Dataset to download (default: coco_samples)')
    parser.add_argument('--output', type=str, default='images/test_dataset',
                        help='Output directory (default: images/test_dataset)')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        for name, urls in SAMPLE_DATASETS.items():
            print(f"  - {name}: {len(urls)} images")
        print()
        return
    
    download_dataset(args.dataset, args.output)


if __name__ == '__main__':
    main()
