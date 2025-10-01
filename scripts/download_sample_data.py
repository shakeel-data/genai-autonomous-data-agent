"""
Sample Data Download Script
Downloads additional sample datasets for testing
"""

import requests
import pandas as pd
from pathlib import Path
import zipfile
import io

def download_public_datasets():
    """Download public datasets for testing"""
    
    output_dir = Path("data/sample_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "iris.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "tips.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
        "flights.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
        "titanic.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    }
    
    print("📥 Downloading public datasets...")
    
    for filename, url in datasets.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filepath = output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    print("🎉 Public datasets download completed!")

if __name__ == "__main__":
    download_public_datasets()
    print("✅ Sample data download script completed!")