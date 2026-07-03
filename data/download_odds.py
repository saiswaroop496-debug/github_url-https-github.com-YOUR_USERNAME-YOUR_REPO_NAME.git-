import os
import requests
from pathlib import Path

def download_odds():
    urls = [
        "https://www.football-data.co.uk/new/WC.csv",   # World Cup
        "https://www.football-data.co.uk/new/EC.csv",   # Euro Cup
    ]
    
    out_dir = Path("data/football_data_odds")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        filename = url.split('/')[-1]
        filepath = out_dir / filename
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filepath.write_bytes(response.content)
                print(f"  -> Saved {len(response.content)} bytes to {filepath}")
            else:
                print(f"  -> Failed (Status {response.status_code})")
        except Exception as e:
            print(f"  -> Error: {e}")

if __name__ == "__main__":
    download_odds()
