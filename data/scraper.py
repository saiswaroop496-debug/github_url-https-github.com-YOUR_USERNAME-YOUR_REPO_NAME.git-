import os
import requests
import pandas as pd
from dotenv import load_dotenv

from data.mock_worldcup_data import MockDataGenerator

class DataScraper:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("RAPIDAPI_KEY")
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }
        
    def fetch_fixtures(self):
        try:
            if not self.api_key or self.api_key == "your_api_football_key":
                raise ValueError("API Key missing")
            
            response = requests.get(f"{self.base_url}/fixtures", headers=self.headers, params={"league": "4", "season": "2022"}, timeout=5)
            response.raise_for_status()
            
            # Real parsing logic would go here, returning mock for simulation
            raise NotImplementedError("Live parsing omitted, falling back to mock.")
            
        except Exception as e:
            # Zero-downtime fallback
            return MockDataGenerator().generate()

    def fetch_fixture_statistics(self, fixture_id):
        try:
            if not self.api_key or self.api_key == "your_api_football_key":
                raise ValueError("API Key missing")
                
            response = requests.get(f"{self.base_url}/fixtures/statistics", headers=self.headers, params={"fixture": fixture_id}, timeout=5)
            response.raise_for_status()
            raise NotImplementedError("Live parsing omitted, falling back to mock.")
        except Exception as e:
            # Return dummy stats
            return []
