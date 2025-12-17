import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Tuple
from config import config



class MovieLensDataLoader:
    """Load and process MovieLens 100K dataset"""
    
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_data(self) -> Path:
        """Download MovieLens 100K dataset"""
        zip_path = self.raw_dir / "ml-100k.zip"
        extract_path = self.raw_dir / "ml-100k"
        
        if extract_path.exists():
            logger.info("Dataset already exists. Skipping download.")
            return extract_path
        
        logger.info(f"Downloading MovieLens 100K from {config.MOVIELENS_URL}")
        
        response = requests.get(config.MOVIELENS_URL, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        logger.info("Dataset downloaded and extracted successfully!")
        return extract_path
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings data"""
        extract_path = self.download_data()
        ratings_path = extract_path / "u.data"
        
        logger.info("Loading ratings data...")
        ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Convert timestamp to datetime
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        logger.info(f"Loaded {len(ratings)} ratings")
        return ratings
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie information"""
        extract_path = self.download_data()
        movies_path = extract_path / "u.item"
        
        logger.info("Loading movie data...")
        
        # Movie genres
        genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        columns = ['item_id', 'title', 'release_date', 'video_release_date',
                  'imdb_url'] + genres
        
        movies = pd.read_csv(
            movies_path,
            sep='|',
            names=columns,
            encoding='latin-1',
            engine='python'
        )
        
        # Parse year from title
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0].astype(float)
        
        logger.info(f"Loaded {len(movies)} movies")
        return movies
    
    def load_users(self) -> pd.DataFrame:
        """Load user information"""
        extract_path = self.download_data()
        users_path = extract_path / "u.user"
        
        logger.info("Loading user data...")
        users = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        logger.info(f"Loaded {len(users)} users")
        return users
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets"""
        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()
        
        return ratings, movies, users


if __name__ == "__main__":
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_all_data()
    
    print("\nRatings shape:", ratings.shape)
    print(ratings.head())
    
    print("\nMovies shape:", movies.shape)
    print(movies.head())
    
    print("\nUsers shape:", users.shape)
    print(users.head())
