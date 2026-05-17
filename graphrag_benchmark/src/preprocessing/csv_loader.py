"""
CSV Data Loader
---------------
Handles loading, validation, and initial processing of IMDb CSV dataset.
"""
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from .models import Movie
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVLoader:
    """Load and validate IMDb movie dataset from CSV."""
    
    REQUIRED_COLUMNS = [
        "Series_Title", "Released_Year", "Genre", "IMDB_Rating",
        "Overview", "Director", "Star1", "Star2", "Star3", "Star4"
    ]
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self.movies: List[Movie] = []
        self.stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "dropped_rows": 0,
            "missing_values": {}
        }
    
    def load(self, sample_size: Optional[int] = None) -> List[Movie]:
        """
        Load CSV and convert to Movie objects.
        
        Args:
            sample_size: If set, load only N random rows (for testing)
        
        Returns:
            List of validated Movie objects
        """
        logger.info(f"Loading CSV from: {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        self.stats["total_rows"] = len(self.df)
        
        logger.info(f"Loaded {self.stats['total_rows']} raw rows")
        
        # Sample if requested
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} rows")
        
        # Validate columns
        self._validate_columns()
        
        # Clean data
        self._clean_dataframe()
        
        # Convert to Movie objects
        self.movies = self._convert_to_movies()
        
        logger.info(f"Successfully loaded {len(self.movies)} movies")
        self._log_stats()
        
        return self.movies
    
    def _validate_columns(self) -> None:
        """Check required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            available = list(self.df.columns)
            raise ValueError(f"Missing columns: {missing}. Available: {available}")
        logger.info("All required columns present")
    
    def _clean_dataframe(self) -> None:
        """Clean and normalize the dataframe."""
        initial_rows = len(self.df)
        
        # Drop rows with missing critical data
        critical_cols = ["Series_Title", "Overview", "IMDB_Rating", "Genre"]
        self.df = self.df.dropna(subset=critical_cols)
        self.stats["dropped_rows"] = initial_rows - len(self.df)
        
        # Clean string columns
        string_cols = ["Series_Title", "Genre", "Director", "Overview",
                      "Star1", "Star2", "Star3", "Star4"]
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Convert numeric columns
        self.df["Released_Year"] = pd.to_numeric(self.df["Released_Year"], errors="coerce").fillna(0).astype(int)
        self.df["IMDB_Rating"] = pd.to_numeric(self.df["IMDB_Rating"], errors="coerce").fillna(0.0)
        self.df["Meta_score"] = pd.to_numeric(self.df["Meta_score"], errors="coerce")
        self.df["No_of_Votes"] = pd.to_numeric(self.df["No_of_Votes"], errors="coerce").fillna(0).astype(int)
        
        # Clean runtime
        if "Runtime" in self.df.columns:
            self.df["Runtime"] = self.df["Runtime"].astype(str).str.replace(" min", "", regex=False)
        
        # Track missing values
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.stats["missing_values"][col] = int(missing_count)
        
        self.stats["valid_rows"] = len(self.df)
        logger.info(f"Cleaned: {self.stats['dropped_rows']} rows dropped, {self.stats['valid_rows']} valid")
    
    def _convert_to_movies(self) -> List[Movie]:
        """Convert DataFrame rows to Movie objects."""
        movies = []
        
        for _, row in self.df.iterrows():
            try:
                movie = Movie(
                    poster_link=str(row.get("Poster_Link", "")),
                    series_title=str(row.get("Series_Title", "")),
                    released_year=int(row.get("Released_Year", 0)) if pd.notna(row.get("Released_Year")) else 0,
                    certificate=str(row.get("Certificate", "")),
                    runtime=str(row.get("Runtime", "")),
                    genre=str(row.get("Genre", "")),
                    imdb_rating=float(row.get("IMDB_Rating", 0.0)) if pd.notna(row.get("IMDB_Rating")) else 0.0,
                    overview=str(row.get("Overview", "")),
                    meta_score=float(row.get("Meta_score")) if pd.notna(row.get("Meta_score")) else None,
                    director=str(row.get("Director", "")),
                    star1=str(row.get("Star1", "")),
                    star2=str(row.get("Star2", "")),
                    star3=str(row.get("Star3", "")),
                    star4=str(row.get("Star4", "")),
                    no_of_votes=int(row.get("No_of_Votes", 0)) if pd.notna(row.get("No_of_Votes")) else 0,
                    gross=str(row.get("Gross")) if pd.notna(row.get("Gross")) else None
                )
                movies.append(movie)
            except Exception as e:
                logger.warning(f"Failed to convert row '{row.get('Series_Title', 'unknown')}': {e}")
        
        return movies
    
    def _log_stats(self) -> None:
        """Log dataset statistics."""
        if not self.movies:
            return
        
        years = [m.released_year for m in self.movies if m.released_year > 0]
        ratings = [m.imdb_rating for m in self.movies if m.imdb_rating > 0]
        
        logger.info(f"Dataset Stats:")
        logger.info(f"  Year range: {min(years)}-{max(years)}")
        logger.info(f"  Avg rating: {sum(ratings)/len(ratings):.2f}")
        logger.info(f"  Unique genres: {len(set(g for m in self.movies for g in m.genre_list))}")
        logger.info(f"  Unique directors: {len(set(m.director for m in self.movies if m.director))}")


class DataSplitter:
    """Split dataset into train/test/validation sets."""
    
    @staticmethod
    def split(movies: List[Movie], 
              train_ratio: float = 0.8,
              val_ratio: float = 0.1,
              random_seed: int = 42) -> Tuple[List[Movie], List[Movie], List[Movie]]:
        """
        Split movies into train/val/test sets.
        
        Args:
            movies: Full list of movies
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train, val, test) movie lists
        """
        import random
        random.seed(random_seed)
        
        shuffled = movies.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        
        logger.info(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test
