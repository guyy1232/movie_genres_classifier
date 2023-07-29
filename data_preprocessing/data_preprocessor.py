from collections import defaultdict
from dataclasses import dataclass
import ast
import pandas as pd


@dataclass
class DataPreprocessor:
    data_path: str

    def __post_init__(self):
        self.raw_data = pd.read_csv(self.data_path)
        self.data = self.raw_data.copy(deep=True)

    # Function to filter genres
    @staticmethod
    def filter_genre_list(genre_list, top_k_genres):
        return [genre for genre in genre_list if genre in top_k_genres]

    def filter_genres(self, k):
        # Convert the 'genres' column from string format to actual list format
        self.data['genres'] = self.data['genres'].apply(ast.literal_eval)

        # Initialize a dictionary to hold genre counts
        genre_counts = defaultdict(int)

        # Calculate genre counts
        for genres in self.data['genres']:
            for genre in genres:
                genre_counts[genre] += 1

        # Sort genres by count and get the top K genres
        top_k_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:k]

        # Filter the genres of each movie
        self.data['genres'] = self.data['genres'].apply(lambda g: self.filter_genre_list(g, top_k_genres))

        # Filter movies with empty labels
        self.data = self.data[self.data['genres'].apply(lambda x: len(x) > 0)]
        self.data = self.data[['plot_summary', 'genres']]

        return self.data
