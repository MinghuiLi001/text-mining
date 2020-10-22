"""
Assignment 2: Text Analysis and Text Mining

Part 1: Data Fetching from selected data source
IMDB Movie Reviews
"""

from imdbpie import Imdb
import pprint
import pickle


imdb = Imdb()

movie_names = [
    "The Shawshank Redemption",
    "The Godfather",
    "Disaster Movie",
    "Saving Christmas",
]

for movie in movie_names:
    """
    This step fetches data from Imdb according to movie names in the list
    and store the movie datas as a pickle file with a standardized file name
    """
    movie_title = imdb.search_for_title(movie)[0]
    movie_data = imdb.get_title_user_reviews(movie_title["imdb_id"])
    review_data = movie_data["reviews"]
    save_pickle_name = movie.replace(" ", "_") + ".pickle"

    with open(save_pickle_name, "wb") as file_stream:
        """
        This step opens a write stream
        then, write the moview review data to pickle file
        """
        pickle.dump(review_data, file_stream) 
    file_stream.close()

    print(f"{movie}'s review data has been saved to {save_pickle_name}")
