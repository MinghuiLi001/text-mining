from pprint import pprint

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import pickle
from tabulate import tabulate
from collections import Counter

def load_pickle_data(movie_names):
    """
    This function reads data from pickle files for next steps
    """
    print(movie_names)
    pickle_movies_data = {}
    for movie in movie_names:
        pickle_file_name = movie.replace(" ", "_") + ".pickle"
        with open(pickle_file_name, "rb") as input_file:
            movie_data = pickle.load(input_file)
            pickle_movies_data[movie] = movie_data
        input_file.close()
    return pickle_movies_data

# METHOD 1: Characterizing by Word Frequencies
def characterize_by_word_freq(movies, movies_data):
    """
    This function generates a dictionary of words from IMDB reviews
    for each movie and ranks the words from highest frequency to lowest frequency
    """

    # Call helper function to generate word list for each movie
    comment_word_list_for_movie = generate_comment_word_list(movies, movies_data)

    for movie in movies:
        comment_word_list = comment_word_list_for_movie[movie]
        # Calculate word frequency for each word
        word_freq = [comment_word_list.count(word) for word in comment_word_list]

        # Generate frequency dictionary
        word_freq_dict = dict(list(zip(comment_word_list, word_freq)))
        sorted_word_freq = {
            key: value
            for key, value in sorted(
                word_freq_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        print(f"{movie}: {sorted_word_freq}")

    print("\n")

    # Generate top 10 frequency word in reviews for each movie
    print("Generating top 10 word in reviews ...")
    for movie in movies:
        comment_word_list = comment_word_list_for_movie[movie]
        counter = Counter(comment_word_list)
        top_ten_words = counter.most_common(10)
        print(f"{movie}: {top_ten_words}")

    return


# Helper Function: Concatenate comment word list
def generate_comment_word_list(movies, movies_data):
    """
    This funciton consolidates all the reviews into a dictionary for each movie.
    Extract all the stopwords to make the dicitonary more meaningful for next steps
    """
    comment_text = {}
    for movie in movies:
        overall_review_text = ""
        current_movie_data = movies_data[movie]

        # Concatenate all reviews to a single string
        for review in current_movie_data:
            review_text = review["reviewText"]
            overall_review_text += review_text

        # Remove special characters from review text
        clean_overall_review_text = clean_text(overall_review_text)

        # Generate word list
        word_list = clean_overall_review_text.split()
        clean_word_list = [
            word for word in word_list if word not in stopwords.words("english")
        ]
        comment_text[movie] = clean_word_list

    return comment_text


# Helper Function: Remove special characters
def clean_text(text):
    """
    This function converts texts into lowercases and removes special character like comma, 
    period, parenthesis, question mark, and stars from the word list
    """
    cleaned = (
        text.replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("?", "")
        .replace(",", "")
        .replace("!", "")
        .replace("*", "")
        .replace(":", "")
        .replace("-", "")
        .replace("$", "")
        .lower()
    )
    return cleaned


# METHOD 2: Computing Summary Statistics
ADJ = "JJ"  # big
ADJ_COMPARE = "JJR"  # bigger
ADJ_SUPER = "JJS"  # biggest
# Resource:
# https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b


def compute_summary_stat(movies, movies_data):
    """
    This function generates a dictionary of adjectives from IMDB reviews
    for each movie and ranks the adjectives from highest frequency to lowest frequency
    """

    # Call helper function to generate concatenated review for each movie
    comment_text_for_movie = generate_comment_text(movies, movies_data)
    for movie in movies:
        comment_text = comment_text_for_movie[movie]
        tokenized = nltk.word_tokenize(comment_text)
        word_tag_list = nltk.pos_tag(tokenized) # Get all word properties from comment text [(word, JJ)]
        clean_word_tag_list = clean_tokenized(word_tag_list)
        print(clean_word_tag_list)

        adj_word_list = [
            word[0]
            for word in clean_word_tag_list
            if word[1] == ADJ or word[1] == ADJ_COMPARE or word[1] == ADJ_SUPER
        ]

        word_freq = [adj_word_list.count(word) for word in adj_word_list]
        word_freq_dict = dict(list(zip(adj_word_list, word_freq)))
        sorted_word_freq = {
            key: value
            for key, value in sorted(
                word_freq_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        print(f"{movie}: {sorted_word_freq}")


# Helper Function: Concatenate comment text
def generate_comment_text(movies, movies_data):
    """
    This funciton concatenate reviews into single string for each movie,
    then return a dictionary where key is the movie name and value is 
    the review text
    """
    comment_text = {}
    for movie in movies:
        overall_review_text = ""
        current_movie_data = movies_data[movie]
        for review in current_movie_data:
            review_text = review["reviewText"]
            overall_review_text += review_text

        # Remove special characters
        clean_overall_review_text = clean_text(overall_review_text)
        comment_text[movie] = clean_overall_review_text
    return comment_text


# Helper Function: Remove stop words from tokenized word list
def clean_tokenized(word_tag_list):
    """
    This function makes the first item into lowercase and eliminate stopwords to make the list more meaningful 
    for next steps

    >>> word_tag_list = [("great", "JJ"), ("i", "NN"), ("He", "JJ")]
    >>> clean_toeknized(word_tag_list)
    >>> [("great", "JJ")]
    """
    return [
        word
        for word in word_tag_list
        if word[0].lower() not in stopwords.words("english")
    ]


# METHOD 3: Doing Natural Language Processing
def nlp_sentiment_analysis(movies, movies_data):
    """
    This function creates a dictionary to store the normalized sentiment data for each movie 
    """
    sentiment_analysis = {}
    for movie in movies:
        reviews = movies_data[movie]
        sentiment_data = {}
        for review in reviews:
            review_text = review["reviewText"]
            review_anaylsis = SentimentIntensityAnalyzer().polarity_scores(review_text)
            for key in review_anaylsis.keys():
                if key in sentiment_data:
                    sentiment_data[key] += review_anaylsis[key]
                else:
                    sentiment_data[key] = review_anaylsis[key]
        normalize_sent_data = normalize_sentiment(sentiment_data, len(reviews))
        sentiment_analysis[movie] = normalize_sent_data

    visualize_sentiment(sentiment_analysis)
    return sentiment_analysis

# Helper Function: Normalize sentiment result based on review counts
def normalize_sentiment(sentiment_data, reviews_count):
    """
    This function normalizes sentiment result based on review counts
    """

    for key in sentiment_data.keys():
        # Normalized sentiment analysis by review count
        sentiment_data[key] = sentiment_data[key] / reviews_count
    return sentiment_data


# Helper Function: Visualize sentiment analysis result in table format
def visualize_sentiment(sentiment_analysis):
    """
    This function visualize the sentiment analysis result in a table format
    """
    table_list = []
    for key in sentiment_analysis.keys():
        current_list = [key]
        for stat in sentiment_analysis[key].keys():
            current_list.append(round(sentiment_analysis[key][stat], 3))
        table_list.append(current_list)

    print(
        tabulate(
            table_list,
            headers=["Movie", "Neg", "Neu", "Pos", "Compound"],
            tablefmt="orgtbl",
        )
    )


if __name__ == "__main__":
    movies = [
        "The Shawshank Redemption",
        "The Godfather",
        "Disaster Movie",
        "Saving Christmas",
    ]


    # print(stopwords.words('english'))
    print("Reading movie data from pickle files ...")
    movies_data = load_pickle_data(movies)
    print("\n")

    print("Characterizing by Word Frequencies ...")
    characterize_by_word_freq(movies, movies_data)
    print("\n")

    print("Computing Summary Statistics ...")
    compute_summary_stat(movies, movies_data)
    print("\n")

    print("Sentiment Analysis with NLP ...")
    sentiment_analysis = nlp_sentiment_analysis(movies, movies_data)
