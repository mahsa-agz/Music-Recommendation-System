import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Function for Random Sampling
def random_sampling(user_id, ratings_df, songs_features_df):
    rated_songs = set(ratings_df[ratings_df['user_id'] == user_id]['song_id'])
    all_songs = set(songs_features_df['song_id'])
    unrated_songs = list(all_songs - rated_songs)
    if unrated_songs:
        return np.random.choice(unrated_songs)
    else:
        return None


# Function for Most Popular Song
def most_popular_song(user_id, ratings_df, songs_features_df):
    rated_songs = set(ratings_df[ratings_df['user_id'] == user_id]['song_id'])
    all_songs = set(songs_features_df['song_id'])

    # Counting the number of ratings each song has received
    song_popularity = Counter(ratings_df['song_id'])

    # Sorting songs by popularity while excluding already rated songs
    most_popular_songs = [song for song, _ in song_popularity.most_common() if
                          song in all_songs and song not in rated_songs]

    if most_popular_songs:
        return most_popular_songs[0]
    else:
        return None

# Function for Most Similar Song
def most_similar_song(user_id, ratings_df, songs_features_df):
    # Extracting rated and unrated songs for the user
    rated_songs = set(ratings_df[ratings_df['user_id'] == user_id]['song_id'])
    unrated_songs = set(songs_features_df['song_id']) - rated_songs

    # Create a dictionary to map song_id to its features
    song_id_to_features = songs_features_df.set_index('song_id')[['album_id', 'artist_id', 'genre_id', 'parent_genre_id', 'gp_genre_id']].astype(str).to_dict('index')

    # Function to convert feature dictionary to a set of strings for comparison
    def features_to_set(features_dict):
        return set(features_dict.values())

    # Find the most similar song
    most_similar_song_id = None
    highest_similarity = -1
    for rated_song_id in rated_songs:
        rated_song_features = features_to_set(song_id_to_features[rated_song_id])
        for unrated_song_id in unrated_songs:
            unrated_song_features = features_to_set(song_id_to_features[unrated_song_id])
            # Calculate similarity as the number of common features
            similarity = len(rated_song_features.intersection(unrated_song_features))
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_song_id = unrated_song_id

    return most_similar_song_id


# Function for Interaction Pattern Similarity-based Negative User Sampling
def interaction_pattern_negative_user(user_id, ratings_df):
    # Create a user-item matrix
    user_item_matrix = pd.pivot_table(ratings_df, index='user_id', columns='song_id', values='rating', fill_value=0)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get least similar users
    least_similar_users = similarity_df[user_id].nsmallest(n=100).index  # n=2 to exclude the user itself
    return np.random.choice(least_similar_users[:-1])


import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

def calculate_similarity(user1_ratings, user2_ratings):
    # Calculate cosine similarity between two sets of ratings
    common_songs = user1_ratings.keys() & user2_ratings.keys()
    if not common_songs:
        return 1  # Completely dissimilar if no common songs

    # Vectors for common songs
    user1_vector = [user1_ratings[song] for song in common_songs]
    user2_vector = [user2_ratings[song] for song in common_songs]

    return cosine(user1_vector, user2_vector)

def find_negative_users(user_id, ratings_df, num_neg_users=1):
    # Get ratings of the target user
    target_user_ratings = dict(ratings_df[ratings_df['user_id'] == user_id][['song_id', 'rating']].values)

    # Calculate similarity with all other users
    user_similarity = {}
    for other_user in ratings_df['user_id'].unique():
        if other_user != user_id:
            other_user_ratings = dict(ratings_df[ratings_df['user_id'] == other_user][['song_id', 'rating']].values)
            similarity = calculate_similarity(target_user_ratings, other_user_ratings)
            user_similarity[other_user] = similarity

    # Sort users by similarity (lower is better)
    sorted_users = sorted(user_similarity, key=user_similarity.get, reverse=True)

    # Select negative users from those with the highest dissimilarity
    negative_users = np.random.choice(sorted_users[:num_neg_users], size=num_neg_users, replace=False)
    return negative_users

# import pandas as pd
# import numpy as np
from tqdm import tqdm
def find_random_negative_user(user_id, all_user_ids):
    # Choose a random user different from the current user
    return np.random.choice([u for u in all_user_ids if u != user_id])

if __name__ == '__main__':
    # Load your datasets
    print('Start loading data')
    ratings_df = pd.read_csv('data/sdata/sdata/train_rating_2d__aa.csv') #, sep='\t', names=['user_id', 'song_id', 'rating'])
    all_user_ids = ratings_df['user_id'].unique()

    # Add the negative samples to your dataframe
    print('Data loaded, start sampling')
    for index, row in tqdm(ratings_df.iterrows()):
        ratings_df.loc[index, 'neg_user_id'] = find_random_negative_user(row['user_id'], all_user_ids)
    ratings_df['neg_user_id'] = ratings_df['user_id'].apply(lambda x: find_random_negative_user(x, all_user_ids))
    ratings_df.to_csv('data/train_rating_with_neg_1.csv', index=False)

    print('Done!')
