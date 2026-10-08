import random
import time
import hashlib

from torch.utils.data import Dataset
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict

def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict

def sample_negative_user(user_set, interaction_user_set):
    users = set(interaction_user_set)
    candidate_users = set(user_set) - set(users)
    return random.sample(list(candidate_users), 1)[0]

def _cache_key(df):
    """Hash of the interaction pairs, used as the cache key."""
    h = hashlib.md5()
    h.update(str(len(df)).encode())
    h.update(pd.util.hash_pandas_object(df['user_id'], index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(df['song_id'], index=False).values.tobytes())
    return h.hexdigest()[:12]


def _build_interaction_dict(train_csv, key_col, val_col, name):
    os.makedirs("pkl", exist_ok=True)
    path = os.path.join("pkl", "{}_{}.pkl".format(name, _cache_key(train_csv)))
    if os.path.exists(path):
        print("Load {} from cache {}".format(name, path))
        with open(path, 'rb') as f:
            return pickle.load(f)[name]
    d = train_csv.groupby(key_col)[val_col].apply(list).to_dict()
    with open(path, 'wb') as f:
        pickle.dump({name: d}, f)
    return d


def build_user_item_interaction_dict(train_csv):
    return _build_interaction_dict(train_csv, 'user_id', 'song_id',
                                   'user_item_interaction_dict')


# Create a new item-user interaction dictionary.
def build_item_user_interaction_dict(train_csv):
    return _build_interaction_dict(train_csv, 'song_id', 'user_id',
                                   'item_user_interaction_dict')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, genres, category_num, user_num,  positive_number, negative_number):
        self.train_csv = train_csv
        # Read other content
        self.genres_dict = genres
        # genres[song_id] = [artist_id, album_id, genre_id, parent_genre_id, gp_genre_id]
        artist_list = [v[0] for k, v in genres.items()]
        album_list = [v[1] for k, v in genres.items()]

        self.artist_serialize_dict = serialize_item(artist_list)
        self.album_serialize_dict = serialize_item(album_list)
        self.artist_number = len(self.artist_serialize_dict)
        self.album_number = len(self.album_serialize_dict)

        # print(self.item_pn_df)
        self.user = self.train_csv["user_id"]
        # self.artist = self.train_csv["artist_id"]
        # self.album = self.train_csv["album_id"]
        self.item = self.train_csv["song_id"]
        self.item_set = set(self.item)
        self.rating = self.train_csv["rating"]
        self.neg_user = self.train_csv['neg_user_id']
        self.item_serialize_dict = serialize_item(self.item)
        # User ids index the embedding table directly, so cover the largest id
        # (including neg_user_id).
        self.user_number = max(int(user_num),
                               int(self.user.max()) + 1,
                               int(self.neg_user.max()) + 1)
        self.item_number = len(self.item_set)
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.category_num = category_num
        self.user_item_interaction_dict = build_user_item_interaction_dict(train_csv)
        self.item_user_interaction_dict = build_item_user_interaction_dict(train_csv)
        # Cache the per-row columns as numpy so __getitem__ avoids a pandas
        # .iloc row lookup (which builds a Series) on every single sample.
        self._user_arr = self.train_csv["user_id"].to_numpy()
        self._item_arr = self.train_csv["song_id"].to_numpy()
        self._neg_user_arr = self.train_csv["neg_user_id"].to_numpy().astype(np.int64)
        # Pre-serialize each user's positives once, so __getitem__ only samples.
        self._user_pos_serialized = {
            u: np.array([self.item_serialize_dict[i] for i in items], dtype=np.int64)
            for u, items in self.user_item_interaction_dict.items()
        }
        self._user_pos_set = {u: set(v.tolist()) for u, v in self._user_pos_serialized.items()}
        print("Number of Users:", self.user_number, "Number of users in file", len(set(self.user)))
        print("Number of Items:", self.item_number,
              "Artists:", self.artist_number, "Albums:", self.album_number)

    def build_item_user_table(self, k_users=32, seed=0):
        """[item_number, k_users] table of user ids for the COCE module.

        The paper builds a co-occurrence embedding z_v = W_v * v from the
        multi-hot vector of users who consumed item v. Audience sizes vary, so
        we sample k_users per item WITH replacement: items with fewer than
        k_users listeners are fully covered, larger ones get a random subset.
        Mean-pooling over this fixed-width table is then a plain EmbeddingBag.
        """
        rng = np.random.default_rng(seed)
        table = np.zeros((self.item_number, k_users), dtype=np.int64)
        for song_id, users in self.item_user_interaction_dict.items():
            pos = self.item_serialize_dict.get(song_id)
            if pos is None:
                continue
            u = np.asarray(users, dtype=np.int64)
            table[pos] = u[rng.integers(0, len(u), size=k_users)] if len(u) else 0
        return torch.from_numpy(table)

    def _sample_negatives(self, n, forbidden):
        '''
        Uniformly sample n serialized item indices not in `forbidden`
        (rejection sampling).
        '''
        out = np.random.randint(0, self.item_number, size=n)
        if not forbidden:
            return out
        bad = np.array([i for i, v in enumerate(out) if v in forbidden], dtype=np.int64)
        for _ in range(10):
            if bad.size == 0:
                break
            out[bad] = np.random.randint(0, self.item_number, size=bad.size)
            bad = np.array([i for i in bad if out[i] in forbidden], dtype=np.int64)
        return out

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        user = self._user_arr[index]
        item = self._item_arr[index]
        neg_user = self._neg_user_arr[index]
        # Process item genres
        genres = np.full((self.category_num,), -1, dtype=np.int64)
        genres_index = self.genres_dict.get(item, [0, 0, 0, 0, 0])
        genres[genres_index[2:]] = 1

        # genres_index = [artist_id, album_id, genre, parent_genre, gp_genre]
        artists = self.artist_serialize_dict[genres_index[0]]
        albums = self.album_serialize_dict[genres_index[1]]

        pos_pool = self._user_pos_serialized.get(user)
        if pos_pool is None or pos_pool.size == 0:
            positive_items_list = np.zeros(self.positive_number, dtype=np.int64)
            forbidden = set()
        else:
            positive_items_list = np.random.choice(pos_pool, self.positive_number, replace=True)
            forbidden = self._user_pos_set[user]
        # Negatives are shared across this sample's positives.
        negative_items_ = self._sample_negatives(self.negative_number * 2, forbidden)
        negative_item_list = negative_items_[:self.negative_number]
        self_neg_list = negative_items_[self.negative_number:]
        return (torch.tensor(int(user)), torch.tensor(self.item_serialize_dict[item]),
                torch.from_numpy(genres), torch.tensor(int(artists)),
                torch.tensor(int(albums)), torch.tensor(int(neg_user)),
                torch.from_numpy(np.ascontiguousarray(positive_items_list)),
                torch.from_numpy(np.ascontiguousarray(negative_item_list)),
                torch.from_numpy(np.ascontiguousarray(self_neg_list)))

# Test data packaging
if __name__ == '__main__':
    print("support.py")

    genre_df = pd.read_csv("data/genre_hierarchy.csv")
    song_df = pd.read_csv("data/final_song_genre_hierarchy.csv")
    genre_num = len(genre_df["genre_id"].unique())
    song_feature_dict = {}
    for idx, row in song_df.iterrows():
        tmp_list = [row['artist_id'], row['album_id'], row['genre_id'], row['parent_genre_id'], row['gp_genre_id']]
        song_feature_dict[row['song_id']] = tmp_list
    train_csv = pd.read_csv("data/train_rating_with_neg_s2.csv") #, sep="\t", names=["user_id", "song_id", "rating"])
    print("ratings.length:", train_csv.__len__())
    users_length = len(train_csv["user_id"].unique())
    dataset = RatingDataset(train_csv, song_feature_dict, genre_num, users_length, 10, 20)
    user, item, genres, neg_user, positive_items_list, negative_item_list, self_neg_list = dataset.run_all()
    print(dataset.__len__())
    # dataIter = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # for i_index in dataIter:
    #     start = time.time()
    #     u, i, g, p_list, n_list, self_n_list = i_index
    #     print("time spend:", time.time()-start)
        # i_index += 1
    # print(u, i, g, i_f, n_user)
    # print("positive_list, negative_list, self_negative_list")
    # print("genres.shape:", g.shape, "img_f.shape:", i_f.shape)
