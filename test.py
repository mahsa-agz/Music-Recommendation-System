import random

import pandas as pd
import torch
import numpy as np
import time
import os
from myargs import get_args
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dcg_k(r):
    '''
    Discounted Cumulative Gain at k. Relevance is binary here, so
    (2^r - 1) reduces to r, but the general form is kept.
    '''
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2))))


def hr_at_k(item, recommend_users, item_user_dict, k):
    '''
    Hit Ratio at k: 1 if any ground-truth user appears in the top-k
    recommended users, else 0.
    '''
    groundtruth_user = item_user_dict.get(item)
    if not groundtruth_user:
        return 0.0
    return 1.0 if set(recommend_users[0:k]) & set(groundtruth_user) else 0.0


def ndcg_k(item, recommend_users, item_user_dict, k):
    '''
    NDCG at k. The ideal DCG uses the number of relevant users capped at k.
    '''
    groundtruth_user = item_user_dict.get(item)
    if not groundtruth_user:
        return 0.0
    gt = set(groundtruth_user)
    ratings = [1.0 if u in gt else 0.0 for u in recommend_users[0:k]]
    ideal_dcg = dcg_k([1.0] * min(len(gt), k))
    if ideal_dcg == 0:
        return 0.0
    return dcg_k(ratings) / ideal_dcg


def precision_at_k(item, recommend_users, item_user_dict, k):
    '''Precision at k.'''
    groundtruth_user = item_user_dict.get(item)
    if not groundtruth_user:
        return 0.0
    return len(set(recommend_users[0:k]) & set(groundtruth_user)) / k


def _score_items(model, genres_batch, artist_batch, album_batch):
    '''Return a [batch, n_users] score matrix for a batch of items.'''
    bsz = genres_batch.shape[0]
    artist_emb = model.artist_embedding[artist_batch]
    album_emb = model.album_embedding[album_batch]
    q_v_c = model(genres_batch, artist_emb, album_emb, bsz)
    return torch.matmul(q_v_c, model.user_embedding.t())


def get_similar_user_speed(model, genres, artist_idx, album_idx, k):
    '''
    Top-k users for a single item. The old signature was model(genres, 1),
    which did not match CCFCRec.forward(attribute, artist, album, batch_size)
    and raised TypeError on the first validation pass.
    '''
    genres = genres.unsqueeze(dim=0).to(device)
    artist_idx = torch.tensor([artist_idx], dtype=torch.int64, device=device)
    album_idx = torch.tensor([album_idx], dtype=torch.int64, device=device)
    ratings = _score_items(model, genres, artist_idx, album_idx)[0]
    return torch.argsort(-ratings)[0:k].cpu().detach().numpy().tolist()


class Validate:
    def __init__(self, validate_csv, genres, category_num,
                 artist_serialize_dict=None, album_serialize_dict=None,
                 eval_batch_size=256, max_items=None, seed=0):
        print("validate class init")
        df = pd.read_csv(validate_csv, sep='\t', names=['user_id', 'song_id', 'rating'])
        grouped = df.groupby('song_id')['user_id'].apply(lambda s: set(s.tolist()))
        self.item_user_dict = grouped.to_dict()
        items = sorted(self.item_user_dict.keys())
        if max_items is not None and len(items) > max_items:
            rng = random.Random(seed)
            items = sorted(rng.sample(items, max_items))
            self.item_user_dict = {i: self.item_user_dict[i] for i in items}
        self.item = items
        self.genres_dict = genres
        self.category_num = category_num
        self.artist_serialize_dict = artist_serialize_dict or {}
        self.album_serialize_dict = album_serialize_dict or {}
        self.eval_batch_size = eval_batch_size
        # Build the per-item attribute tensors once, not on every validation pass
        self.genres_mat, self.artist_idx, self.album_idx = self._build_features()
        print("validate items: {}, mean ground-truth users/item: {:.3f}".format(
            len(self.item), float(np.mean([len(v) for v in self.item_user_dict.values()]))))

    def _build_features(self):
        genres_mat = torch.full((len(self.item), self.category_num), -1, dtype=torch.long)
        artist_idx = torch.zeros(len(self.item), dtype=torch.int64)
        album_idx = torch.zeros(len(self.item), dtype=torch.int64)
        for row, it in enumerate(self.item):
            feat = self.genres_dict.get(it, [0, 0, 0, 0, 0])
            genres_mat[row, feat[2:]] = 1
            # feat = [artist_id, album_id, genre_id, parent_genre_id, gp_genre_id]
            artist_idx[row] = self.artist_serialize_dict.get(feat[0], 0)
            album_idx[row] = self.album_serialize_dict.get(feat[1], 0)
        return genres_mat, artist_idx, album_idx

    def _metrics_from_ranking(self, topk_all):
        '''topk_all: [n_items, max_k] array of already-ranked user ids.'''
        out = {}
        for k in (5, 10, 20):
            hr = float(np.mean([hr_at_k(it, topk_all[i], self.item_user_dict, k)
                                for i, it in enumerate(self.item)]))
            nd = float(np.mean([ndcg_k(it, topk_all[i], self.item_user_dict, k)
                                for i, it in enumerate(self.item)]))
            pr = float(np.mean([precision_at_k(it, topk_all[i], self.item_user_dict, k)
                                for i, it in enumerate(self.item)]))
            out[k] = (hr, nd, pr)
        return out

    def start_validate(self, model, max_k=20, verbose=True):
        model = model.to(device)
        model.eval()
        topk_all = np.zeros((len(self.item), max_k), dtype=np.int64)
        with torch.no_grad():
            for s in range(0, len(self.item), self.eval_batch_size):
                e = min(s + self.eval_batch_size, len(self.item))
                scores = _score_items(model,
                                      self.genres_mat[s:e].to(device),
                                      self.artist_idx[s:e].to(device),
                                      self.album_idx[s:e].to(device))
                topk_all[s:e] = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()
        m = self._metrics_from_ranking(topk_all)
        hr_5, ndcg_5, p_5 = m[5]
        hr_10, ndcg_10, p_10 = m[10]
        hr_20, ndcg_20, p_20 = m[20]
        if verbose:
            print("            HR@5      HR@10     HR@20     NDCG@5    NDCG@10   NDCG@20")
            print("model     {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20))
            print("          (precision@5/10/20: {:.6f} {:.6f} {:.6f})".format(p_5, p_10, p_20))
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20

    # ---------------- baselines ----------------

    def baseline_random(self, n_users, max_k=20, seed=0):
        rng = np.random.default_rng(seed)
        topk_all = np.stack([rng.choice(n_users, max_k, replace=False)
                             for _ in range(len(self.item))])
        return self._metrics_from_ranking(topk_all)

    def baseline_popularity(self, train_df, n_users, max_k=20):
        '''Always recommend the most active users, ignoring the item.'''
        pop = train_df['user_id'].value_counts().index.to_numpy()[:max_k]
        if len(pop) < max_k:
            pop = np.pad(pop, (0, max_k - len(pop)), constant_values=0)
        topk_all = np.tile(pop[:max_k], (len(self.item), 1))
        return self._metrics_from_ranking(topk_all)

    def report_baselines(self, train_df, n_users, max_k=20):
        rows = [("random", self.baseline_random(n_users, max_k)),
                ("pop", self.baseline_popularity(train_df, n_users, max_k))]
        print("            HR@5      HR@10     HR@20     NDCG@5    NDCG@10   NDCG@20")
        for name, m in rows:
            print("{:<10}{:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                name, m[5][0], m[10][0], m[20][0], m[5][1], m[10][1], m[20][1]))
        return dict(rows)
