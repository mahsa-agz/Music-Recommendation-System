"""
Content-KNN baseline for cold-start item -> user ranking.

For a cold song v, find the training songs most similar to it by content
(genre / parent genre / grandparent genre / artist / album), then rank users by
how often they appear in those neighbours' audiences.

Uses the same information as the model (item content + warm interactions)
with no learning, and the same metric code.

Usage:
  python knn_baseline.py [--k 50] [--eval_items 3000]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse

from test import Validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', default='data/train_cold.csv')
    ap.add_argument('--valid_path', default='data/test_cold.txt')
    ap.add_argument('--k', type=int, default=50, help='neighbours per cold item')
    ap.add_argument('--eval_items', type=int, default=3000)
    args = ap.parse_args()

    train = pd.read_csv(args.train_path)
    song_df = pd.read_csv("data/final_song_genre_hierarchy.csv")
    genre_num = len(pd.read_csv("data/genre_hierarchy.csv")["genre_id"].unique())
    sfd = {r.song_id: [r.artist_id, r.album_id, r.genre_id, r.parent_genre_id, r.gp_genre_id]
           for r in song_df.itertuples()}

    val = Validate(args.valid_path, sfd, genre_num, max_items=(args.eval_items or None))
    n_users = int(max(train.user_id.max(), 0)) + 1

    # ---- content feature matrix over train items -------------------------
    train_items = np.sort(train.song_id.unique())
    item_pos = {s: i for i, s in enumerate(train_items)}
    artist_ids = sorted({sfd[s][0] for s in train_items})
    album_ids = sorted({sfd[s][1] for s in train_items})
    a_off, al_off = genre_num, genre_num + len(artist_ids)
    a_pos = {a: i for i, a in enumerate(artist_ids)}
    al_pos = {a: i for i, a in enumerate(album_ids)}
    n_feat = al_off + len(album_ids)

    def feat_rows(songs):
        rows, cols = [], []
        for r, s in enumerate(songs):
            f = sfd.get(s, [0, 0, 0, 0, 0])
            for g in set(f[2:]):
                rows.append(r); cols.append(int(g))
            if f[0] in a_pos:
                rows.append(r); cols.append(a_off + a_pos[f[0]])
            if f[1] in al_pos:
                rows.append(r); cols.append(al_off + al_pos[f[1]])
        m = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(songs), n_feat))
        norms = np.sqrt(m.multiply(m).sum(1)).A.ravel()
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ m

    t0 = time.time()
    X_train = feat_rows(train_items)
    X_test = feat_rows(val.item)
    print("feature matrices built in {:.1f}s  ({} train items, {} feats)".format(
        time.time() - t0, len(train_items), n_feat))

    # user x item interaction matrix (train)
    ui = sparse.csr_matrix(
        (np.ones(len(train), dtype=np.float32),
         (train.user_id.to_numpy(), [item_pos[s] for s in train.song_id.to_numpy()])),
        shape=(n_users, len(train_items)))

    # ---- rank users for each cold item -----------------------------------
    t0 = time.time()
    max_k = 20
    topk = np.zeros((len(val.item), max_k), dtype=np.int64)
    B = 256
    for s in range(0, len(val.item), B):
        e = min(s + B, len(val.item))
        sim = (X_test[s:e] @ X_train.T).toarray()          # [b, n_train_items]
        if args.k < sim.shape[1]:
            cut = np.partition(sim, -args.k, axis=1)[:, -args.k][:, None]
            sim[sim < cut] = 0.0
        scores = sim @ ui.T.toarray().T if False else (sparse.csr_matrix(sim) @ ui.T).toarray()
        topk[s:e] = np.argpartition(-scores, max_k, axis=1)[:, :max_k]
        order = np.argsort(-np.take_along_axis(scores, topk[s:e], axis=1), axis=1)
        topk[s:e] = np.take_along_axis(topk[s:e], order, axis=1)
    print("scored {} cold items in {:.1f}s".format(len(val.item), time.time() - t0))

    m = val._metrics_from_ranking(topk)
    print("\n            HR@5      HR@10     HR@20     NDCG@5    NDCG@10   NDCG@20")
    print("knn(k={:<3}) {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
        args.k, m[5][0], m[10][0], m[20][0], m[5][1], m[10][1], m[20][1]))
    val.report_baselines(train, n_users)


if __name__ == '__main__':
    main()
