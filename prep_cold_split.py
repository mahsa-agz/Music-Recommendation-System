"""
Build a cold-start item split.

Holds out whole songs: every interaction of a held-out song goes to the test
side, so the model must generate its embedding from content alone.
Also regenerates neg_user_id, excluding users who interacted with the song.

Outputs:
  data/train_cold.csv   user_id,song_id,rating,neg_user_id
  data/test_cold.txt    user_id \t song_id \t rating   (format Validate expects)
"""
import numpy as np
import pandas as pd

SEED = 42
TEST_ITEM_FRAC = 0.20


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("data/train_rating_with_neg_1.csv")
    print("input: {} rows, {} users, {} songs".format(
        len(df), df.user_id.nunique(), df.song_id.nunique()))

    items = df.song_id.unique()
    rng.shuffle(items)
    n_cold = int(TEST_ITEM_FRAC * len(items))
    cold = set(items[:n_cold].tolist())

    is_cold = df.song_id.isin(cold)
    train = df[~is_cold].copy()
    test = df[is_cold].copy()

    # Keep only users the model will actually have an embedding for.
    train_users = set(train.user_id.unique())
    test = test[test.user_id.isin(train_users)].copy()

    # --- regenerate negative users, excluding the item's true audience -------
    item_users = train.groupby('song_id')['user_id'].apply(set).to_dict()
    all_users = np.array(sorted(train_users))
    neg = rng.choice(all_users, size=len(train))
    song_arr = train.song_id.to_numpy()
    user_arr = train.user_id.to_numpy()
    for _ in range(12):
        bad = np.array([i for i in range(len(neg))
                        if neg[i] == user_arr[i] or neg[i] in item_users.get(song_arr[i], ())],
                       dtype=np.int64)
        if bad.size == 0:
            break
        neg[bad] = rng.choice(all_users, size=bad.size)
    train['neg_user_id'] = neg
    print("negative users still colliding with a true positive: {}".format(int(bad.size)))

    train.to_csv("data/train_cold.csv", index=False)
    test[['user_id', 'song_id', 'rating']].to_csv(
        "data/test_cold.txt", sep='\t', header=False, index=False)

    tr_items, te_items = set(train.song_id.unique()), set(test.song_id.unique())
    print("\ntrain : {:>7} rows  {:>5} users  {:>6} songs".format(
        len(train), train.user_id.nunique(), len(tr_items)))
    print("test  : {:>7} rows  {:>5} users  {:>6} songs".format(
        len(test), test.user_id.nunique(), len(te_items)))
    print("item leakage train<->test : {}".format(len(tr_items & te_items)))
    print("mean ground-truth users per test song : {:.3f}".format(
        len(test) / max(len(te_items), 1)))
    print("random HR@5 for this test set ~ {:.5f}".format(
        1 - (1 - (len(test) / len(te_items)) / train.user_id.nunique()) ** 5))


if __name__ == '__main__':
    main()
