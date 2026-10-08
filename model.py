from myargs import get_args, args_tostring
import math
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import pandas as pd
import time
from support import RatingDataset
from test import Validate
# import wandb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # for XLA to work with JAX


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CCFCRec
class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()
        self.args = args
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(args.attr_num, args.attr_present_dim))
        # Define the attribute attention layer
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        # Controls the activation function of the entire model
        self.h = nn.LeakyReLU()
        # Mapping matrix for images
        self.sigmoid = torch.nn.Sigmoid()  # Map the gate signal between [0, 1]
        # Embedding layers for user and item, can be initialized with pre-trained ones
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
                self.artist_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
                self.album_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
                self.artist_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
                self.album_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))
            self.artist_embedding = nn.Parameter(torch.FloatTensor(args.artist_number, args.implicit_dim))
            self.album_embedding = nn.Parameter(torch.FloatTensor(args.album_number, args.implicit_dim))
        # COCE: co-occurrence module, z_v = W_v * v, where v is the multi-hot
        # vector of users who consumed item v.
        self.use_coce = bool(getattr(args, 'coce', False))
        if self.use_coce:
            self.user_basis = nn.EmbeddingBag(args.user_number, args.implicit_dim, mode='mean')
            self.register_buffer('item_users', torch.zeros(args.item_number,
                                 getattr(args, 'coce_users', 32), dtype=torch.long))
            # the free item table is unused in this mode; drop it from the params
            del self.item_embedding
        # Define the generation layer, jointly generate q_v_c with the information of
        # (q_v_a, u), generating item embedding containing collaborative information.
        self.gen_layer1 = nn.Linear(args.attr_present_dim*3, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)
        # Parameter initialization
        self.__init_param__()

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        # Initialization of the generation layer
        # Initialization of user, item embedding layer, if not pre-trained
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            if not getattr(self, 'use_coce', False):
                nn.init.xavier_normal_(self.item_embedding)
            else:
                nn.init.xavier_normal_(self.user_basis.weight)
            nn.init.xavier_normal_(self.artist_embedding)
            nn.init.xavier_normal_(self.album_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def set_item_users(self, table):
        self.item_users.copy_(table.to(self.item_users.device))

    def item_emb(self, idx):
        '''Item embedding for serialized item indices of ANY shape.'''
        if not self.use_coce:
            return self.item_embedding[idx]
        shape = idx.shape
        # EmbeddingBag pools without materialising a [..., K, D] tensor
        pooled = self.user_basis(self.item_users[idx.reshape(-1)])
        return pooled.reshape(*shape, -1)

    def forward(self, attribute, artist, album, batch_size):
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1)+self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2).to(device)
        neg_inf = torch.full(z_v_squeeze.shape, -1e6).to(device)
        if attribute.dim() == 3:
            attribute = attribute.squeeze(-1)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)
        ar_v = artist
        al_v = album
        q_v_a = torch.cat((final_attr_emb, ar_v, al_v), dim=1)
        # q_v_a = torch.cat((final_attr_emb, p_v), dim=1)
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        # q_v_c = self.gen_layer2(self.h(self.gen_layer1(final_attr_emb)))
        return q_v_c


def train(model, train_loader, optimizer, valida, args, model_save_dir):
    print("model start train!")
    test_save_path = model_save_dir + "/result.csv"
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # Write hyperparameters
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    with open(test_save_path, 'a+') as f:
        f.write("loss,contrast_sum,hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n")
    save_index = 0
    for i_epoch in range(args.epoch):
        i_batch = 0
        batch_time = time.time()
        for user, item, item_genres, artist, album, neg_user, positive_item_list, negative_item_list, self_neg_list in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            # item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            negative_item_list = negative_item_list.to(device)
            self_neg_list = self_neg_list.to(device)
            # run model
            # if artist id is larger than artist embedding change to zero
            artist = artist.to(device)
            album = album.to(device)
            artist = torch.where(artist < len(model.artist_embedding), artist, torch.zeros(artist.shape).to(device))
            album = torch.where(album < len(model.album_embedding), album, torch.zeros(album.shape).to(device))
            # cast to int
            artist = artist.to(torch.int64)
            album = album.to(torch.int64)
            artist_emb = model.artist_embedding[artist]
            album_emb = model.album_embedding[album]
            # if artist < len(model.artist_embedding):
            #     artist_emb = model.artist_embedding[artist]
            # else:
            #     artist_emb = torch.zeros(1, 256).to(device)
            #
            # if album < len(model.album_embedding):
            #     album_emb = model.album_embedding[album]
            # else:
            #     album_emb = torch.zeros(1, 256).to(device)
            q_v_c = model(item_genres, artist_emb, album_emb, user.shape[0])
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)
            # --- co-occurrence contrastive loss (InfoNCE over cosine sims) ---
            # Normalising once and using matmuls avoids materialising the old
            # [B, P, N, D] broadcast tensor, which dominated the step time.
            q_n = torch.nn.functional.normalize(q_v_c, dim=1)                    # [B, D]
            pos_n = torch.nn.functional.normalize(
                model.item_emb(positive_item_list), dim=2)                 # [B, P, D]
            neg_n = torch.nn.functional.normalize(
                model.item_emb(negative_item_list), dim=2)                 # [B, N, D]

            pos_sim = torch.einsum('bd,bpd->bp', q_n, pos_n) / args.tau          # [B, P]
            neg_sim = torch.einsum('bd,bnd->bn', q_n, neg_n) / args.tau          # [B, N]
            # logsumexp over [positive_p, all negatives] is the numerically
            # stable form of -log(exp(pos) / (exp(pos) + sum exp(neg)))
            neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)              # [B, 1]
            contrast_val = -(pos_sim - torch.logaddexp(pos_sim, neg_lse.expand_as(pos_sim)))
            contrast_sum = contrast_val.mean()
            '''
            contrast self
            '''
            self_neg_n = torch.nn.functional.normalize(
                model.item_emb(self_neg_list), dim=2)                      # [B, N, D]
            item_emb = model.item_emb(item)
            self_pos_sim = (q_n * torch.nn.functional.normalize(item_emb, dim=1)).sum(1) / args.tau
            self_neg_sim = torch.einsum('bd,bnd->bn', q_n, self_neg_n) / args.tau
            self_neg_lse = torch.logsumexp(self_neg_sim, dim=1)
            self_contrast_val = -(self_pos_sim - torch.logaddexp(self_pos_sim, self_neg_lse))
            self_contrast_sum = self_contrast_val.mean()
            # Compute the BPR rank loss.
            user_emb = model.user_embedding[user]
            item_emb = model.item_emb(item)
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
            y_ukv = -logsigmoid(y_uv - y_kv).mean()
            # Backpropagate the total loss and update model parameters.
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).mean()
            total_loss = args.lambda1*(contrast_sum+self_contrast_sum) + (1-args.lambda1)*(y_ukv+y_ukv2)
            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            # wandb.log({
            #         'epoch': i_epoch,
            #         'i_batch': i_batch,
            #         'total_loss': total_loss.item(),
            #         'contrast_sum': contrast_sum,
            # })
            if i_batch % args.save_batch_time == 0:
                model.eval()
                print("[{}/{}] total_loss: {:.4f}  ({}s)".format(
                        i_batch * args.batch_size, len(train_loader.dataset),
                        total_loss.item(), int(time.time()-batch_time)))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = valida.start_validate(model)
                    if getattr(valida, 'warm', None) is not None:
                        w = valida.warm.start_validate(model, verbose=False)
                        print("warm      {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}"
                              "   <- seen in training".format(*w))
                with open(test_save_path, 'a+') as f:
                    f.write("{},{},{},{},{},{},{},{}\n".format(total_loss.item(), contrast_sum, hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20))
                # Save the model

                # wandb.log({
                #     'epoch': i_epoch,
                #     'batch_time': int(time.time()-batch_time),
                #     'hr_5': hr_5,
                #     'hr_10': hr_10,
                #     'hr_20': hr_20,
                #     'ndcg_5': ndcg_5,
                #     'ndcg_10':ndcg_10,
                #     'ndcg_20': ndcg_20
                # })
                batch_time = time.time()
                save_index += 1
                model_save_path = model_save_dir + '/' + str(save_index)+".pt"
                # torch.save(model.state_dict(), model_save_path)
                # wandb.save(model_save_path)


if __name__ == '__main__':
    # result save dir
    args = get_args()
    torch.set_num_threads(args.threads or max(1, os.cpu_count() - args.num_workers))
    save_dir = 'result/' + time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())) + (('_' + args.tag) if args.tag else '')
    os.makedirs(save_dir)
    # wandb.init(project='Yahoo', config=args)
    # wandb.run.name = 'Yahoo b{} l{} -- {}'.format(args.batch_size, args.learning_rate, wandb.run.id)
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = args.train_path
    valid_path = args.valid_path
    train_df = pd.read_csv(train_path)
    genre_df = pd.read_csv("data/genre_hierarchy.csv")
    genre_num = len(genre_df["genre_id"].unique())
    song_df = pd.read_csv("data/final_song_genre_hierarchy.csv")
    user_set = set(train_df['user_id'])
    song_feature_dict = {}
    for idx, row in song_df.iterrows():
        tmp_list = [row['artist_id'], row['album_id'], row['genre_id'], row['parent_genre_id'], row['gp_genre_id']]
        song_feature_dict[row['song_id']] = tmp_list
    if args.shuffle_content:
        # Control: reattach every content vector to a random song.
        import numpy as _np
        _rng = _np.random.default_rng(args.shuffle_content)
        _keys = list(song_feature_dict.keys())
        _vals = [song_feature_dict[k] for k in _keys]
        _perm = _rng.permutation(len(_vals))
        song_feature_dict = {k: _vals[_perm[i]] for i, k in enumerate(_keys)}
        print("control: item->content shuffled")

    dataSet = RatingDataset(train_df, song_feature_dict, genre_num, len(user_set), args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    args.artist_number = dataSet.artist_number
    args.album_number = dataSet.album_number
    train_loader = torch.utils.data.DataLoader(dataSet, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    print("Model hyperparameters:", args_tostring(args))
    myModel = CCFCRec(args)
    if args.coce:
        myModel.set_item_users(dataSet.build_item_user_table(args.coce_users))
        print("COCE enabled: item table {} -> user basis {}".format(
            tuple(myModel.item_users.shape), tuple(myModel.user_basis.weight.shape)))
    print("trainable parameters: {:.2f}M".format(
        sum(p.numel() for p in myModel.parameters()) / 1e6))
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    validator = Validate(validate_csv=valid_path, genres=song_feature_dict, category_num=genre_num,
                         artist_serialize_dict=dataSet.artist_serialize_dict,
                         album_serialize_dict=dataSet.album_serialize_dict,
                         max_items=(args.eval_items or None))
    if args.warm_path:
        # Control: same metric on warm items seen in training.
        validator.warm = Validate(validate_csv=args.warm_path, genres=song_feature_dict,
                                  category_num=genre_num,
                                  artist_serialize_dict=dataSet.artist_serialize_dict,
                                  album_serialize_dict=dataSet.album_serialize_dict,
                                  max_items=(args.eval_items or None))
    print("\n--- baselines ---")
    validator.report_baselines(train_df, args.user_number)
    print("--- untrained model ---")
    validator.start_validate(myModel)
    print()
    train(myModel, train_loader, optimizer, validator, args, save_dir)

