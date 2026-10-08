import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help="batch_size") #1024
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--positive_number', type=int, default=5, help='contrast positive number')
    parser.add_argument('--negative_number', type=int, default=40, help='contrast negative number')
    parser.add_argument('--self_negative_number', type=int, default=40, help='self contrast negative number')
    parser.add_argument('--attr_num', type=int, default=216, help='item attribute number')
    parser.add_argument('--attr_present_dim', type=int, default=256, help='the dimension of present')
    parser.add_argument('--implicit_dim', type=int, default=256, help='the dimension of u/i present')
    parser.add_argument('--cat_implicit_dim', type=int, default=256, help='the q_v_c dimension')
    parser.add_argument('--user_number', type=int, default=138493, help='user number in training set')
    parser.add_argument('--item_number', type=int, default=16803, help='item number in training set')
    parser.add_argument('--artist_number', type=int, default=1, help='artist number, overwritten from dataset')
    parser.add_argument('--album_number', type=int, default=1, help='album number, overwritten from dataset')
    parser.add_argument('--tau', type=float, default=0.1, help='contrast loss temperature')
    parser.add_argument('--lambda1', type=float, default=0.6, help='collaborative contrast loss weight')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--pretrain', type=bool, default=False, help='user/item embedding pre-training')
    parser.add_argument('--pretrain_update', type=bool, default=False, help='u/i pretrain embedding update')
    parser.add_argument('--contrast_flag', type=bool, default=True, help='contrast job flag')
    parser.add_argument('--user_flag', type=bool, default=False, help='use user to q_v_c flag')
    parser.add_argument('--save_batch_time', type=int, default=95, help='every batch time save the model')
    parser.add_argument('--shuffle_content', type=int, default=0,
                        help='control: permute item->content with this random state (0=off)')
    parser.add_argument('--warm_path', type=str, default='',
                        help='control: also evaluate on these warm items')
    parser.add_argument('--coce', action='store_true',
                        help="use the paper's co-occurrence module z_v = W_v*v instead of a free item lookup table")
    parser.add_argument('--coce_users', type=int, default=32, help='users sampled per item for COCE')
    parser.add_argument('--threads', type=int, default=0, help='torch threads (0=all cores not used by workers)')
    parser.add_argument('--tag', type=str, default='', help='label for the run directory')
    parser.add_argument('--train_path', type=str, default='data/train_cold.csv')
    parser.add_argument('--valid_path', type=str, default='data/test_cold.txt')
    parser.add_argument('--dim', type=int, default=0, help='if >0, sets attr_present/implicit/cat_implicit dims')
    parser.add_argument('--eval_items', type=int, default=4000, help='validation item subsample (0=all)')
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers (0 on Windows)')
    a = parser.parse_args()
    if a.dim > 0:
        a.attr_present_dim = a.implicit_dim = a.cat_implicit_dim = a.dim
    return a


def args_tostring(args):
    str_ = ""
    for arg in vars(args):
        str_ += str(arg) + ":" + str(getattr(args, arg)) + "\n"
    return str_
