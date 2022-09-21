import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', type=str, default='model_default',
                    help='experiment in experiments.exps')
parser.add_argument('-r', '--repeat', type=int, default=1, help='repeat times')
parser.add_argument('-e', '--epoch', type=int, default=60, help='number of epoch')
parser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--step', type=int, default=20, help='evaluation step')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--valid', dest='is_test', action='store_false')
parser.set_defaults(is_test=True)

# data
parser.add_argument('-d', '--dataset', type=str, default='ali_home@posq3_last',
                    help='dataset registered in cfg_data')
parser.add_argument('-s', '--strategy', type=str, default='u_last2_uq',
                    help='split strategy, (u_last2_uq)')
parser.add_argument('-p', '--prop', type=float, default=1,
                    help='proportion of sampling from the whole dataset')
# device
parser.add_argument('--cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)

# control log output
parser.add_argument('--dir_log', type=str, default='./logs', help='caption')
parser.add_argument('-c', '--caption', type=str, default='ndcg_uq', help='caption')
parser.add_argument('--mute', dest='output', action='store_false')
parser.set_defaults(output=True)
parser.add_argument('--fast', dest='fast', action='store_true')
parser.set_defaults(fast=False)

arg = parser.parse_args()

if arg.fast:
    arg.repeat = 1
    arg.epoch = 2
    arg.step = 2
