from dataset import *
from model import *
from evaluate import *
from engine import *

import argparse
import numpy as np
import pandas as pd
import random
import sys
import os
import json
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Electronics')
parser.add_argument('--input_dir', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3')
parser.add_argument('--matrix_path', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3')
parser.add_argument('--embed_model_path', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3/a2cf_pretrain')
parser.add_argument('--failed_case', type=str, default='/raid/brutusxu/agnostic/case_study/sm_failed_items_Electronics_vanilla.pickle')

parser.add_argument('--num_user', type=int, default=3151)
parser.add_argument('--num_item', type=int, default=3253)
parser.add_argument('--num_aspect', type=int, default=200)

parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--use_cuda', type=int, default=1)
parser.add_argument('--epsilon', type=float, default=8.)
parser.add_argument('--num_negative', type=int, default=7)
parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('--top_aspect', type=int, default=1)

args = parser.parse_args()
config = vars(args)

engine = A2CFEngine(config)
dataset = A2CFDataset(args.input_dir, args.dataset)
gt = np.load(os.path.join(args.input_dir, 'test.pickle'), allow_pickle=True)
item_pool = set([i for i in range(args.num_item)])

failed_case = np.load(args.failed_case, allow_pickle=True)
interact_hist = dataset.history

test_loader = dataset.instance_a_test_loader(1, True, 4)

checkpoint = os.path.join(args.input_dir, 'a2cf_checkpoints/a2cf_epoch_1.pt')
engine.model.load_state_dict(torch.load(checkpoint))

engine.output_ranklist_mask(test_loader, item_pool, interact_hist, failed_case, args.top_aspect)