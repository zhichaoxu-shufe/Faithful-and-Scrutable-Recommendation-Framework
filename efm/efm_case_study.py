from dataset import *
from model import *
from evaluate import *
from engine import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, math, os, json, copy, logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--failed_case', type=str, default='/raid/brutusxu/agnostic/case_study/failed_items_Electronics_efm.pickle')
parser.add_argument('--dest_dir', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3')
parser.add_argument('--dataset', type=str, default='Electronics')
parser.add_argument('--checkpoint', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3/efm_checkpoints/efm_epoch_4.pt')

parser.add_argument('--num_users', type=int, default=3151)
parser.add_argument('--num_items', type=int, default=3253)
parser.add_argument('--num_aspect', type=int, default=200)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--l2', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=0)

parser.add_argument('--top_aspect', type=int, default=5)
args = parser.parse_args()

config = vars(args)

failed_case = np.load(args.failed_case, allow_pickle=True)

engine = EFMEngine(config)
dataset = EFMDataset(args.dest_dir, args.dataset)

item_pool = set([i for i in range(args.num_items)])

test_loader = dataset.instance_a_test_loader(1, False, 1)

engine.model.load_state_dict(torch.load(args.checkpoint))

engine.output_ranklist_mask(test_loader, item_pool, failed_case, args.top_aspect)