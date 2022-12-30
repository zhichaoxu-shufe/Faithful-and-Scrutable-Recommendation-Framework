from functions import *
from dataset import *
from agnostic_model import *
from evaluate import *
from engine import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, math, os, json, copy, logging
from tqdm import tqdm

import torch
import torch.nn as nn
from six.moves import xrange
from torch.utils.data import DataLoader, Dataset


def wbx_topk(model, user_indice, item_indice, k):
	user_embed = model.user_embeddings(user_indice)
	item_embed = model.item_embeddings(item_indice)
	aspect_embed = model.aspect_embeddings.weight

	Ru = model.relation_embeddings.weight[0]
	Rip = model.relation_embeddings.weight[1]
	Rin = model.relation_embeddings.weight[2]

	P = torch.mul(user_embed+Ru, aspect_embed).sum(dim=1)
	Qip = torch.mul(item_embed+Rip, aspect_embed).sum(dim=1)
	Qin = torch.mul(item_embed+Rin, aspect_embed).sum(dim=1)

	# rating = P * Qip - P * Qin
	rating = P * Qip
	aspect_indices = torch.topk(rating, k)[1]


	return aspect_indices.tolist()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ranklist', type=str, required=True)
	parser.add_argument('--checkpoint', type=str, required=True)
	parser.add_argument('--data_dir', type=str, required=True)
	parser.add_argument('--check_wbx_perf', type=int, default=0)

	parser.add_argument('--num_user', type=int, default=3151)
	parser.add_argument('--num_item', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--num_relation', type=int, default=3)
	parser.add_argument('--uia_path', type=str, default="empty")
	parser.add_argument('--use_TransE', type=int, default=0)

	parser.add_argument('--k', type=int, default=5)

	args = parser.parse_args()
	config = vars(args)

	model = SimpleModel(config)
	model.load_state_dict(torch.load(args.checkpoint))

	ranklist = np.load(args.ranklist, allow_pickle=True)

	ui_pairs = []
	for i, row in enumerate(ranklist):
		for j in range(20):
			ui_pairs.append(torch.tensor([row[0], row[1][j]]))
	aspect_lists = []

	pbar = tqdm(total=len(ui_pairs))	
	for ui_pair in ui_pairs:
		top_aspect = wbx_topk(model, ui_pair[0], ui_pair[1], args.k)
		aspect_lists.append([ui_pair[0].item(), ui_pair[1].item(), top_aspect])
		pbar.update(1)
	pbar.close()

	fn = "_".join(args.checkpoint.split("/")[-2].split("_")[:-1]) + ".pickle"
	if not os.path.isdir(os.path.join(args.data_dir, 'common_aspects')):
		os.mkdir(os.path.join(args.data_dir, 'common_aspects'))
	np.array(aspect_lists).dump(os.path.join(args.data_dir, 'common_aspects', fn))