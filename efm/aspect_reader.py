from functions import *
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
from six.moves import xrange
from torch.utils.data import DataLoader, Dataset


def efm_topk(model, user_indice, item_indice, k):
	user_emb = model.embedding_user(user_indice)
	item_emb = model.embedding_item(item_indice)
	aspect_emb = model.embedding_aspect.weight
	user_aspect_emb = torch.mul(user_emb, aspect_emb)
	item_aspect_emb = torch.mul(item_emb, aspect_emb)
	aspect_score = torch.mul(user_aspect_emb, item_aspect_emb).sum(dim=1)
	aspect_indices = torch.topk(aspect_score, k)[1]
	return aspect_indices.tolist()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ranklist', type=str, required=True)
	parser.add_argument('--checkpoint', type=str, required=True)
	parser.add_argument('--data_dir', type=str, required=True)

	parser.add_argument('--num_users', type=int, default=3151)
	parser.add_argument('--num_items', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)
	parser.add_argument('--latent_dim', type=int, default=128)

	parser.add_argument('--k', type=int, default=5)


	args = parser.parse_args()
	config = vars(args)

	with open(args.ranklist, 'r') as f:
		ranklist = json.load(f)

	ui_pairs = []
	for k, v in ranklist.items():
		for i in range(20):
			ui_pairs.append(torch.tensor([int(k), int(v[i])]))

	aspect_lists = []
	
	model = EFM(config)
	model.load_state_dict(torch.load(args.checkpoint))
	
	pbar = tqdm(total=len(ui_pairs))
	
	for ui_pair in ui_pairs:
		top_aspect = efm_topk(model, ui_pair[0], ui_pair[1], args.k)
		aspect_lists.append([ui_pair[0].item(), ui_pair[1].item(), top_aspect])
		pbar.update(1)
	pbar.close()

	if not os.path.isdir(os.path.join(args.data_dir, 'common_aspects')):
		os.mkdir(os.path.join(args.data_dir, 'common_aspects'))
	np.array(aspect_lists).dump(os.path.join(args.data_dir, 'common_aspects', 'efm_top_aspects.pickle'))