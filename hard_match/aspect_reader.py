from functions import *
from dataset import *
from hard_match_model import *
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


def wbx_topk(model, user_idx, item_idx, k):
	"""
	Args:

	Returns:
		aspect_indices: `List` with length of k
	"""
	meta_weights = model.meta_weights
	ua_scores = torch.sigmoid(model.ua_embeddings.weight) * model.ua_mask_mtx
	ia_scores = torch.sigmoid(model.ia_embeddings.weight) * model.ia_mask_mtx

	# meta 1 
	aspect_1 = ua_scores[user_idx, :] * ia_scores[item_idx, :] * meta_weights[0]

	# meta 2 
	# aspect_21 = ua_scores[user_idx, :] * (ia_scores.T @ ia_scores @ ia_scores.T).T[item_idx, :] * meta_weights[1]
	aspect_22 = (ua_scores @ ia_scores.T @ ia_scores)[user_idx, :] * ia_scores[item_idx, :] * meta_weights[1]

	# meta 3 
	# aspect_31 = ua_scores[user_idx, :] * (ua_scores.T @ ua_scores @ ia_scores.T).T[item_idx, :] * meta_weights[2]
	aspect_32 = (ua_scores @ ua_scores.T @ ua_scores)[user_idx, :] * ia_scores[item_idx, :] * meta_weights[2] 

	aspect = aspect_1 + aspect_21 + aspect_22 + aspect_31 + aspect_32
	aspect = aspect_1 + aspect_22 + aspect_32
	#print(aspect_1.topk(k)[1], aspect_21.topk(k)[1], aspect_22.topk(k)[1], aspect_31.topk(k)[1], aspect_32.topk(k)[1])
	aspect_indices = torch.topk(aspect, k)[1]

	return aspect_indices.tolist()

def uia_lookup(model):
	"""
	Return:
		lookup: [num_user, num_item, num_aspect]
	"""
	meta_weights = model.meta_weights
	ua_scores = torch.sigmoid(model.ua_embeddings.weight) * model.ua_mask_mtx
	ia_scores = torch.sigmoid(model.ia_embeddings.weight) * model.ia_mask_mtx
	cpn_21 = (ia_scores.T @ ia_scores @ ia_scores.T).T 
	cpn_22 = ua_scores @ ia_scores.T @ ia_scores
	cpn_31 = (ua_scores.T @ ua_scores @ ia_scores.T).T 
	cpn_32 = ua_scores @ ua_scores.T @ ua_scores

	num_user, num_item, num_aspect = model.num_user, model.num_item, model.num_aspect

	# meta 1
	a1_lookup = ua_scores.view(num_user, 1, num_aspect) * ia_scores.view(1, num_item, num_aspect) * meta_weights[0] #[num_user, num_item, num_aspect]
	
	# meta 2
	a21_lookup = ua_scores.view(num_user, 1, num_aspect) * cpn_21.view(1, num_item, num_aspect) * meta_weights[1]
	a22_lookup = cpn_22.view(num_user, 1, num_aspect) * ia_scores.view(1, num_item, num_aspect) * meta_weights[1]

	# meta 3
	a31_lookup = ua_scores.view(num_user, 1, num_aspect) * cpn_31.view(1, num_item, num_aspect) * meta_weights[2]
	a32_lookup = cpn_32.view(num_user, 1, num_aspect) * ia_scores.view(1, num_item, num_aspect) * meta_weights[2]

	lookup = a1_lookup

	return lookup


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
	parser.add_argument('--train_uif_path', type=str, default="/raid/brutusxu/agnostic/datasets/electronics/train_uif.pickle")
	parser.add_argument('--use_cuda', type=int, default=0)
	parser.add_argument('--meta_weights', type=str, default='0.4,0.3,0.3')
	parser.add_argument('--k', type=int, default=5)
	
	args = parser.parse_args()
	config = vars(args)

	ranklist = np.load(args.ranklist, allow_pickle=True)
	ui_pairs = []
	for i, row in enumerate(ranklist):
		for j in range(20):
			ui_pairs.append(torch.tensor([row[0], row[1][j]]))

	aspect_lists = []

	model = HardMatchModel(config)
	#model.item_bias = nn.Embedding(args.num_item+1, 1)
	model.load_state_dict(torch.load(args.checkpoint))

	if args.check_wbx_perf:
		print(50*"-", "For debug", 50*"-")
		parser.add_argument('--item_scores_dir', type=str, default="datasets/Electronics_5_5_3/efm_item_scores")
		parser.add_argument('--dataset', type=str, default="Electronics_5_5_3")
		parser.add_argument('--optimizer', type=str, default='adam')
		parser.add_argument('--lr', type=float, default=1e-4)
		parser.add_argument('--l2', type=float, default=0)
		parser.add_argument('--batch_size', type=int, default=64)
		args = parser.parse_args()
		config = vars(args)

		engine = HMEngine(config)
		dataset = AgnosticDataset(args.data_dir, args.dataset, config)
		interact_hist = dataset.interact_hist
		item_pool = set([i for i in range(args.num_item)])
		test_loader = dataset.instance_a_test_loader(args.batch_size, shuffle=False, num_workers=4)
		gt = np.load(os.path.join(args.data_dir, 'test.pickle'), allow_pickle=True)
		
		engine.model = model
		rnk_list = engine.output_ranklist(test_loader, interact_hist, item_pool)
		print("the model true ranklist: ")
		engine.evaluate(rnk_list, gt, 20)
		print("the ranklist read from file: ")
		engine.evaluate(ranklist, gt, 20)

	lookup_table = uia_lookup(model)
	pbar = tqdm(total=len(ui_pairs))
	for j, ui_pair in enumerate(ui_pairs):
		#top_aspect = wbx_topk(model, ui_pair[0], ui_pair[1], args.k)
		top_aspect = lookup_table[ui_pair[0], ui_pair[1]].topk(args.k)[1].tolist()
		aspect_lists.append([ui_pair[0].item(), ui_pair[1].item(), top_aspect])
		pbar.update(1)
	pbar.close()

	if not os.path.isdir(os.path.join(args.data_dir, 'common_aspects')):
		os.mkdir(os.path.join(args.data_dir, 'common_aspects'))

	fn = "_".join(args.checkpoint.split("/")[-2].split("_")[:-1]) + ".pickle"
	np.array(aspect_lists).dump(os.path.join(args.data_dir, 'common_aspects', fn))
