from functions import *
from dataset import *
from model import *
from evaluate import *
from engine import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, os, json, copy, logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

def compute_vp(X_user, Y_item, F, epsilon):
	X_user, Y_item = X_user.unsqueeze(dim=0), Y_item.unsqueeze(dim=0)

	softmax = torch.nn.Softmax()
	ln = softmax(torch.mul(X_user, Y_item)/epsilon)
	# print(ln.shape)
	# sys.exit()
	vp = torch.mm(ln, F)
	return vp.squeeze()

# def a2cf_topk(X, Y, U, I, F, model, u, i, epsilon, K):
# 	X.requires_grad = True
# 	Y.requires_grad = True
# 	U.requires_grad = True
# 	I.requires_grad = True
# 	F.requires_grad = True

# 	u_tensor, i_tensor = torch.tensor(u), torch.tensor(i)
	
# 	u_embed, i_embed = X[u], Y[i]
# 	ui_product = torch.mul(U[u], I[i]).squeeze()
# 	# print(U[u].shape, I[i].shape, F.shape)
# 	vp = compute_vp(X[u], Y[i], F, epsilon)
# 	cat = torch.cat((ui_product, vp), dim=0)
# 	pred = model.forward(cat)

# 	pred.backward()
# 	aspect_grad = F.grad.sum(axis=1)
# 	aspect_indices = torch.topk(aspect_grad, K)[1]

# 	return aspect_indices.tolist()

def a2cf_topk(X, Y, U, I, F, model, u, i, epsilon, K):
	ln = torch.mul(X[u], Y[i])
	aspect_indices = torch.topk(ln, K)[1]
	return aspect_indices.tolist()

def a2cf_2way_topk(X, Y, U, I, F, model, u, i, epsilon, K):

	ln = torch.mul(X[u], Y[i])
	aspect_indices = torch.topk(ln, K)[1]
	return aspect_indices.tolist()






if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--ranklist', type=str, required=True)
	parser.add_argument('--checkpoint', type=str, required=True)
	parser.add_argument('--input_dir', type=str, required=True)

	parser.add_argument('--num_user', type=int, default=3151)
	parser.add_argument('--num_item', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)
	parser.add_argument('--latent_dim', type=int, default=128)

	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--epsilon', type=float, default=8.)
	parser.add_argument('--alpha', type=float, default=0.5)

	args = parser.parse_args()
	config = vars(args)

	with open(args.ranklist, 'r') as f:
		ranklist = json.load(f)

	ui_pairs = []
	for k, v in ranklist.items():
		for i in range(20):
			ui_pairs.append(torch.tensor([int(k), int(v[i])]))

	aspect_lists = []
	# model = TwoWayNet(args.latent_dim, args.alpha)
	model = RankingScoreNet(args.latent_dim)
	model.load_state_dict(torch.load(args.checkpoint))

	user_matrix_path = os.path.join(config['input_dir'], 'a2cf_user_matrix_pretrained.pickle')
	item_matrix_path = os.path.join(config['input_dir'], 'a2cf_item_matrix_pretrained.pickle')
	user_embed = np.load(user_matrix_path, allow_pickle=True)
	item_embed = np.load(item_matrix_path, allow_pickle=True)

	X, Y = torch.from_numpy(user_embed), torch.from_numpy(item_embed)

	embedding_model = EmbeddingNet(X, Y, X.shape[0], Y.shape[0], X.shape[1], config['latent_dim'])

	embed_model_path = os.path.join(config['input_dir'], 'a2cf_pretrain/pretrain.pt')
	embedding_model.load_state_dict(torch.load(embed_model_path))

	U = embedding_model.state_dict()['U'].detach().float()
	I = embedding_model.state_dict()['I'].detach().float()
	F = embedding_model.state_dict()['F'].detach().float()

	aspect_lists = []

	pbar = tqdm(total=len(ui_pairs))
	for ui_pair in ui_pairs:
		top_aspect = a2cf_topk(X, Y, U, I, F, model, ui_pair[0], ui_pair[1], args.epsilon, args.k)
		aspect_lists.append([ui_pair[0].item(), ui_pair[1].item(), top_aspect])
		pbar.update(1)
	pbar.close()

	if not os.path.isdir(os.path.join(args.input_dir, 'common_aspects')):
		os.mkdir(os.path.join(args.input_dir, 'common_aspects'))
	np.array(aspect_lists).dump(os.path.join(args.input_dir, 'common_aspects/a2cf_top_aspects.pickle'))