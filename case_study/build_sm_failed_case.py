import json
import numpy as np
import time
import sys
import os
import argparse

from soft_match import *

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

	parser.add_argument('--input_dir', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3')
	parser.add_argument('--checkpoint', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3/best_sm_wbx_kl_checkpoints/best_model.pt')
	parser.add_argument('--num_user', type=int, default=3151)
	parser.add_argument('--num_item', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--num_relation', type=int, default=3)
	parser.add_argument('--use_TransE', type=int, default=0)
	parser.add_argument('--uia_path', type=str, default=' ')

	parser.add_argument('--train_uif_path', type=str, default="/raid/brutusxu/agnostic/datasets/Electronics_5_5_3/train_uif.pickle")

	parser.add_argument('--use_cuda', type=int, default=0)

	parser.add_argument('--meta_weights', type=str, default='0.4,0.3,0.3')

	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--num_failed_case', type=int, default=3)

	args = parser.parse_args()
	config = vars(args)

	model = SimpleModel(config)
	model.load_state_dict(torch.load(args.checkpoint))

	with open(os.path.join(args.input_dir, 'aspect2id.json'), 'r') as f:
		aspect2id = json.load(f)

	id2aspect = {int(v):k for k,v in aspect2id.items()}

	with open(os.path.join(args.input_dir, 'efm_ranklist/ranklist.json'), 'r') as f:
		efm_ranklist  = json.load(f)

	reformed = {}
	for k,v in efm_ranklist.items():
		reformed[int(k)] = []
		for item in v:
			reformed[int(k)].append(int(item))

	efm_ranklist = reformed

	test = np.load(os.path.join(args.input_dir, 'test.pickle'), allow_pickle=True)

	user_aspect_dict = {}
	item_aspect_dict = {}
	sentiment = np.load(os.path.join(args.input_dir, 'sentiment_data.pickle'), allow_pickle=True)

	for i, row in enumerate(sentiment):
		user = row[0]
		item = row[1]

		if user not in user_aspect_dict.keys():
			user_aspect_dict[user]=[]
		if item not in item_aspect_dict.keys():
			item_aspect_dict[item]=[]

		for aos in row[2:]:
			aspect = aos[0]
			if aspect not in user_aspect_dict[user]:
				user_aspect_dict[user].append(aspect)
			if aspect not in item_aspect_dict[item]:
				item_aspect_dict[item].append(aspect)

	user_failed_item_aspects = []
	count = 0
	for i, row in enumerate(test):
		user = row[0]
		items = row[1]
		top_ranklist = efm_ranklist[user][:args.num_failed_case]
		for item in items:
			if item in top_ranklist:
				top_ranklist.remove(item)

		for j in top_ranklist:
			user_indice = torch.tensor([user])
			item_indice = torch.tensor([j])
			aspect_holder = wbx_topk(model, user_indice, item_indice, args.k)
			user_failed_item_aspects.append([user, j, aspect_holder])

	np.array(user_failed_item_aspects).dump('/raid/brutusxu/agnostic/case_study/sm_failed_items_Electronics_efm.pickle')