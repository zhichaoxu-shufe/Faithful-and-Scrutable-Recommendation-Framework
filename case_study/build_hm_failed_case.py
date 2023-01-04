import json
import numpy as np
import time
import sys
import os
import argparse

from hard_match import *

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

	parser.add_argument('--input_dir', type=str, default='/raid/brutusxu/agnostic/datasets/Office_5_5_3')
	parser.add_argument('--checkpoint', type=str, default='/raid/brutusxu/agnostic/datasets/Office_5_5_3/best_bbx_kl_checkpoints/best_model.pt')
	parser.add_argument('--num_user', type=int, default=1983)
	parser.add_argument('--num_item', type=int, default=957)
	parser.add_argument('--num_aspect', type=int, default=452)
	parser.add_argument('--latent_dim', type=int, default=128)

	parser.add_argument('--train_uif_path', type=str, default="/raid/brutusxu/agnostic/datasets/Office_5_5_3/train_uif.pickle")

	parser.add_argument('--use_cuda', type=int, default=0)

	parser.add_argument('--meta_weights', type=str, default='0.4,0.3,0.3')

	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--num_failed_case', type=int, default=3)

	args = parser.parse_args()
	config = vars(args)

	model = HardMatchModel(config)
	model.load_state_dict(torch.load(args.checkpoint))
	lookup_table = uia_lookup(model)

	with open(os.path.join(args.input_dir, 'aspect2id.json'), 'r') as f:
		aspect2id = json.load(f)

	id2aspect = {int(v):k for k,v in aspect2id.items()}

	with open(os.path.join(args.input_dir, 'efm_ranklist/ranklist.json'), 'r') as f:
		efm_ranklist  = json.load(f)

	reformed = {}
	for k, v in efm_ranklist.items():
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
			# aspect_holder = item_aspect_dict[j]-item_aspect_dict[items[0]]-item_aspect_dict[items[1]]
			aspect_holder = lookup_table[user, j].topk(args.k)[1].tolist()
			user_failed_item_aspects.append([user, j, aspect_holder])

	np.array(user_failed_item_aspects).dump('/raid/brutusxu/agnostic/case_study/failed_items_Office_efm.pickle')