import json, sys, time
import numpy as np
from scipy import stats
import random
import argparse



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--bbx_item_scores', type=str, required=True)
	parser.add_argument('--wbx_item_scores', type=str, required=True)
	parser.add_argument('--gt', type=str, required=True)
	parser.add_argument('--num_user', type=int)
	parser.add_argument('--num_item', type=int)

	args = parser.parse_args()
	config = vars(args)
	with open(args.bbx_item_scores, 'r') as f:
		bb_ranklist = json.load(f)

	with open(args.wbx_item_scores, 'r') as f:
		white_ranklist = json.load(f)

	gt = np.load(args.gt, allow_pickle=True)

	gt_dict = {}
	for i, row in enumerate(gt):
		gt_dict[row[0]]=row[1]

	num_users = args.num_user
	num_items = args.num_item
	item_pool = [i for i in range(num_items)]
	spearman = []
	kendall_tau = []
	for i in range(num_users):
		user = str(i)
		bb_ranklist_single, bb_scores = bb_ranklist[user][0], bb_ranklist[user][1]
		white_ranklist_single, white_scores = white_ranklist[user][0], white_ranklist[user][1]

		bb_scores_sample, white_scores_sample = [], []
		neg_items = random.sample(set(item_pool)-set(gt_dict[i]), 300)
		for item in gt_dict[i]:
			bb_scores_sample.append(bb_scores[item])
			white_scores_sample.append(white_scores[item])
		for item in neg_items:
			bb_scores_sample.append(bb_scores[item])
			white_scores_sample.append(white_scores[item])

		spearman.append(stats.spearmanr(bb_scores_sample, white_scores_sample)[0])
		kendall_tau.append(stats.kendalltau(bb_scores_sample, white_scores_sample)[0])

	print('spearman correlation: ', sum(spearman)/len(spearman))
	print('kendall tau correlation: ', sum(kendall_tau)/len(kendall_tau))