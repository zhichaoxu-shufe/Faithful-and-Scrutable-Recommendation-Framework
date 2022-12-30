import json, sys, time
import numpy as np
from scipy import stats
import random

bbx_path = 'datasets/Electronics_5_5_3/efm_item_scores/item_scores.json'
with open(bbx_path, 'r') as f:
	bb_ranklist = json.load(f)

#wbx_path = 'datasets/Electronics_5_5_3/bbx_kl_item_scores/wbx_item_scores.json'
wbx_path = 'datasets/Electronics_5_5_3/gt_softmax_item_scores/best_bbx_kl.json'
with open(wbx_path, 'r') as f:
	white_ranklist = json.load(f)

gt = np.load('datasets/Electronics_5_5_3/test.pickle', allow_pickle=True)

gt_dict = {}
for i, row in enumerate(gt):
	gt_dict[row[0]]=row[1]

num_users = 3151
num_items = 3253
item_pool = [i for i in range(num_items)]
spearman = []
kendall_tau = []
for i in range(num_users):
	user = str(i)
	bb_ranklist_single, bb_scores = bb_ranklist[user][0], bb_ranklist[user][1]
	white_ranklist_single, white_scores = white_ranklist[user][0], white_ranklist[user][1]

	bb_scores_sample, white_scores_sample = [], []
	neg_items = random.sample(set(item_pool)-set(gt_dict[i]), 198)
	for item in gt_dict[i]:
		bb_scores_sample.append(bb_scores[item])
		white_scores_sample.append(white_scores[item])
	for item in neg_items:
		bb_scores_sample.append(bb_scores[item])
		white_scores_sample.append(white_scores[item])

	spearman.append(stats.spearmanr(bb_scores_sample, white_scores_sample)[0])
	kendall_tau.append(stats.kendalltau(bb_scores_sample, white_scores_sample)[0])

print(sum(spearman)/len(spearman))
print(sum(kendall_tau)/len(kendall_tau))