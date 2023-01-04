import json
import numpy as np
import time
import sys
import os
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str)
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--num_failed_case', type=int, default=3)
	parser.add_argument('--ranklist_type', type=str, default='json')

	args = parser.parse_args()

	a2cf_ranklist_path = os.path.join(args.input_dir, 'efm_ranklist/ranklist.json')
	modified_ranklist_path = os.path.join(args.input_dir, 'efm_ranklist/scrutable_ranklist.json')
	trainset_path = os.path.join(args.input_dir, 'train.pickle')
	testset_path = os.path.join(args.input_dir, 'test.pickle')
	aspect_path = os.path.join(args.input_dir, 'aspect2id.json')

	with open(aspect_path, 'r') as f:
		aspect2id = json.load(f)

	id2aspect = {int(v):k for k, v in aspect2id.items()}

	with open(a2cf_ranklist_path, 'r') as f:
		a2cf_ranklist = json.load(f)

	reformed = {}
	for k, v in a2cf_ranklist.items():
		reformed[int(k)] = []
		for item in v:
			reformed[int(k)].append(int(item))

	a2cf_ranklist = reformed

	if args.ranklist_type == 'json':

		with open(modified_ranklist_path, 'r') as f:
			modified_a2cf_ranklist = json.load(f)

		reformed = {}
		for k, v in modified_a2cf_ranklist.items():
			reformed[int(k)] = []
			for item in v:
				reformed[int(k)].append(int(item))

		modified_a2cf_ranklist = reformed

	elif args.ranklist_type == 'np':
		modified_a2cf_ranklist = np.load(modified_ranklist_path, allow_pickle=True)
		reformed = {}
		for i, row in enumerate(modified_a2cf_ranklist):
			reformed[row[0]] = row[1]
		modified_a2cf_ranklist = reformed

	trainset = np.load(trainset_path, allow_pickle=True)
	for i, row in enumerate(trainset):
		user = row[0]
		items = row[1]
		for item in items:
			if item in modified_a2cf_ranklist[user]:
				modified_a2cf_ranklist[user].remove(item)

	testset = np.load(testset_path, allow_pickle=True)
	testset_dict = {}
	for i, row in enumerate(testset):
		testset_dict[row[0]] = row[1]

	user_item_comparison_failed_case = {}
	user_item_comparison_gt = {}
	for k, v in a2cf_ranklist.items():
		user = k
		top_items = v[:args.num_failed_case]

		for item in testset_dict[user]:
			if item in top_items:
				top_items.remove(item)
		user_item_comparison_failed_case[user] = []
		user_item_comparison_gt[user] = []
		for i, item in enumerate(top_items):
			new_index = modified_a2cf_ranklist[user].index(item)
			user_item_comparison_failed_case[user].append(a2cf_ranklist[user].index(item))
			user_item_comparison_failed_case[user].append(new_index)

		for i, item in enumerate(testset_dict[user]):
			user_item_comparison_gt[user].append(a2cf_ranklist[user].index(item))
			user_item_comparison_gt[user].append(modified_a2cf_ranklist[user].index(item))

	rank_count_holder = []
	for k, v in user_item_comparison_failed_case.items():
		for i in range(len(v)//2):
			rank_count_holder.append(v[i*2+1]-v[i*2])

	print('Avg pos change of failed case: ', sum(rank_count_holder)/len(rank_count_holder))

	rank_count_holder = []
	for i, row in enumerate(testset):
		user = row[0]
		items = row[1]
		for item in items:
			if item not in a2cf_ranklist[user][:args.num_failed_case]:
				rank_count_holder.append(a2cf_ranklist[user].index(item)-modified_a2cf_ranklist[user].index(item))

	print('Avg pos changed of gt case: ', sum(rank_count_holder)/len(rank_count_holder))
	print(min(rank_count_holder), max(rank_count_holder))
