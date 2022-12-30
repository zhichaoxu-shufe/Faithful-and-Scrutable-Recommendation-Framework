
from functions import *

import argparse
import numpy as np
import torch
import random, sys, time, os
from copy import deepcopy
import json
from tqdm import tqdm
import math
from six.moves import xrange
from torch.utils.data import DataLoader, Dataset


class UserDataset(Dataset):
	# wrapper
	def __init__(self, user_tensor):
		self.user_tensor = user_tensor

	def __getitem__(self, index):
		return self.user_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserItemScoreDataset(Dataset):
	def __init__(self, user_indices, item_indices, item_scores):
		self.user_indices = torch.LongTensor(user_indices)
		self.item_indices = torch.LongTensor(item_indices)
		self.item_scores = torch.FloatTensor(item_scores)
	
	def __getitem__(self, i):
		return self.user_indices[i], self.item_indices[i], self.item_scores[i]
	
	def __len__(self):
		return self.user_indices.size(0)

class UserItemDataset(Dataset):
	# wrapper, convert <user, item> tensor into pytorch dataset
	def __init__(self, user_tensor, item_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class PackedDataset(Dataset):
	# wrapper, 
	def __init__(self, packed_tensor):
		self.packed_tensor = packed_tensor
	
	def __getitem__(self, index):
		return self.packed_tensor[index]
	
	def __len__(self):
		return self.packed_tensor.size(0)

class AgnosticDataset(object):
	def __init__(self, input_dir, dataset, config):
		self.input_dir = input_dir
		self.config = config

		self.trainset = self.build_trainset()
		self.testset = np.load(os.path.join(input_dir, 'test.pickle'), allow_pickle=True)
		self.history_data = np.load(os.path.join(input_dir, 'train.pickle'), allow_pickle=True)
		self.sentiment = np.load(os.path.join(input_dir, 'sentiment_data.pickle'), allow_pickle=True)

		ratings = np.load(os.path.join(input_dir, 'user_item_rating.pickle'), allow_pickle=True)
		self.ratings = {}
		for i in range(ratings.shape[0]):
			self.ratings[(ratings[i][0], ratings[i][1])] = ratings[i][2]

		self.interact_hist = self.build_interaction_history() # get self.history

		self.sentiment = self.sentiment.tolist()
		sentiment_dict = {}
		for i, row in enumerate(self.sentiment):
			sentiment_dict[(row[0], row[1])] = row[2:]
		self.sentiment = sentiment_dict

	def get_user_item_num(self):
		return self.config['num_user'], self.config['num_item']

	def build_trainset(self):
		data = np.load(os.path.join(self.config['item_scores_dir'], 'item_scores.pickle'), allow_pickle=True)
		return data 

	def build_interaction_history(self):
		history = {}
		for i, row in enumerate(self.history_data):
			if row[0] not in history.keys():
				history[row[0]] = set(row[1])
		return history

	def instance_gt_train_loader(self, batch_size, shuffle, num_workers, item_pool):
		interact_hist = self.build_interaction_history()
		packed = []
		for k, v in interact_hist.items():
			for item in v:
				users, items = [], []
				users.append(k)
				items.append(item)
				neg_items = random.sample(item_pool-interact_hist[k], self.config['num_neg'])
				for j in neg_items:
					users.append(k)
					items.append(j)

				packed.append([users, items])
		
		packed_tensor = torch.LongTensor(packed)
		dataset=PackedDataset(packed_tensor)

		return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

	def instance_kl_train_loader(self, batch_size, shuffle, num_workers):
		user_indices, item_indices, item_scores = zip(*self.trainset) #[user_num], [user_num, item_num], [user_num, item_num]
		dataset = UserItemScoreDataset(user_indices, item_indices, item_scores)

		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	
	def instance_a_test_loader(self, batch_size, shuffle, num_workers):
		users = []
		for i, row in enumerate(self.testset):
			users.append(int(row[0]))
		dataset = UserDataset(user_tensor=torch.LongTensor(users))
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


		


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", type=str, required=True)
	parser.add_argument("--dataset", type=str, required=True)

	args = parser.parse_args()

	#build_matrix(args.dataset, args.dest_dir)
	# trainset = np.load(args.dest_dir+'/'+args.dataset+'_train.pickle', allow_pickle=True)
	
	"""
	dataset = EFMDataset(args.input_dir, args.dataset)
	dataset.build_interaction_history()
	train_loader = dataset.instance_a_train_loader(32)
	for batch_id, batch in enumerate(train_loader):
		print(batch[0], batch[1], batch[2])
		print("-"*100)
		print(batch[0].shape, batch[1].shape, batch[2].shape)
		exit()
		continue
	"""

	dataset = AgnosticDataset(args.input_dir, args.dataset)
	train_loader = dataset.instance_a_train_loader(32, True, 1)

	print("test train_loader ...")
	for i, (user_indices, item_indices, item_scores) in enumerate(train_loader):
		print(user_indices, item_indices, item_scores)
		print("shape: ", user_indices.shape, item_indices.shape, item_scores.shape)
		break 

	print("-"*100)
	print("test test_loader ...")
	test_loader = dataset.instance_a_test_loader(32, False, 1)
	for i, user_indices in enumerate(test_loader):
		print(user_indices)
		print("shape: ", user_indices)
		break

	
