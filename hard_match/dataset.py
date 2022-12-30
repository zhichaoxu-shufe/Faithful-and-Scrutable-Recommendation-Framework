from functions import *

import argparse
import numpy as np
import torch
import random, sys, time, os
from copy import deepcopy
import json
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class UserDataset(Dataset):
	# wrapper
	def __init__(self, user_tensor):
		self.user_tensor = user_tensor

	def __getitem__(self, index):
		return self.user_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class BundleDataset(Dataset):
	def __init__(self, packed_tensor):
		self.packed_tensor = packed_tensor

	def __getitem__(self, index):
		user_idxes = self.packed_tensor[index][0]
		item_idxes = self.packed_tensor[index][1]
		return user_idxes, item_idxes

	def __len__(self):
		return self.packed_tensor.size(0)

class HardMatchDataset(Dataset):
	def __init__(self, ui_tensor):
		self.ui_tensor = ui_tensor

	def __getitem__(self, index):
		return index, self.ui_tensor[index]

	def __len__(self):
		return self.ui_tensor.size()[0]

class PackedDataset(Dataset):
	# wrapper
	def __init__(self, packed_tensor):
		self.packed_tensor = packed_tensor

	def __getitem__(self, index):
		return self.packed_tensor[index]

	def __len__(self):
		return self.packed_tensor.size(0)

class UserItemScoreDataset(Dataset):
	def __init__(self, user_indices, item_indices, item_scores):
		self.user_indices = torch.LongTensor(user_indices)
		self.item_indices = torch.LongTensor(item_indices)
		self.item_scores = torch.FloatTensor(item_scores)

		# sotfmax item_scores
		self.item_scores = F.softmax(self.item_scores, dim=1)
	def __getitem__(self, i):
		return self.user_indices[i], self.item_indices[i], self.item_scores[i]
	
	def __len__(self):
		return self.user_indices.size(0)


class AgnosticDataset(object):
	def __init__(self, input_dir, dataset, config):
		self.input_dir = input_dir
		self.config = config

		self.trainset = self.build_trainset()
		self.testset = np.load(os.path.join(input_dir, 'test.pickle'), allow_pickle=True)

		self.history_data = np.load(os.path.join(input_dir, 'train.pickle'), allow_pickle=True)
		self.interact_hist = self.build_interaction_history()

	def get_user_item_num(self):
		return self.config['num_user'], self.config['num_item']

	def build_interaction_history(self):
		history = {}
		for i, row in enumerate(self.history_data):
			if row[0] not in history.keys():
				history[row[0]] = set(row[1])
		return history

	def build_trainset(self):
		data = np.load(os.path.join(self.config['item_scores_dir'], 'item_scores.pickle'), allow_pickle=True)
		return data

	def instance_gt_train_loader(self, batch_size, shuffle, num_workers, item_pool):
		interact_hist = self.build_interaction_history()

		packed = []
		for k, v in interact_hist.items():
			for item in v:
				users, items = [], []
				users.append(k)
				items.append(item)
				neg_items = random.sample(item_pool - interact_hist[k], self.config['num_neg'])
				for j in neg_items:
					users.append(k)
					items.append(j)
				packed.append([users, items])
		dataset = PackedDataset(torch.LongTensor(packed))
		return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

	
	def instance_kl_bbx_train_loader(self, batch_size, shuffle, num_workers):
		num_user, num_item = self.get_user_item_num()
		labels = torch.zeros(num_user, num_item).fill_(.01)
		in_path = os.path.join(self.config["data_dir"], "uis_labels.pickle") #[u, i, score]

		if os.path.exists(in_path):
			with open(in_path, "rb") as f:
				labels = pickle.load(f)
		else:
			for (u, items, scores) in self.trainset:
				for (i, s) in zip(items, scores):
					labels[u,i] = s 
			labels = F.softmax(labels, dim=1)
			with open(in_path, "wb") as f:
				pickle.dump(labels, f)
		dataset = HardMatchDataset(labels)
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

			
	def instance_a_test_loader(self, batch_size, shuffle, num_workers):
		users = []
		for i, row in enumerate(self.testset):
			users.append(int(row[0]))
		dataset = UserDataset(user_tensor=torch.LongTensor(users))
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)