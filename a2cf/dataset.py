from functions import *

import argparse
import numpy as np
import random, sys, time, json
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

class A2CFTrainingDataset(Dataset):
	def __init__(self, tuple_embeddings):
		self.tuple_embeddings = tuple_embeddings

	def __getitem__(self, index):
		return self.tuple_embeddings[index, 0], self.tuple_embeddings[index, 1]

	def __len__(self):
		return len(self.tuple_embeddings)

class A2CFFillMatrixDataset(Dataset):
	def __init__(self, data_pairs):
		self.data_pairs = data_pairs

	def __getitem__(self, index):
		return self.data_pairs[index]

	def __len__(self):
		return len(self.data_pairs)

class UserDataset(Dataset):
	# wrapper
	def __init__(self, user_tensor):
		self.user_tensor = user_tensor

	def __getitem__(self, index):
		return self.user_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserItemDataset(Dataset):
	# wrapper
	def __init__(self, user_tensor, item_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
	
	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserPosNegDataset(Dataset):
	# wrapper
	def __init__(self, user_tensor, pos_item_tensor, neg_item_tensor):
		self.user_tensor = user_tensor
		self.pos_item_tensor = pos_item_tensor
		self.neg_item_tensor = neg_item_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.pos_item_tensor[index], self.neg_item_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserItemScoreDataset(Dataset):
	# wrapper, convert <user, item> tensor into pytorch dataset
	def __init__(self, user_tensor, item_tensor, score_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.score_tensor = score_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index], self.score_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class PackedDataset(Dataset):
	# wrapper
	def __init__(self, packed_tensor):
		self.packed_tensor = packed_tensor

	def __getitem__(self, index):
		return self.packed_tensor[index]

	def __len__(self):
		return self.packed_tensor.size(0)


class A2CFDataset(object):
	def __init__(self, input_dir, dataset):
		trainset = np.load(os.path.join(input_dir, 'train.pickle'), allow_pickle=True)
		self.packed_trainset = trainset
		self.build_interaction_history()

		self.testset = np.load(os.path.join(input_dir, 'test.pickle'), allow_pickle=True)

		# unpack trainset
		unpacked = []
		for i, row in enumerate(trainset):
			user, items = row[0], row[1]
			for item in items:
				unpacked.append([user, item])
		self.trainset = unpacked


	def build_interaction_history(self):
		self.history = {}
		for i, row in enumerate(self.packed_trainset):
			self.history[row[0]] = set(row[1])

	# non batch implementation
	def instance_pair_train_loader(self, batch_size, shuffle, num_workers, num_negatives, item_pool):
		users, pos_items, neg_items = [], [], []
		for i, row in enumerate(self.trainset):
			user, item = row[0], row[1]
			negatives = list(random.sample(item_pool-self.history[user], num_workers))
			for neg in negatives:
				users.append(user)
				pos_items.append(item)
				neg_items.append(neg)
		dataset = UserPosNegDataset(torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items))
		return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

	# batch implementation
	def instance_pack_train_loader(self, batch_size, shuffle, num_workers, num_negatives, item_pool):
		packed = []
		for i, row in enumerate(self.trainset):
			users, items = [row[0]], [row[1]]
			negatives = list(random.sample(item_pool-self.history[row[0]], num_negatives))
			for negative in negatives:
				users.append(row[0])
				items.append(negative)

			packed.append([users, items])

		packed_tensor = torch.LongTensor(packed)
		dataset = PackedDataset(packed_tensor)

		return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

	def instance_a_test_loader(self, batch_size, shuffle, num_workers):
		users = []
		for i, row in enumerate(self.testset):
			users.append(int(row[0]))
		dataset = UserDataset(user_tensor=torch.LongTensor(users))
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--num_user', type=int, default=3151)
	parser.add_argument('--num_item', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)

	args = parser.parse_args()

	sentiment_path = os.path.join(args.input_dir, 'num_sentiment_data.pickle')
	sentiment = np.load(sentiment_path, allow_pickle=True)
