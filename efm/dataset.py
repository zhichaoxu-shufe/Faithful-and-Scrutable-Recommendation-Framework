
from functions import *
import argparse

import numpy as np
import torch
import random, sys, time
from copy import deepcopy
import json
from tqdm import tqdm
import math
import os
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


class TrainDataset(Dataset):
	# wrapper
	def __init__(self, user_tensor, item_tensor, aspect_tensor, user_matrix, item_matrix, rating_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.aspect_tensor = aspect_tensor
		self.user_matrix = user_matrix
		self.item_matrix = item_matrix
		self.rating_tensor = rating_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index], self.aspect_tensor[index], \
		self.user_matrix[index], self.item_matrix[index], self.rating_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserItemDataset(Dataset):
	# wrapper, convert <user, item> tensor into pytorch dataset
	def __init__(self, user_tensor, item_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserItemRatingDataset(Dataset):
	# wrapper, convert <user, item, rating> tensor into pytorch dataset
	def __init__(self, user_tensor, item_tensor, target_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.target_tensor = target_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)


class EFMDataset(object):
	def __init__(self, input_dir, dataset):

		self.trainset = np.load(os.path.join(input_dir, 'train.pickle'), allow_pickle=True)
		self.testset = np.load(os.path.join(input_dir, 'test.pickle'), allow_pickle=True)
		self.sentiment = np.load(os.path.join(input_dir, 'sentiment_data.pickle'), allow_pickle=True)
		ratings = np.load(os.path.join(input_dir, 'user_item_rating.pickle'), allow_pickle=True)
		self.ratings = {}
		for i in range(ratings.shape[0]):
			self.ratings[(ratings[i][0], ratings[i][1])] = ratings[i][2]

		num_user = 0
		num_item = 0
		for i, row in enumerate(self.trainset):
			if row[0] > num_user:
				num_user = row[0]
			for item in row[1]:
				if item > num_item:
					num_item = item
		for i, row in enumerate(self.testset):
			for item in row[1]:
				if item > num_item:
					num_item = item
		self.num_user = num_user
		self.num_item = num_item

		self.user_aspect_matrix = np.load(os.path.join(input_dir, 'user_attention_matrix.pickle'), allow_pickle=True)
		self.item_aspect_matrix = np.load(os.path.join(input_dir, 'item_quality_matrix.pickle'), allow_pickle=True)
		self.user_aspect_matrix = torch.tensor(self.user_aspect_matrix)
		self.item_aspect_matrix = torch.tensor(self.item_aspect_matrix)

		self.sentiment = self.sentiment.tolist()
		sentiment_dict = {}
		for i, row in enumerate(self.sentiment):
			sentiment_dict[(row[0], row[1])] = row[2:]
		self.sentiment = sentiment_dict

		self.build_interaction_history()

	def build_interaction_history(self):
		
		self.history = {}
		for i, row in enumerate(self.trainset):
			if row[0] not in self.history.keys():
				self.history[row[0]] = set(row[1])

	def instance_a_train_loader(self, batch_size, shuffle, num_workers):
		users, items, aspects, user_matrix, item_matrix, ratings = [], [], [], [], [], []


		for i, row in enumerate(self.trainset):
			for item in row[1]:
				for k in self.sentiment[(row[0],item)]:
					users.append(int(row[0]))
					items.append(int(item))
					aspects.append(int(k[0]))
					user_matrix.append(float(self.user_aspect_matrix[row[0]][k[0]]))
					item_matrix.append(float(self.item_aspect_matrix[item][k[0]]))
					ratings.append(float(self.ratings[(row[0], item)]))
		dataset = TrainDataset(user_tensor=torch.LongTensor(users), 
								item_tensor=torch.LongTensor(items),
								aspect_tensor=torch.LongTensor(aspects),
								user_matrix=torch.FloatTensor(user_matrix),
								item_matrix=torch.FloatTensor(item_matrix),
								rating_tensor=torch.FloatTensor(ratings) 
								)

		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


	def instance_a_test_loader(self, batch_size, shuffle, num_workers):
		users, items = [], []
		# for i, row in enumerate(self.testset):
		# 	for item in row[1]:
		# 		users.append(int(row[0]))
		for i, row in enumerate(self.testset):
			users.append(int(row[0]))
		dataset = UserDataset(user_tensor=torch.LongTensor(users))
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--dest_dir", type=str, required=True)

	args = parser.parse_args()

	build_matrix(args.dataset, args.dest_dir)
	# trainset = np.load(args.dest_dir+'/'+args.dataset+'_train.pickle', allow_pickle=True)
	# print(type(trainset))
	# dataset = EFMDataset(args.dest_dir, args.dataset)
	# dataset.build_interaction_history()
	# train_loader = dataset.instance_a_train_loader(32)
	# for batch_id, batch in enumerate(train_loader):
	# 	continue
