from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import random, json, sys, time, argparse
import copy

class UserTensor(Dataset):
	# wrapper
	def __init__(self, user_tensor):
		self.head_tensor = user_tensor

	def __getitem__(self, index):
		return self.user_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserAspectTensor(Dataset):
	def __init__(self, user_tensor, relation_tensor, aspect_tensor):
		self.user_tensor = user_tensor
		self.relation_tensor = relation_tensor
		self.aspect_tensor = aspect_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.relation_tensor[index], self.aspect_tensor[index]

	def __len__(self, index):
		return self.user_tensor.size(0)

class HRTTensor(Dataset):
	def __init__(self, heads, relations, tails):
		self.heads = heads
		self.relations = relations
		self.tails = tails

	def __getitem__(self, index):
		return self.heads[index], self.relations[index], self.tails[index]

	def __len__(self, index):
		return self.user_tensor.size(0)

# class SampleGenerator(object):
# 	def __init__(self, train_triples, testset, config):
# 		# triples: [head, tail, relation]
# 		self.len = len(triples)
# 		self.train_triples = train_triples
# 		self.train_triple_set = set(train_triples)
# 		self.testset = testset # list

# 		self.config = config
# 		self.num_user = config['num_user']
# 		self.num_item = config['num_item']
# 		self.num_aspect = config['num_aspect']

# 		self.num_neg = config['num_neg']

# 		self.user_set = set([i for i in range(self.num_user)])
# 		self.item_set = set([i for i in range(self.num_item)])
# 		self.aspect_set = set([i for i in range(self.num_aspect)])

# 	@staticmethod
# 	def build_history(self):
# 		self.user_hist = {}
# 		self.item_hist = {}
# 		for i in self.triples:
# 			if i[-1] == 0:
# 				if i[0] not in self.user_hist.keys():
# 					self.user_hist[i[0]] = []
# 				self.user_hist[i[0]].append(i[1])
# 			else:
# 				if i[0] not in self.item_hist.keys():
# 					self.item_hist[i[0]] = []
# 				self.item_hist[i[0]].append(i[1])


# 	def __getitem__(self, idx):
# 		pass

# 	def __len__(self):
# 		return self.len

# 	def user_neg_sample(self, head, num_neg):
# 		return list(random.sample(self.aspect_set - set(self.user_hist[head]), num_neg))

# 	def item_neg_sample(self, head, num_neg):
# 		return list(random.sample(self.aspect_set - set(self.item_hist[head])))

# 	def instance_a_train_loader(self, batch_size):
# 		heads, relations, tails = [], [], [] 
# 		for i, row in enumerate(self.train_triples):
# 			if row[-1] in set([3, 4, 5]):
# 				continue
# 			else:
# 				heads.append(int(row[0]))
# 				tails.append(int(row[1]))
# 				relations.append(int(row[2]))
# 				if int(row[2]) == 0:
# 					negative_samples = list(random.sample(self.aspect_set-self.user_hist[int(row[0])], self.num_neg))
# 					for aspect in negative_samples:
# 						heads.append(int(row[0]))
# 						tails.append(int(aspect))
# 						relations.append(0)
# 				else:
# 					negative_samples = list(random.sample(self.aspect_set-self.item_hist[int(row[0])], self.num_neg))
# 					for aspect in negative_samples:
# 						heads.append(int(row[0]))
# 						tails.append(int(aspect))
# 						relations.append(int(row[2]))
# 		dataset = HRTTensor(heads=torch.LongTensor(heads),
# 							relations=torch.LongTensor(relations),
# 							tails=torch.LongTensor(tails),
# 							)
# 		return dataset

# 	def instance_a_test_loader(self):
# 		heads, relations, tails = [], [], []
# 		for i, row in enumerate(self.testset):
# 			if row[1] in [0, 3]:
# 				# user
# 				for j in row[2]:
# 					heads.append(row[0])
# 					relations.append(row[1])

class KGETrainDataset(Dataset):
	def __init__(self, triples, aspect_pool, config):
		self.triples = triples
		self.config = config
		self.num_user = config['num_user']
		self.num_item = config['num_item']
		self.num_neg = config['num_neg']

		self.user_hist, self.item_hist = self.build_history()
		self.aspect_pool = set(aspect_pool)

	def __len__(self):
		return len(self.triples)

	def build_history(self):
		user_hist = {}
		item_hist = {}
		for i in self.triples:
			if i[-1] == 0:
				if i[0] not in user_hist.keys():
					user_hist[i[0]] = []
				user_hist[i[0]].append(i[1])
			else:
				if i[0] not in item_hist.keys():
					item_hist[i[0]] = []
				item_hist[i[0]].append(i[1])
		return user_hist, item_hist

	def sample_neg(self, is_user, idx):
		if is_user:
			return list(random.sample(self.aspect_pool-set(self.user_hist[idx]), self.num_neg))
		else:
			return list(random.sample(self.aspect_pool-set(self.item_hist[idx]), self.num_neg))

	def __getitem__(self, idx):
		row = self.triples[idx]
		head = row[0]
		tail = row[1]
		relation = row[2]
		neg_relation = copy.deepcopy(relation)
		if relation == 0:
			neg_tails = self.sample_neg(1, head)
		else:
			neg_tails = self.sample_neg(0, head)
		neg_tails = torch.LongTensor(neg_tails)
		neg_relations = torch.LongTensor([neg_relation]*neg_tails.shape[0])
		neg_heads = torch.LongTensor([head]*neg_tails.shape[0])
		return head, relation, tail, neg_heads, neg_relations, neg_tails

class KGETestDataset(Dataset):
	def __init__(self, testset, aspect_pool, config):
		# kge test triples [[head, relation, [candidates tails, gt tail]]]
		self.testset = testset
		self.config = config
		self.num_user = config['num_user']
		self.num_item = config['num_item']
		self.num_aspect = config['num_aspect']

	def __len__(self):
		return len(self.testset)

	def __getitem__(self, idx):
		row = self.testset[idx]
		head, relation, tails = row[0], row[1], row[2]
		# print(len(tails))
		heads = [head]*len(tails)
		relations = [relation]*len(tails)
		return torch.LongTensor(heads).squeeze(), torch.LongTensor(relations).squeeze(), torch.LongTensor(tails).squeeze()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_dir', type=str, default='datasets/Electronics/')
	parser.add_argument('--num_user', type=int, default=2911)
	parser.add_argument('--num_item', type=int, default=20229)
	parser.add_argument('--num_aspect', type=int, default=210)
	parser.add_argument('--num_neg', type=int, default=4)

	args = parser.parse_args()
	config = vars(args)

	with open(args.input_dir+'aspect2id.json', 'r') as f:
		aspect2id = json.load(f) 
	# id2aspect = {v: k for k, v in aspect2id.items()}

	with open(args.input_dir+'entity2id.json', 'r') as f:
		entity2id = json.load(f)
	# id2entity = {v: k for k, v in entity2id.items()}

	aspect_pool = []
	for k, v in aspect2id.items():
		aspect_pool.append(entity2id[k])

	trainset = np.load(args.input_dir+'train_triples.pickle', allow_pickle=True)
	trainset = KGETrainDataset(trainset, aspect_pool, config)
	trainLoader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=1)
	for i, batch in enumerate(trainLoader):
		print(batch)
		print(batch[0].shape)
		print(batch[4].shape)
		sys.exit()