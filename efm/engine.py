from functions import *
from dataset import *
from model import *
from evaluate import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, math, os, json, copy, logging
from tqdm import tqdm

import torch
from six.moves import xrange
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_

def save_checkpoint(epoch, model, dataset, output_dir):
	if not os.path.isdir(os.path.join(output_dir, 'efm_checkpoints')):
		os.mkdir(os.path.join(output_dir, 'efm_checkpoints'))
	torch.save(model.state_dict(), os.path.join(output_dir, 'efm_checkpoints', 'efm_epoch_{}.pt'.format(epoch)))

def resume_checkpoint(path, model):
	return model.load_state_dict(torch.load(path))

def init_weights(m):
	if type(m) == torch.nn.Embedding:
		torch.nn.init.normal(m.weight, 0.0, 0.01)

def use_optimizer(network, params):
	if params['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(network.parameters(),
			lr = params['lr'],
			momentum = params['momentum'],
			weight_decay = params['l2']
			)
	if params['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(),
			lr = params['lr'],
			weight_decay = params['l2']
			)
	return optimizer

def save_scrutable_ranklist(ranklist, dest_dir):
	ranklist2save = []
	for k,v in ranklist.items():
		ranklist2save.append([k,v])
	print(ranklist2save.shape)
	sys.exit()
	np.array(ranklist2save).dump(os.path.join(dest_dir,'scrutable_ranklist.pickle'))


class Engine(object):
	# meta engine for training efm model
	def __init__(self, config):
		self.config = config
		# self.crit = torch.nn.BCEWithLogitsLoss()
		self.crit = torch.nn.MSELoss()
		self.opt = use_optimizer(self.model, config)

	def train_an_epoch(self, train_loader, epoch_id):
		if self.config['use_cuda']:
			self.model.cuda()
		self.model.train()
		total_loss = 0

		pbar = tqdm(total=len(train_loader))
		for batch_id, batch in enumerate(train_loader):
			assert isinstance(batch[0], torch.LongTensor)
			users, items, aspects, user_matrix, item_matrix, ratings = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
			ratings = ratings.float()
			loss = self.train_single_batch(users, items, aspects, user_matrix, item_matrix, ratings)
			total_loss += loss
			pbar.update(1)
		pbar.close()

		print('Epoch average loss: ', total_loss/len(train_loader))
		return total_loss/len(train_loader)

	def train_single_batch(self, users, items, aspects, user_matrix, item_matrix, ratings):
		if self.config['use_cuda'] is True:
			users, items, aspects, user_matrix, item_matrix, ratings = users.cuda(), items.cuda(), aspects.cuda(), user_matrix.cuda(), item_matrix.cuda(), ratings.cuda()
		self.opt.zero_grad()
		ratings_pred, user_matrix_pred, item_matrix_pred = self.model(users, items, aspects)
		loss = self.crit(ratings_pred.view(-1), ratings)+self.config['lambda_x']*self.crit(user_matrix_pred.view(-1), user_matrix)+self.config['lambda_y']*self.crit(item_matrix_pred.view(-1), item_matrix)
		loss.backward()
		gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
		self.opt.step()
		loss = loss.item()
		return loss

	def evaluate(self, ranklist, gt, epoch_id, cut_off=20):
		
		gt = gt.tolist()
		qrel_map = {}
		for i, row in enumerate(gt):
			qrel_map[row[0]] = row[1]
		reformed = {}
		for key in qrel_map.keys():
			key = int(key)
			reformed[str(key)] = []
			for value in qrel_map[key]:
				value = int(value)
				reformed[str(key)].append(str(value))
		qrel_map = reformed
		recall, precision, ndcg, hit, reciprocal_rank = print_metrics_with_rank_cutoff(ranklist, qrel_map, cut_off)

		return recall, precision, ndcg, hit, reciprocal_rank


	def output_ranklist(self, test_loader, interact_hist, item_pool):
		self.model.cpu()
		self.model.eval()
		# print('output ranklist')
		test_items = list(item_pool)

		with torch.no_grad():
			full_ranklist = {}
			user_rating_embedding = self.model.embedding_user.weight
			item_rating_embedding = self.model.embedding_item.weight
			user_aspect_embedding = self.model.embedding_user_2.weight
			item_aspect_embedding = self.model.embedding_item_2.weight
			aspect_embedding = self.model.embedding_aspect.weight

			# print(user_embedding.shape, item_embedding.shape)
			rating_matrix = torch.mm(user_rating_embedding, torch.transpose(item_rating_embedding, 0, 1))+\
							torch.mm(user_aspect_embedding, torch.transpose(item_aspect_embedding, 0, 1))

			user_aspect_matrix = torch.mm(user_aspect_embedding, aspect_embedding)

			item_aspect_matrix = torch.mm(item_aspect_embedding, aspect_embedding)


			for i, batch in enumerate(test_loader):
				for j in range(batch.shape[0]):
					test_user_single = batch[j].item()
					item_scores = []
					sorted_items = []
					for k in test_items:
						# print((1-self.config['alpha'])*rating_matrix[test_user_single][k].item(), self.config['alpha']*torch.dot(user_aspect_matrix[test_user_single][:], item_aspect_matrix[k][:]).item())
						# time.sleep(0.5)
						item_scores.append((1-self.config['alpha'])*rating_matrix[test_user_single][k].item()+self.config['alpha']*torch.dot(user_aspect_matrix[test_user_single][:], item_aspect_matrix[k][:]).item())
					sorted_scores_index=sorted(range(len(item_scores)), key=lambda k:item_scores[k], reverse=True)[:1000]
					for k in xrange(len(sorted_scores_index)):
						sorted_items.append(str(test_items[sorted_scores_index[k]]))
					full_ranklist[str(test_user_single)]=sorted_items

			if not os.path.isdir(os.path.join(self.config['dest_dir'], 'efm_ranklist')):
				os.mkdir(os.path.join(self.config['dest_dir'], 'efm_ranklist'))

			with open(os.path.join(self.config['dest_dir'], 'efm_ranklist', 'ranklist.json'), 'w') as fp:
				json.dump(full_ranklist, fp)

	def output_ranklist_mask(self, test_loader, item_pool, failed_case, top_aspect):
		self.model.cpu()
		self.model.eval()

		d = {}
		for i, row in enumerate(failed_case):
			if row[0] not in d:
				d[row[0]] = []	
			d[row[0]].append([row[1], row[2]])

		print('output ranklist')
		pbar=tqdm(len(test_loader))
		with torch.no_grad():
			full_ranklist = {}
			test_scores = {}
			for i, batch in enumerate(test_loader):
				pbar.update(1)
				for j in range(batch.shape[0]):
					test_user_single = batch[j].item()
					test_items = list(item_pool)
					sorted_items = []
					test_users=torch.tensor([test_user_single]*len(test_items))
					test_items=torch.tensor(test_items)
					mask = torch.ones_like(torch.rand([self.config['num_items'], self.config['num_aspect']]))
					for k in d[test_user_single]:
						for m in k[1]:
							mask[k[0]][m]=0

					item_scores = self.model.predict_mask(test_users, test_items, mask).tolist()

					sorted_scores_index=sorted(range(len(item_scores)), key=lambda k:item_scores[k], reverse=True)
					for k in xrange(len(sorted_scores_index)):
						sorted_items.append(str(test_items[sorted_scores_index[k]].item()))
					full_ranklist[str(test_user_single)]=sorted_items
			pbar.close()
		if not os.path.isdir(os.path.join(self.config['dest_dir'], 'efm_ranklist')):
			os.mkdir(os.path.join(self.config['dest_dir'], 'efm_ranklist'))

		with open(os.path.join(self.config['dest_dir'], 'efm_ranklist', 'scrutable_ranklist.json'), 'w') as fp:
			json.dump(full_ranklist, fp)

	def output_ranklist_full(self, test_loader, interact_hist, item_pool):
		self.model.cpu()
		self.model.eval()
		print('output ranklist')
		# test_items = list(item_pool)
		with torch.no_grad():
			full_ranklist = {}
			user_aspect_embedding = self.model.embedding_user.weight
			item_aspect_embedding = self.model.embedding_item.weight
			user_rating_embedding = self.model.embedding_user_2.weight
			item_rating_embedding = self.model.embedding_item_2.weight
			aspect_embedding = self.model.embedding_aspect.weight

			# print(user_embedding.shape, item_embedding.shape)
			rating_matrix = torch.mm(user_rating_embedding, torch.transpose(item_rating_embedding, 0, 1))+\
							torch.mm(user_aspect_embedding, torch.transpose(item_aspect_embedding, 0, 1))
			# rating_matrix = torch.mm(user_rating_embedding, torch.transpose(item_rating_embedding, 0, 1))

			user_aspect_matrix = torch.mm(user_aspect_embedding, torch.transpose(aspect_embedding, 0, 1))
			item_aspect_matrix = torch.mm(item_aspect_embedding, torch.transpose(aspect_embedding, 0, 1))

			test_scores = {}

			pbar = tqdm(total=len(test_loader))
			for i, batch in enumerate(test_loader):
				for j in range(batch.shape[0]):
					test_user_single = batch[j].item()
					test_items = set(item_pool)-set(interact_hist[test_user_single])
					test_items = list(test_items)
					
					sorted_items = []
					test_users=torch.tensor([test_user_single]*len(test_items))
					test_items=torch.tensor(test_items)

					item_scores=self.model.predict(test_users,test_items).tolist()
					test_items=test_items.tolist()

					test_scores[test_user_single] = [test_items, item_scores]
					# [test_user, [test_item1, ..., test_itemk], [scores]]

					sorted_scores_index=sorted(range(len(item_scores)), key=lambda k:item_scores[k], reverse=True)
					for k in xrange(len(sorted_scores_index)):
						sorted_items.append(str(test_items[sorted_scores_index[k]]))
					full_ranklist[str(test_user_single)]=sorted_items
				pbar.update(1)
			pbar.close()

		return full_ranklist





	def output_item_scores(self, test_loader, interact_hist, item_pool):
		self.model.cpu()
		self.model.eval()
		print('output item scores!')
		with torch.no_grad():
			user_aspect_embedding = self.model.embedding_user.weight
			item_aspect_embedding = self.model.embedding_item.weight
			user_rating_embedding = self.model.embedding_user_2.weight
			item_rating_embedding = self.model.embedding_item_2.weight
			aspect_embedding = self.model.embedding_aspect.weight
			rating_matrix = torch.mm(user_rating_embedding, torch.transpose(item_rating_embedding, 0, 1))+\
							torch.mm(user_aspect_embedding, torch.transpose(item_aspect_embedding, 0, 1))
			user_aspect_matrix = torch.mm(user_aspect_embedding, torch.transpose(aspect_embedding, 0, 1))
			item_aspect_matrix = torch.mm(item_aspect_embedding, torch.transpose(aspect_embedding, 0, 1))
			test_scores = {}

			for i, batch in enumerate(test_loader):
				for j in range(batch.shape[0]):
					test_user_single = batch[j].item()
					test_items = list(item_pool)

					test_users = torch.tensor([test_user_single]*len(test_items))
					test_items = torch.tensor(test_items)
					item_scores = self.model.predict(test_users, test_items).tolist()
					test_items = test_items.tolist()

					test_scores[test_user_single] = [test_items, item_scores]

		return test_scores




class EFMEngine(Engine):
	def __init__(self, config):
		self.model = EFM(config)
		super(EFMEngine, self).__init__(config)
		self.model.apply(init_weights)