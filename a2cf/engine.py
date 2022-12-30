from dataset import *
from model import *
from evaluate import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, os, json, copy, logging
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_

def point_multiplier(tensor1, tensor2):
	return torch.einsum('xy,xy->xy', tensor1, tensor2)

def save_checkpoint(epoch, model, dataset, output_dir):
	if not os.path.isdir(os.path.join(output_dir, 'a2cf_checkpoints')):
		os.mkdir(os.path.join(output_dir, 'a2cf_checkpoints'))
	torch.save(model.state_dict(), os.path.join(output_dir, 'a2cf_checkpoints', 'a2cf_epoch_{}.pt'.format(epoch)))

def save_ranklist(ranklist, dest_dir):
	ranklist2save = []
	for k,v in ranklist.items():
		ranklist2save.append([k,v])
	np.array(ranklist2save).dump(os.path.join(dest_dir,'ranklist.pickle'))

def save_scrutable_ranklist(ranklist, dest_dir):
	ranklist2save = []
	for k,v in ranklist.items():
		ranklist2save.append([k,v])
	np.array(ranklist2save).dump(os.path.join(dest_dir,'scrutable_ranklist.pickle'))

def save_item_scores(item_scores, dest_dir):
	item_scores2save = []
	for k, v in item_scores.items():
		item_scores2save.append([k, v[0], v[1]])
	np.array(item_scores2save).dump(os.path.join(dest_dir, 'item_scores.pickle'))


def use_optimizer(network, params):
	if params['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(network.parameters(),
			lr = params['lr'],
			momentum = params['momentum'],
			weight_decay = params['l2']
			)
	elif params['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(),
			lr = params['lr'],
			weight_decay = params['l2']
			)
	return optimizer

class BPRLoss(torch.nn.Module):
	"""
	BPR loss
	Input scores of positive and negative samples. The score of positive sample is expected to be larger
	"""

	def __init__(self, margin):
		super(BPRLoss, self).__init__()
		self.margin = margin

	def forward(self, positive_score, negative_score, size_average=False):
		losses = - torch.sigmoid((positive_score - negative_score).squeeze())
		return losses.mean() if size_average else losses.sum()


# use for a2cf fully connected
class Engine(object):
	# meta engine to train a2cf model
	def __init__(self, config):
		self.config = config
		self.opt = use_optimizer(self.model, config)

		user_matrix_path = os.path.join(config['pretrain_path'], 'a2cf_user_matrix_pretrained.pickle')
		item_matrix_path = os.path.join(config['pretrain_path'], 'a2cf_item_matrix_pretrained.pickle')

		self.X = np.load(user_matrix_path, allow_pickle=True)
		self.Y = np.load(item_matrix_path, allow_pickle=True)
		self.X, self.Y = torch.tensor(self.X).float(), torch.tensor(self.Y).float()

		self.embedding_model = EmbeddingNet(self.X, self.Y, len(self.X), len(self.Y), len(self.X[0]), self.config['latent_dim'])
		self.embedding_model.load_state_dict(torch.load(os.path.join(config['pretrain_path'], 'pretrain.pt')))
		
		self.U = self.embedding_model.state_dict()['U'].detach().float()
		self.I = self.embedding_model.state_dict()['I'].detach().float()
		self.F = self.embedding_model.state_dict()['F'].detach().float()

		self.softmax = torch.nn.Softmax()
		self.epsilon = self.config['epsilon']

		if self.config['use_cuda']:
			self.X = self.X.cuda()
			self.Y = self.Y.cuda()
			self.U = self.U.cuda()
			self.I = self.I.cuda()
			self.F = self.F.cuda()
			self.softmax = self.softmax.cuda()

		self.loss = BPRLoss(margin=0)

	def compute_vp(self, X_user, Y_item, F, epsilon):

		ln = self.softmax(torch.mul(X_user, Y_item)/epsilon)
		vp = torch.mm(ln, F)
		return vp.squeeze()

	def compute_vp_mask(self, X_user, Y_item, F, mask, epsilon):
		ln = self.softmax(torch.mul(X_user, Y_item)/epsilon)
		ln = point_multiplier(ln, mask)
		vp = torch.mm(ln, F)
		return vp.squeeze()


	def train_pair_epoch(self, train_loader, epoch_id):
		if self.config['use_cuda']:
			self.model.cuda()
		self.model.train()
		total_loss = 0
		pbar=tqdm(total=len(train_loader))
		for batch_id, batch in enumerate(train_loader):
			user, pos, neg = batch[0], batch[1], batch[2]
			if self.config['use_cuda']:
				user, pos, neg = user.cuda(), pos.cuda(), neg.cuda()
			self.opt.zero_grad()
			u_embed, pos_embed, neg_embed = self.X[user], self.Y[pos], self.Y[neg]
			pos_vp = self.compute_vp(u_embed, pos_embed, self.F, self.epsilon)
			neg_vp = self.compute_vp(u_embed, neg_embed, self.F, self.epsilon)
			pos_ui_product = torch.mul(self.U[user], self.I[pos]).squeeze()
			neg_ui_product = torch.mul(self.U[user], self.I[neg]).squeeze()
			pos_cat = torch.cat((pos_ui_product, pos_vp), dim=0)
			neg_cat = torch.cat((neg_ui_product, neg_vp), dim=0)
			pos_pred = self.model.forward(pos_cat)
			neg_pred = self.model.forward(neg_cat)
			loss = self.loss(pos_pred, neg_pred) * 10000
			loss.backward()

			gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
			self.opt.step()
			total_loss += loss.item()
			pbar.update(1)
		pbar.close()
		print('loss: {}'.format(total_loss/len(train_loader)))
		return total_loss/len(train_loader)


	def train_batch_epoch(self, train_loader, epoch_id):
		if self.config['use_cuda']:
			self.model.cuda()
			self.X = self.X.cuda()
			self.Y = self.Y.cuda()
			self.U = self.U.cuda()
			self.I = self.I.cuda()
			self.F = self.F.cuda()
			self.softmax = self.softmax.cuda()

		self.model.train()
		total_loss = 0
		pbar=tqdm(len(train_loader))
		for batch_id, batch in enumerate(train_loader):
			batch = batch.squeeze()
			user_indices = batch[0]
			item_indices = batch[1]

			if self.config['use_cuda']:
				user_indices, item_indices = user_indices.cuda(), item_indices.cuda()

			self.opt.zero_grad()
			u_embed = self.X[user_indices]
			i_embed = self.Y[item_indices]

			vp = self.compute_vp(u_embed, i_embed, self.F, self.epsilon)

			ui_product = torch.mul(self.U[user_indices], self.I[item_indices])
			batch_embed = torch.cat((ui_product, vp), dim=1)
			gt = batch_embed[0]
			neg = batch_embed[1]
			pred = self.model.forward(batch_embed)
			loss = pred[0]*(pred.shape[0]-1)-pred[1:].sum()
			loss = -loss

			loss.backward()
			
			gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
			self.opt.step()

			total_loss += loss.item()
			pbar.update(1)
		pbar.close()

		print('loss: {}'.format(total_loss/len(train_loader)))
		return total_loss/len(train_loader)

	def output_ranklist(self, test_loader, interact_hist, item_pool):
		self.model.cpu()
		self.model.eval()
		if self.config['use_cuda']:
			self.X = self.X.cpu()
			self.Y = self.Y.cpu()
			self.U = self.U.cpu()
			self.I = self.I.cpu()
			self.F = self.F.cpu()
			self.softmax = self.softmax.cpu()

		print('output ranklist')
		with torch.no_grad():
			full_ranklist = {}

			pbar = tqdm(len(test_loader))
			for i, batch in enumerate(test_loader):
				test_user_indice = batch.item()
				test_items = list(item_pool-interact_hist[test_user_indice])
				test_items_tensor = torch.tensor(test_items)
				test_users_tensor = batch.repeat(test_items_tensor.shape[0])
				sorted_items = []

				u_embed = self.X[test_users_tensor]
				i_embed = self.Y[test_items_tensor]
				vp = self.compute_vp(u_embed, i_embed, self.F, self.epsilon)
				ui_product = torch.mul(self.U[test_users_tensor], self.I[test_items_tensor])
				emb_cat = torch.cat((ui_product, vp), dim=1)
				score = self.model.forward(emb_cat)
				score = score.squeeze().tolist()

				sorted_items = []
				sorted_scores_index = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
				for k in range(len(sorted_scores_index)):
					sorted_items.append(test_items[sorted_scores_index[k]])
				full_ranklist[test_user_indice] = sorted_items

				pbar.update(1)
			pbar.close()

		if not os.path.isdir(os.path.join(self.config['input_dir'], 'a2cf_ranklist')):
			os.mkdir(os.path.join(self.config['input_dir'], 'a2cf_ranklist'))

		dest_dir = os.path.join(self.config['input_dir'], 'a2cf_ranklist')
		save_ranklist(full_ranklist, dest_dir)

		return full_ranklist

	def output_ranklist_mask(self, test_loader, item_pool, interact_hist, failed_case, num_aspect):
		self.model.cpu()
		self.model.eval()
		if self.config['use_cuda']:
			self.X = self.X.cpu()
			self.Y = self.Y.cpu()
			self.U = self.U.cpu()
			self.I = self.I.cpu()
			self.F = self.F.cpu()
			self.softmax = self.softmax.cpu()

		# construct failure case dict
		d = {}
		for i, row in enumerate(failed_case):
			if row[0] not in d:
				d[row[0]] = []
			d[row[0]].append([row[1], row[2]])

		print('output_ranklist')
		with torch.no_grad():
			full_ranklist = {}
			pbar = tqdm(len(test_loader))
			for i, batch in enumerate(test_loader):
				test_user_indice = batch.item()
				test_items = list(item_pool)
				test_items_tensor = torch.tensor(test_items)
				test_users_tensor = batch.repeat(test_items_tensor.shape[0])
				sorted_items = []
				u_embed = self.X[test_users_tensor]
				i_embed = self.Y[test_items_tensor]

				mask = torch.ones_like(torch.mul(u_embed, i_embed))
				for j, row in enumerate(d[test_user_indice]):
					for k in row[1][:num_aspect]:
						mask[row[0]][k] = 0
				vp = self.compute_vp_mask(u_embed, i_embed, self.F, mask, self.epsilon)
				ui_product = torch.mul(self.U[test_users_tensor], self.I[test_items_tensor])
				emb_cat = torch.cat((ui_product, vp), dim=1)
				score = self.model.forward(emb_cat)
				score = score.squeeze().tolist()

				sorted_items = []
				sorted_scores_index = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
				for k in range(len(sorted_scores_index)):
					sorted_items.append(test_items[sorted_scores_index[k]])
				full_ranklist[test_user_indice] = sorted_items

				pbar.update(1)
			pbar.close()

		if not os.path.isdir(os.path.join(self.config['input_dir'], 'a2cf_ranklist')):
			os.mkdir(os.path.join(self.config['input_dir'], 'a2cf_ranklist'))

		dest_dir = os.path.join(self.config['input_dir'], 'a2cf_ranklist')
		save_scrutable_ranklist(full_ranklist, dest_dir)

		return full_ranklist


	def output_item_scores(self, test_loader):
		self.model.cpu()
		self.model.eval()
		if self.config['use_cuda']:
			self.X = self.X.cpu()
			self.Y = self.Y.cpu()
			self.U = self.U.cpu()
			self.I = self.I.cpu()
			self.F = self.F.cpu()
			self.softmax = self.softmax.cpu()

		print('output item scores for white box training')
		test_scores = {}
		test_items = [i for i in range(self.config['num_item'])]

		with torch.no_grad():
			pbar = tqdm(len(test_loader))
			for i, batch in enumerate(test_loader):
				pbar.update(1)
				test_user_int = batch.item()
				test_items_tensor = torch.tensor(test_items)
				test_users_tensor = batch.repeat(test_items_tensor.shape[0])
				sorted_items = []

				u_embed = self.X[test_users_tensor]
				i_embed = self.Y[test_items_tensor]
				vp = self.compute_vp(u_embed, i_embed, self.F, self.epsilon)
				ui_product = torch.mul(self.U[test_users_tensor], self.I[test_items_tensor])
				emb_cat = torch.cat((ui_product, vp), dim=1)
				score = self.model.forward(emb_cat)
				score = score.squeeze().tolist()

				test_scores[test_user_int] = [test_items, score]
			pbar.close()

		if not os.path.isdir(os.path.join(self.config['input_dir'], 'a2cf_item_scores')):
			os.mkdir(os.path.join(self.config['input_dir'], 'a2cf_item_scores'))

		dest_dir = os.path.join(self.config['input_dir'], 'a2cf_item_scores')
		save_item_scores(test_scores, dest_dir)

	def evaluate(self, ranklist, gt, epoch_id, cut_off=20):
		gt = gt.tolist()
		qrel_map = {}
		for i, row in enumerate(gt):
			qrel_map[row[0]] = row[1]

		recall, precision, ndcg, hit, reciprocal_rank = print_metrics_with_rank_cutoff(ranklist, qrel_map, cut_off)

		return recall, precision, ndcg, hit, reciprocal_rank


class A2CFEngine(Engine):
	def __init__(self, config):
		self.model = RankingScoreNet(config['latent_dim'])
		super(A2CFEngine, self).__init__(config)

class A2CF2WayEngine(Engine):
	def __init__(self, config):
		self.model = TwoWayNet(config['latent_dim'], config['alpha'])
		super(A2CF2WayEngine, self).__init__(config)


