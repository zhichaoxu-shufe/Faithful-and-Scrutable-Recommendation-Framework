from functions import *
from dataset import AgnosticDataset
from hard_match_model import HardMatchModel
from evaluate import print_metrics_with_rank_cutoff

import random, sys, time
import copy
import json
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_

def save_checkpoint(epoch, model, dataset, output_dir, model_name):
	ckpt_dir = os.path.join(output_dir, f'{model_name}_checkpoints')
	if not os.path.isdir(ckpt_dir):
		os.mkdir(ckpt_dir)
	torch.save(model.state_dict(), os.path.join(ckpt_dir, 'wbx_epoch_{}.pt'.format(epoch)))

def save_best_checkpoint(model, output_dir, model_name):
	ckpt_dir = os.path.join(output_dir, f'{model_name}_checkpoints')
	if not os.path.isdir(ckpt_dir):
		os.mkdir(ckpt_dir)
	torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))

def save_ranklist(ranklist, dest_dir):
	ranklist2save = []
	for k,v in ranklist.items():
		ranklist2save.append([k,v])
	np.array(ranklist2save).dump(os.path.join(dest_dir,'ranklist.pickle'))

def save_item_scores(item_scores, dest_dir):
	item_scores2save = []
	for k, v in item_scores.items():
		item_scores2save.append([k, v[0], v[1]])
	np.array(item_scores2save).dump(os.path.join(dest_dir, 'item_scores.pickle'))

def resume_checkpoint(path, model):
	return model.load_state_dict(torch.load(path))

def generate_float_mask(inputs, pad_idx=-1):
	mask = torch.ones(inputs.size())
	poss = torch.where(inputs==pad_idx)
	mask[poss] = 0

	return mask 

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

def batch_kl_div(pred_dists, target_dists, masks):
	"""
	Args: 
		pred_dists: FloatTensor; [bz, num_sample]
		target_dists: FloatTensor; [bz, num_samples]
		masks: FloatTensor; [bz, num_sample]
	
	Returns: 
		kl_losses: [bz]
	"""
	bz, num_sample = pred_dists.size()
	kl_losses = torch.randn(bz).to(pred_dists.device)

	for i in range(bz):
		each_loss = F.kl_div(pred_dists[i].log(), target_dists[i], reduction="none")
		each_loss = torch.sum(each_loss)
		kl_losses[i] = each_loss
	
	return kl_losses

class Engine(object):
	def __init__(self, config):
		self.config = config
		self.opt = use_optimizer(self.model, config)
		self.softmax = nn.Softmax(dim=0)

	def evaluate(self, ranklist, gt, epoch_id, cut_off=20):
		gt = gt.tolist()
		qrel_map = {}
		for i, row in enumerate(gt):
			qrel_map[row[0]] = row[1]
		recall, precision, ndcg, hit, reciprocal_rank = print_metrics_with_rank_cutoff(ranklist, qrel_map, cut_off)

		return recall, precision, ndcg, hit, reciprocal_rank

	def train_hard_match_softmax_epoch(self, train_loader, epoch_id):
		if self.config['use_cuda']:
			self.model.cuda()
		self.model.train()
		total_loss = 0
		pbar = tqdm(total=len(train_loader))
		for batch_id, batch in enumerate(train_loader):
			batch = batch.squeeze()
			logits = self.model.full_compute()[batch[0], batch[1]]
			# logits = self.model.unscaled_full_compute()[torch.tensor(0), ...]
			if self.config['use_temperature']:
				logits /= self.config['temperature']
			softmaxed = self.softmax(logits)
			loss = -softmaxed[0]
			loss.backward()
			gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
			self.opt.step()

			total_loss += loss.item()
			pbar.update(1)
		pbar.close()
		print('loss: ', total_loss/len(train_loader))
		return total_loss/len(train_loader)

	def train_hard_match_kl_epoch(self, train_loader, epoch_id):
		if self.config['use_cuda']:
			self.model.cuda()
		self.model.train()
		total_loss = 0
		for idx, (user_indices, labels) in enumerate(train_loader):
			if self.config['use_cuda']:
				user_indices = user_indices.cuda()
				labels = labels.cuda()

			self.opt.zero_grad()
			pred_probs = self.model.batch_compute(user_indices)
			batch_loss = batch_kl_div(pred_probs, labels, None).mean()

			batch_loss.backward()
			gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
			self.opt.step()

			total_loss += batch_loss.item()

			# print info
			if (idx+1) % (len(train_loader)//5) == 0:
				print(f"epoch: {epoch_id+1}/{self.config['epoch']}, step: {idx+1}/{len(train_loader)}, cur_loss: {batch_loss.cpu().item():.3f}, gnorm: {gnorm:.3f}")

		return total_loss / len(train_loader)


	def output_ranklist(self, test_loader, interact_hist, item_pool):
		self.model.cpu()
		self.model.eval()
		print('output ranklist')
		with torch.no_grad():
			full_ranklist = {}			
			test_scores = {}

			ui_ratings = self.model.full_compute().cpu().data

			pbar = tqdm(len(test_loader))
			for i, batch in enumerate(test_loader):
				for j in range(batch.shape[0]):
					test_user_single = batch[j].item()
					test_items = set(item_pool)-set(interact_hist[test_user_single])
					test_items = list(test_items)
					
					sorted_items = []
					test_users=torch.tensor([test_user_single]*len(test_items))
					test_items=torch.tensor(test_items)

					item_scores = ui_ratings[test_users, test_items].tolist()
					test_items=test_items.tolist()
					
					test_scores[test_user_single] = [test_items, item_scores]
					sorted_scores_index=sorted(range(len(item_scores)), key=lambda k:item_scores[k], reverse=True)
					for k in range(len(sorted_scores_index)):
						sorted_items.append(test_items[sorted_scores_index[k]])
					full_ranklist[test_user_single]=sorted_items
					pbar.update(1)
					
			pbar.close()

		return full_ranklist

	def output_item_scores(self, test_loader, interact_hist, item_pool):
		self.model.cpu()
		self.model.eval()
		print('output item scores!')
		with torch.no_grad():
			test_scores = {}
			pbar = tqdm(total=self.config['num_user'])
			for i, batch in enumerate(test_loader):
				for j in range(batch.shape[0]):
					test_user_single = batch[j].item()
			
					test_items = list(item_pool)
					test_users = torch.tensor([test_user_single]*len(test_items))
					test_items = torch.tensor(test_items)
					item_scores = self.model.predict(test_users, test_items).tolist()
					test_items = test_items.tolist()

					test_scores[test_user_single] = [test_items, item_scores]
					pbar.update(1)
			pbar.close()

		return test_scores

class HMEngine(Engine):
	def __init__(self, config):
		self.model = HardMatchModel(config)
		super().__init__(config)
		# self.model.apply(init_weights)