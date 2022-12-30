from model import *
from dataloader import *


import argparse
import numpy as np
import torch
import random, sys, time
import pandas
import copy
import json
from tqdm import tqdm
import math
from six.moves import xrange
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_
import os
import pickle 


def init_weights(m, scale):
	if type(m) == torch.nn.Embedding:
		torch.nn.init.normal(m.weight, 0.0, scale)

def use_optimizer(network, params):
	if params['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(network.parameters(),
		lr = param['lr'],
		momentum = param['momentum'],
		weight_decay = params['l2']
		)
	if params['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(),
		lr = params['lr'],
		weight_decay = params['lr']
		)
	return optimizer

class Engine(object):
	# meta engine for training TransE model
	def __init__(self, config):
		self.config = config
		self.crit = torch.nn.MSELoss()
		self.opt = use_optimizer(self.model, config)

	def train_an_epoch(self, train_loader, epoch_id):
		if self.config['use_cuda']:
			self.model.cuda()
		self.model.train()
		total_loss = 0
		pbar=tqdm(total=len(train_loader))
		for batch_id, batch in enumerate(train_loader):
			assert isinstance(batch[0], torch.LongTensor), 'wrong tensor type'
			heads, relations, tails, neg_heads, neg_relations, neg_tails = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

			loss = self.train_single_batch(heads, relations, tails, neg_heads, neg_relations, neg_tails)
			gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
			total_loss += loss
			pbar.update(1)
		pbar.close()
		print('Epoch loss: ', total_loss)

	def train_single_batch(self, heads, relations, tails, neg_heads, neg_relations, neg_tails):
		if self.config['use_cuda']:
			heads, relations, tails, neg_heads, neg_relations, neg_tails = \
			heads.cuda(), relations.cuda(), tails.cuda(), neg_heads.cuda(), neg_relations.cuda(), neg_tails.cuda()
			self.opt.zero_grad()

			neg_heads, neg_relations, neg_tails = torch.flatten(neg_heads), torch.flatten(neg_relations), torch.flatten(neg_tails)
			
			loss = torch.norm(self.model(heads, relations, tails), p=1, dim=1).sum(axis=0)
			loss.backward()
			self.opt.step()
			# print(loss.item())
			return loss.item()

		else:
			self.opt.zero_grad()
			loss = torch.norm(self.model(heads, relations, tails), p=1, dim=1).sum(axis=0)
			# print(loss)
			loss.backward()
			self.opt.step()
			return loss.item()

	def test(self, test_loader, epoch_id):
		self.model.cpu()
		self.model.eval()
		total_hit = 0
		pbar = tqdm(total=len(test_loader))
		for batch_id, batch in enumerate(test_loader):
			assert isinstance(batch[0], torch.LongTensor), 'wrong tensor type'
			# print('candidate length: ', batch[0].shape, batch[1].shape, batch[2].shape)
			heads, relations, tails = batch[0].squeeze(), batch[1].squeeze(), batch[2].squeeze()
			scores = self.model(heads, relations, tails)
			# print('score shape', scores.shape, heads.shape, relations.shape, tails.shape)
			scores = torch.norm(self.model(heads, relations, tails), p=1, dim=1)
			# print(scores.shape)
			# print(torch.topk(scores, 1, largest=False).indices.item())
			# print('scores shape: ', scores.shape, '   top indice: ', torch.topk(scores, 1, largest=False).indices.item())
			# time.sleep(0.5)
			if torch.topk(scores, 1, largest=False).indices.item() == 10:
				total_hit += 1
			pbar.update(1)
		pbar.close()
		hr = total_hit/len(test_loader)
		print('Hit Rate: ', hr)
		
		return hr

	def save_model(self, model_path):
		torch.save(self.model, model_path)



class TransEEngine(Engine):
	def __init__(self, config):
		self.model=KGEModel(config)
		if config['use_cuda'] is True:
			use_cuda(True, config['device_id'])
			self.model.cuda()
		super(TransEEngine, self).__init__(config)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--input_dir', type=str)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_user', type=int, default=2911)
	parser.add_argument('--num_item', type=int, default=20229)
	parser.add_argument('--num_aspect', type=int, default=210)
	parser.add_argument('--model_name', type=str, default='TransE')
	parser.add_argument('--num_neg', type=int, default=4)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--lr', type=float, default=5e-6)
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--momentum', type=float, default=1e-1)
	parser.add_argument('--l2', type=float, default=1e-4)
	parser.add_argument('--epsilon', type=float, default=2.0)
	parser.add_argument('--gamma', type=float, default=2.0)
	parser.add_argument('--use_cuda', type=int, default=0)
	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--double_entity_embedding', type=int, default=0)
	parser.add_argument('--model_dir', type=str, default='wbx/hetero/model_dir')

	args=parser.parse_args()
	config=vars(args)

	engine = TransEEngine(config)
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
	testset = np.load(args.input_dir+'kge_test.pickle', allow_pickle=True)
	testset = testset.tolist()
	testset = KGETestDataset(testset, aspect_pool, config)

	trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
	testLoader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
	for i in range(args.epoch):
		trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
		engine.train_an_epoch(trainLoader, i+1)
		engine.test(testLoader, i+1)

		if (i+1) % 1 == 0:
			if not os.path.exists(config["model_dir"]):
				os.mkdir(config["model_dir"])
			save_path = os.path.join(config["model_dir"], f"kge_{i+1}.pkl")
			engine.save_model(save_path)
			print("save model to " + save_path)

			uia_ebds = {}
			print("here", engine.model.user_embeddings.weight.data.numpy().shape)
			print("here", engine.model.item_embeddings.weight.data.numpy().shape)
			print("here", engine.model.aspect_embeddings.weight.data.numpy().shape)
			uia_ebds["user_embeddings"] = engine.model.user_embeddings.weight.data.numpy()
			uia_ebds["item_embeddings"] = engine.model.item_embeddings.weight.data.numpy()
			uia_ebds["aspect_embeddings"] = engine.model.aspect_embeddings.weight.data.numpy()
			uia_ebds["relation_embeddings"] = engine.model.relation_embeddings.weight.data.numpy()
			uia_path = os.path.join(config["model_dir"], f"uia_ebds_{i+1}.pkl")

			with open(uia_path, "wb") as f:
				pickle.dump(uia_ebds, f)
			print("save uia_ebds to " + uia_path)
