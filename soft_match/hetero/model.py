from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import sys, time, json

import torch
import torch.nn as nn
# import torch.nn.Functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

class KGEModel(torch.nn.Module):
	def __init__(self, config):
		super(KGEModel, self).__init__()
		self.config = config

		self.model_name = config['model_name']
		self.num_user = config['num_user']
		self.num_item = config['num_item']
		self.num_aspect = config['num_aspect']
		self.epsilon = config['epsilon']
		self.num_entity = self.num_user+self.num_item+self.num_aspect
		self.num_relation = 3
		self.gamma = config['gamma']
		self.hidden_dim = config['latent_dim']

		self.gamma = nn.Parameter(torch.tensor([self.gamma]), requires_grad=False)
		print(self.num_user)

		self.embedding_range = nn.Parameter(torch.tensor([(self.gamma.item()+self.epsilon)/self.hidden_dim]), requires_grad=False)
		self.entity_dim = self.hidden_dim*2 if config['double_entity_embedding'] else self.hidden_dim
		self.relation_dim = self.hidden_dim*2 if config['double_entity_embedding'] else self.hidden_dim

		self.entity_embeddings = nn.Embedding(self.num_entity, self.entity_dim)
		self.entity_embeddings.weight.data.normal_(0, 0.01)
		self.relation_embeddings = nn.Embedding(self.num_relation, self.entity_dim)
		self.relation_embeddings.weight.data.normal_(0, 0.01)

		# Hack: num_user + num_item + num_aspect == num_entity
		self.user_embeddings = nn.Embedding(self.num_user, self.hidden_dim)
		self.item_embeddings = nn.Embedding(self.num_item, self.hidden_dim)
		self.aspect_embeddings = nn.Embedding(self.num_aspect, self.hidden_dim)
		assert self.num_entity == self.num_user + self.num_item + self.num_aspect

		if config['model_name'] == 'pRotatE':
			self.modulus = nn.Parameter(torch.tensor([[0.5*self.embedding_range.item()]]))

		if config['model_name'] not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
			raise ValueError('model %s not supported' % config['model_name'])

		if config['model_name'] == 'RotatE' and (not config['double_entity_embedding'] or config['double_relation_embedding']):
			raise ValueError('RotatE should use --double entity embedding')

		if config['model_name'] == 'ComplEx' and (not config['double_entity_embedding'] or not config['double_relation_embedding']):
			raise ValueError('ComplEx should use --double entity embedding and --double relation embedding')

	def TransE(self, head, relation, tail):
		# head: [bz, hidden_dim]

		score = head+relation-tail
		# score = self.gamma.item()-torch.norm(score, p=1, dim=1) # norm along dim=1
		return score

	def forward(self, head, relation, tail):
		head = self.entity_embeddings(head)
		relation = self.relation_embeddings(relation)
		tail = self.entity_embeddings(tail)

		self.trans_entity_ebds_to_uia_ebds()

		return self.TransE(head, relation, tail)

	def mse_loss(self, head, relation, tail):
		return torch.norm(self.forward(head, relation, tail), dim=1)

	def trans_entity_ebds_to_uia_ebds(self):
		self.user_embeddings.weight.data = self.entity_embeddings.weight[:self.num_user].data  
		self.item_embeddings.weight.data = self.entity_embeddings.weight[self.num_user: self.num_user+self.num_item].data  
		self.aspect_embeddings.weight.data = self.entity_embeddings.weight[self.num_user+self.num_item: ].data 