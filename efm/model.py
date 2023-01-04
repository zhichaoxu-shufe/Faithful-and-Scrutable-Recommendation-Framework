import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import normal_
from tqdm import tqdm
import json, time
from six.moves import xrange

def use_cuda(enabled, device_id=0):
	if enabled:
		assert torch.cuda.is_available(), 'CUDA is not available'
		torch.cuda.set_device(device_id)

def point_multiplier(tensor1, tensor2):
	return torch.einsum('xy,xy->xy', tensor1, tensor2)

class EFM(torch.nn.Module):
	def __init__(self, config):
		super(EFM, self).__init__()
		self.config = config
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.num_aspect = config['num_aspect']
		
		self.latent_dim = config['latent_dim']

		self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.embedding_aspect = torch.nn.Embedding(num_embeddings=self.num_aspect, embedding_dim=self.latent_dim)
		# rating embedding
		self.embedding_user_2 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item_2 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.item_bias = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=1)

		self.item_bias.weight.data = torch.zeros_like(self.item_bias.weight.data)
		self.affine_output = torch.nn.Linear(in_features=self.latent_dim*2, out_features=1)
		self.logistic = torch.nn.Sigmoid()
		nn.init.xavier_uniform_(self.affine_output.weight)

		self.topk = {}



	def forward(self, user_indices, item_indices, aspect_indices):
		user_rating_embedding = self.embedding_user_2(user_indices) # [bz, latent_dim]
		item_rating_embedding = self.embedding_item_2(item_indices) # [bz, latent_dim]
		
		user_embedding = self.embedding_user(user_indices)
		item_embedding = self.embedding_item(item_indices)
		# print(user_embedding, user_embedding.shape)
		attribute_embedding = self.embedding_aspect(aspect_indices)

		user_rating_embedding = torch.cat((user_rating_embedding, user_embedding), 1)
		item_rating_embedding = torch.cat((item_rating_embedding, item_embedding), 1)

		element_product = torch.mul(user_rating_embedding, item_rating_embedding)
		element_product = element_product.sum(axis=1)
		item_bias = self.item_bias(item_indices).squeeze()

		assert item_bias.shape==element_product.shape, 'wrong shape'

		X=torch.mm(user_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]
		Y=torch.mm(item_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]
		X_indiced=X[:, aspect_indices].sum(axis=1)
		Y_indiced=Y[:, aspect_indices].sum(axis=1)

		return element_product+item_bias, X_indiced, Y_indiced

	def predict(self, user_indices, item_indices):
		user_embedding=self.embedding_user(user_indices)
		item_embedding=self.embedding_item(item_indices)
		user_rating_embedding=self.embedding_user_2(user_indices)
		item_rating_embedding=self.embedding_item_2(item_indices)

		user_rating_embedding = torch.cat((user_rating_embedding, user_embedding), 1)
		item_rating_embedding = torch.cat((item_rating_embedding, item_embedding), 1)

		X=torch.mm(user_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]
		Y=torch.mm(item_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]		

		element_product=torch.mul(X, Y).sum(axis=1)

		rating=torch.mul(user_rating_embedding, item_rating_embedding).sum(axis=1)

		return (1-self.config['alpha'])*element_product+self.config['alpha']*rating

	def predict_mask(self, user_indices, item_indices, mask):
		user_embedding=self.embedding_user(user_indices)
		item_embedding=self.embedding_item(item_indices)
		user_rating_embedding=self.embedding_user_2(user_indices)
		item_rating_embedding=self.embedding_item_2(item_indices)

		user_rating_embedding = torch.cat((user_rating_embedding, user_embedding), 1)
		item_rating_embedding = torch.cat((item_rating_embedding, item_embedding), 1)
		X=torch.mm(user_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]
		Y=torch.mm(item_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]	
		
		element_product = torch.mul(X,Y)
		element_product = point_multiplier(element_product, mask).sum(axis=1)

		rating=torch.mul(user_rating_embedding, item_rating_embedding).sum(axis=1)
		return (1-self.config['alpha'])*element_product+self.config['alpha']*rating


	def init_weight(self):
		pass