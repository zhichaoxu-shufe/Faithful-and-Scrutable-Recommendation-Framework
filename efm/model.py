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

		# # aspect embedding
		# self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim*(2**(self.config['mlp_layer']-1)))
		# self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim*(2**(self.config['mlp_layer']-1)))
		# self.embedding_aspect = torch.nn.Embedding(num_embeddings=self.num_aspect, embedding_dim=self.latent_dim*(2**(self.config['mlp_layer']-1)))
		# # rating embedding
		# self.embedding_user_2 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim*(2**(self.config['mlp_layer']-1)))
		# self.embedding_item_2 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim*(2**(self.config['mlp_layer']-1)))
		# self.item_bias = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=1)

		# self.embedding_user.weight.data.normal_(0, 0.01)
		# self.embedding_item.weight.data.normal_(0, 0.01)
		# self.embedding_user_2.weight.data.normal_(0, 0.01)
		# self.embedding_item_2.weight.data.normal_(0, 0.01)
		# self.embedding_aspect.weight.data.normal_(0, 0.01)
		# self.item_bias.weight.data = torch.zeros_like(self.item_bias.weight.data)

		# self.MLP_modules=[]
		# for i in range(self.config['mlp_layer']):
		# 	if i == 0:
		# 		input_size=self.latent_dim*(2**self.config['mlp_layer'])
		# 	else:
		# 		input_size = input_size//2
		# 	self.MLP_modules.append(nn.Dropout(p=self.config['dropout']))
		# 	self.MLP_modules.append(nn.Linear(input_size, input_size//2))
		# 	self.MLP_modules.append(nn.ReLU())
		# self.MLP_layers=nn.Sequential(*self.MLP_modules)

		# for m in self.MLP_layers:
		# 	if isinstance(m, nn.Linear):
		# 		nn.init.xavier_uniform_(m.weight)

		# self.predict_layer=nn.Linear(self.latent_dim, 1)

		self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.embedding_aspect = torch.nn.Embedding(num_embeddings=self.num_aspect, embedding_dim=self.latent_dim)
		# rating embedding
		self.embedding_user_2 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item_2 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.item_bias = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=1)

		# self.embedding_user.weight.data.normal_(0, 0.5)
		# self.embedding_item.weight.data.normal_(0, 0.5)
		# self.embedding_user_2.weight.data.normal_(0, 0.5)
		# self.embedding_item_2.weight.data.normal_(0, 0.5)
		# self.embedding_attribute.weight.data.normal_(0, 0.5)
		self.item_bias.weight.data = torch.zeros_like(self.item_bias.weight.data)
		self.affine_output = torch.nn.Linear(in_features=self.latent_dim*2, out_features=1)
		self.logistic = torch.nn.Sigmoid()
		nn.init.xavier_uniform_(self.affine_output.weight)

		self.topk = {}



	def forward(self, user_indices, item_indices, aspect_indices):
		# user_rating_embedding = self.embedding_user_2(user_indices) # [bz, latent_dim]
		# item_rating_embedding = self.embedding_item_2(item_indices) # [bz, latent_dim]
		# user_embedding = self.embedding_user(user_indices)
		# item_embedding = self.embedding_item(item_indices)
		# # print(user_embedding, user_embedding.shape)
		# aspect_embedding = self.embedding_aspect(aspect_indices)
		# user_rating_embedding = torch.cat((user_rating_embedding, user_embedding), 1)
		# item_rating_embedding = torch.cat((item_rating_embedding, item_embedding), 1)
		# element_product = torch.mul(user_rating_embedding, item_rating_embedding) # [bz, latent_dim*2]
		# element_product_1=self.MLP_layers(element_product)
		# element_product_1=self.predict_layer(element_product_1)
		# item_bias = self.item_bias(item_indices) # [bz, 1]
		# assert element_product_1.shape==item_bias.shape, 'shape not matching'
		# element_product_2 = torch.mul(user_embedding, aspect_embedding).sum(axis=1)
		# element_product_3 = torch.mul(item_embedding, aspect_embedding).sum(axis=1)
		# return element_product_1+item_bias, element_product_2, element_product_3

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
		# element_product = self.affine_output(element_product)
		# element_product = self.logistic(element_product)
		item_bias = self.item_bias(item_indices).squeeze()
		# item_bias = self.item_bias(item_indices)
		assert item_bias.shape==element_product.shape, 'wrong shape'

		# print(user_embedding.shape, self.embedding_aspect.weight.shape)
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

		# element_product=torch.mul(X, Y)
		# print(torch.topk(element_product[0], k=5, largest=True).indices)

		element_product=torch.mul(X, Y).sum(axis=1)

		rating=torch.mul(user_rating_embedding, item_rating_embedding).sum(axis=1)

		# rating=torch.mul(user_rating_embedding, item_rating_embedding)
		# rating=self.affine_output(rating)
		# rating=self.logistic(rating)
		return (1-self.config['alpha'])*element_product+self.config['alpha']*rating

		# user_embedding=self.embedding_user(user_indices) # [bz, embed_size]
		# item_embedding=self.embedding_item(item_indices)

		# X=torch.mm(user_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]
		# Y=torch.mm(item_embedding, self.embedding_aspect.weight.T) # [bz, num of aspects]

		# element_product_1=torch.mul(X, Y).sum(axis=1)

		# user_rating_embedding=self.embedding_user_2(user_indices)
		# item_rating_embedding=self.embedding_item_2(item_indices)
		# user_rating_embedding=torch.cat((user_rating_embedding, user_embedding), 1)
		# item_rating_embedding=torch.cat((item_rating_embedding, item_embedding), 1)

		# element_product_2=torch.mul(user_rating_embedding, item_rating_embedding)
		# element_product_2=self.MLP_layers(element_product_2)
		# rating=self.predict_layer(element_product_2)
		# # print(user_indices, item_indices)
		# # time.sleep(0.5)
		# return (1-self.config['alpha'])*element_product_1+self.config['alpha']*rating


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

		# element_product = torch.mul(X,Y).sum(axis=1)
		# element_product = torch.mul(element_product, mask)
		rating=torch.mul(user_rating_embedding, item_rating_embedding).sum(axis=1)
		# print(rating.sum(), element_product.sum())
		# time.sleep(0.1)
		return (1-self.config['alpha'])*element_product+self.config['alpha']*rating


	def init_weight(self):
		pass