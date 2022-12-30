import torch
import torch.nn as nn 
import torch.nn.functional as F 

# from wbx.hetero.model import KGEModel
from hetero.model import KGEModel

import pickle 
import sys, time


class SimpleModel(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.config = config 
		self.num_user = config['num_user']
		self.num_item = config['num_item']
		self.num_aspect = config['num_aspect']
		self.num_relation = config["num_relation"]
		self.latent_dim = config['latent_dim']
		self.uia_path = config['uia_path']

		self.use_TransE = config["use_TransE"]

		self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
		self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
		self.aspect_embeddings = nn.Embedding(self.num_aspect, self.latent_dim)
		self.relation_embeddings = nn.Embedding(self.num_relation, self.latent_dim)

		if self.use_TransE == 1:
			with open(self.uia_path, "rb") as f:
				self.uia_ebds = pickle.load(f)
				print("load pretrained kge embeddings ... \n")
			
			print(self.uia_ebds["user_embeddings"].shape, self.user_embeddings.weight.data.shape)
			print(self.uia_ebds["item_embeddings"].shape, self.item_embeddings.weight.data.shape)
			self.user_embeddings.weight.data.copy_(torch.from_numpy(self.uia_ebds["user_embeddings"]))
			self.item_embeddings.weight.data.copy_(torch.from_numpy(self.uia_ebds["item_embeddings"]))
			self.aspect_embeddings.weight.data.copy_(torch.from_numpy(self.uia_ebds["aspect_embeddings"]))
			self.relation_embeddings.weight.data.copy_(torch.from_numpy(self.uia_ebds["relation_embeddings"]))

		self.item_bias = nn.Embedding(self.num_item, 1)
		self.item_bias.weight.data.fill_(0.)

	def forward(self, user_indices, item_indices, mode='batch'):
		"""
		See each mode's comment.
		"""
		#print("user_indices: ", user_indices)
		#print("item_indices: ", item_indices)
		if mode == 'batch':
			"""
			Args:
			user_indices: [bz]
			item_indices: [bz, item_num]
		
			Returns:
				score_dists: [bz, item_num]; the score distribution of items given a certain user 
			"""
			bz, num_sample =  item_indices.size()
			score_dists = torch.zeros(bz, num_sample).to(device=item_indices.device)
			user_indices = user_indices.view(-1, 1).repeat(1, num_sample)

			for i in range(score_dists.size()[0]):
				dist = F.softmax(self.rating_scores(user_indices[i], item_indices[i]), dim=0)
				score_dists[i] = dist
			return score_dists
		elif mode == "bundle":
			"""
			Args:
				user_indices: [bz, 1+neg_item]
				item_indices: [bz, 1+neg_item]
			Returns:
				bundle_logits: [bz, 1+neg_item]
			"""
			bz, num_sample = item_indices.size()

			user_indices = user_indices.view(bz*num_sample)
			item_indices = item_indices.view(bz*num_sample)

			bundle_logits = self.rating_scores(user_indices, item_indices)
			bundle_logits = bundle_logits.view(bz, num_sample)
			#print("bundle_logits", bundle_logits)

			return bundle_logits

		elif mode == 'vanilla':
			return self.rating_scores(user_indices, item_indices)

	def rating_scores(self, user_indices, item_indices):
		"""
		batchlized rating score from user to item

		Args:
			user_indices: [bz]
			item_indices: [bz]

		Returns:
			ratings: [bz]
		"""
		bz = user_indices.size()[0]

		U = self.user_embeddings(user_indices)
		I = self.item_embeddings(item_indices)
		A = self.aspect_embeddings.weight
		Ru = self.relation_embeddings.weight[0].repeat(bz, 1)
		Rip = self.relation_embeddings.weight[1].repeat(bz,1)
		Rin = self.relation_embeddings.weight[2].repeat(bz, 1)

		P = torch.softmax((U + Ru) @ A.T, dim=1)
		Qip = torch.softmax((I+Rip) @ A.T, dim=1)
		Qin = torch.softmax((I+Rin) @ A.T, dim=1)

		item_bias = self.item_bias(item_indices).squeeze()
		ratings = (P * Qip - P * Qin).sum(dim=1) + item_bias
		# print(I)
		# I.register_hook(lambda k: print(k))
		return ratings

	def predict(self, user_indices, item_indices):
		return self.rating_scores(user_indices, item_indices)


if __name__ == "__main__":
	configs = {"num_user": 8, "num_item": 32, "num_aspect": 14, "num_relation": 3, "latent_dim": 64, "use_TransE": 0, "uia_path": ""}
	simple_model = SimpleModel(configs)
	
	bz = 16
	num_user = configs["num_user"]
	num_item = configs["num_item"]
	user_indices = torch.randint(0, num_user, size=(bz,))
	item_indices = torch.randint(0, num_item, size=(bz,20))

	dists = simple_model(user_indices, item_indices)
	print(dists, dists.sum(dim=-1))


			
