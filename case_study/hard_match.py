import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import pickle
np.set_printoptions(precision=3, suppress=True)

class HardMatchModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.num_user = config['num_user']
		self.num_item = config['num_item']
		self.num_aspect = config['num_aspect']
		self.latent_dim = config['latent_dim']
		self.train_uif_path = config['train_uif_path']
		self.device = torch.device("cuda") if config["use_cuda"] else torch.device("cpu")
		self.meta_weights = [float(_) for _ in config["meta_weights"].split(",")]
		assert sum(self.meta_weights) == 1 
		print("Meta Weights: ", self.meta_weights)

		# start 
		self._build_mask_mtx()
		self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
		self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
		self.aspect_embeddings = nn.Embedding(self.num_aspect, self.latent_dim)

		self.ua_embeddings = nn.Embedding(self.num_user, self.num_aspect)
		self.ia_embeddings = nn.Embedding(self.num_item, self.num_aspect)

		self.item_bias = nn.Embedding(self.num_item, 1)

	def _build_mask_mtx(self):
		with open(self.train_uif_path, "rb") as f:
			uif_data = pickle.load(f)
		ua_mask_mtx = torch.zeros(self.num_user, self.num_aspect)
		ia_mask_mtx = torch.zeros(self.num_item, self.num_aspect)

		for (u,i,a) in uif_data:
			ua_mask_mtx[u, a] = 1
			ia_mask_mtx[i, a] = 1
		
		self.register_buffer("ua_mask_mtx", ua_mask_mtx)
		self.register_buffer("ia_mask_mtx", ia_mask_mtx)

	def full_compute(self):
		"""
		Returns:
			ui_ratings: [num_user, num_item]. 
		"""
		user_embeddings = self.user_embeddings.weight 
		item_embeddings = self.item_embeddings.weight
		aspect_embeddings = self.aspect_embeddings.weight

		#ua_scores = torch.sigmoid(user_embeddings @ aspect_embeddings.T) * self.ua_mask_mtx
		#ia_scores = torch.sigmoid(item_embeddings @ aspect_embeddings.T) * self.ia_mask_mtx
		ua_scores = torch.sigmoid(self.ua_embeddings.weight) * self.ua_mask_mtx
		ia_scores = torch.sigmoid(self.ia_embeddings.weight) * self.ia_mask_mtx
		
		# 2-hop 
		meta_1 = F.softmax(ua_scores @ ia_scores.T, dim=1)

		# 4-hop 
		meta_2 = F.softmax(ua_scores @ ia_scores.T @ ia_scores @ ia_scores.T, dim=1)
		meta_3 =  F.softmax(ua_scores @ ua_scores.T @ ua_scores @ ia_scores.T, dim=1)

		ui_ratings = meta_1 * self.meta_weights[0] + meta_2 * self.meta_weights[1] + meta_3 * self.meta_weights[2] 

		return ui_ratings

	def batch_compute(self, user_indices):
		"""
		Args:
			user_indices: [bz]
		Returns:
			item_scores: [bz, num_item]
		"""
		ui_ratings = self.full_compute()
		item_scores = torch.index_select(ui_ratings, dim=0, index=user_indices)

		return item_scores

	def unscaled_full_compute(self):
		"""
		Returns:
			ui_logits: [num_user, num_item]
		"""
		user_embeddings = self.user_embeddings.weight 
		item_embeddings = self.item_embeddings.weight
		aspect_embeddings = self.aspect_embeddings.weight

		ua_scores = torch.sigmoid(self.ua_embeddings.weight) * self.ua_mask_mtx
		ia_scores = torch.sigmoid(self.ia_embeddings.weight) * self.ia_mask_mtx

		meta_1 = ua_scores @ ia_scores.T
		meta_2 = ua_scores @ ia_scores.T @ ia_scores @ ia_scores.T
		meta_3 = ua_scores @ ua_scores.T @ ua_scores @ ia_scores.T

		ui_logits = meta_1 * self.meta_weights[0] + meta_2 * self.meta_weights[1] + meta_3 * self.meta_weights[2] 

		return ui_logits

	def unscaled_batch_compute(self, user_indices):
		"""
		Unscaled version of 'batch_compute'
		"""
		ui_logits = self.unscaled_full_compute()
		return torch.index_select(ui_logits, dim=0, index=user_indices)


	def bundle_compute(self, user_indices, item_indices):
		"""
		Returns:
			user_indices: [bz, 1+num_neg]
			item_indices: [bz, 1+num_neg]

		Args:
			item_scores: [bz, 1+num_neg]
		"""
		bz, k = user_indices.size()
		user_indices = user_indices.view(-1)
		item_indices = item_indices.view(-1)

		item_scores = self.full_compute()[user_indices, item_indices]
		item_scores = item_scores.view(bz, k)

		return item_scores

	def bundle_ui_logit(self, user_indices, item_indices):
		"""
		Args:
			user_indices: [bz, k]
			item_indices: [bz, k]
		
		Returns: 
			logits: [bz, k]
		"""
		bz, k = user_indices.size()

		user_indices = user_indices.view(-1)
		item_indices = item_indices.view(-1)
		logits = self.unscaled_full_compute()[user_indices, item_indices] 
		logits = logits.view(bz, k)

		return logits

	def batch_pairwise_compute(self, user_indices, item_indices):
		"""
		Args:
			user_indices: [bz]
			item_indices: [bz, num_sampled]
		
		Returns:
			item_scores: [bz, num_sampled]
		"""
		bz, num_sampled = item_indices.size()
		user_indices = user_indices.view(-1, 1).repeat(1, num_sampled) #[bz, num_sampled]
		user_indices = user_indices.view(-1)
		item_indices = item_indices.view(-1)

		ui_ratings = self.full_compute()
		item_scores = ui_ratings[user_indices, item_indices]
		item_scores = item_scores.view(bz, num_sampled)

		return item_scores

	def forward(self, user_indices, item_indices):
		return self.batch_pairwise_compute(user_indices, item_indices)

	def predict(self, user_indices, item_indices):
		"""
		Args:
			user_indices: [bz]
			item_indices: [bz]
		Returns:
			scores: [bz]
		"""
		return self.full_compute()[user_indices, item_indices]