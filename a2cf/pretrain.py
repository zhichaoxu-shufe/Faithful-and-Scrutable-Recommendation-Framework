import numpy as np
from tqdm import tqdm
import os, sys, time
import copy
import argparse

from torch.utils.data import DataLoader
import torch

from dataset import *
from model import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--epoch', type=int, default=50)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--l2', type=float, default=0)

	args = parser.parse_args()

	user_matrix_path = os.path.join(args.input_dir, 'user_attention_matrix.pickle')
	item_matrix_path = os.path.join(args.input_dir, 'item_quality_matrix.pickle')
	X = np.load(user_matrix_path, allow_pickle=True)
	Y = np.load(item_matrix_path, allow_pickle=True)

	num_user = X.shape[0]
	num_item = Y.shape[0]
	num_aspect = X.shape[1]
	if torch.cuda.is_available():
		device = torch.device('cuda')

	u_f_pairs = np.transpose(X.nonzero())
	i_f_pairs = np.transpose(Y.nonzero())

	u_f_train_loader = DataLoader(dataset=A2CFFillMatrixDataset(u_f_pairs), batch_size=128, shuffle=True)
	i_f_train_loader = DataLoader(dataset=A2CFFillMatrixDataset(i_f_pairs), batch_size=128, shuffle=True)

	model = EmbeddingNet(torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device), num_user, num_item, num_aspect, args.latent_dim)
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
	crit = torch.nn.MSELoss()

	if not os.path.isdir(os.path.join(args.input_dir, 'a2cf_logs')):
		os.mkdir(os.path.join(args.input_dir, 'a2cf_logs'))

	best_loss = np.inf
	best_epoch_num = 0
	best_model_wts = copy.deepcopy(model.state_dict())

	model.train()

	mode = 'u_f'
	losses = []
	for u_f_pair in u_f_train_loader:
		u_f_pair = u_f_pair.to(device)
		# print(u_f_pair.type())
		pre_score, gt_score = model(u_f_pair, mode)
		loss = crit(pre_score, gt_score)
		losses.append(loss.cpu().item())
	ave_train = np.mean(np.array(losses))
	print('init user training loss: ', ave_train)

	mode = 'i_f'
	losses = []
	for i_f_pair in i_f_train_loader:
		i_f_pair = i_f_pair.to(device)
		pre_score, gt_score = model(i_f_pair, mode)
		loss = crit(pre_score, gt_score)
		losses.append(loss.cpu().item())
	ave_train = np.mean(np.array(losses))
	print('init item training loss: ',ave_train)

	pbar=tqdm(args.epoch)
	for epoch in range(args.epoch):
		current_train_loss = 0
		model.train()
		optimizer.zero_grad()

		# train user, aspect embedding
		mode = 'u_f'
		losses = []
		for u_f_pair in u_f_train_loader:
			u_f_pair.to(device)
			pre_score, gt_score = model(u_f_pair, mode)
			# print(pre_score.type(), gt_score.type())
			# sys.exit()
			loss = crit(pre_score, gt_score)
			loss.backward()
			
			# gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			losses.append(loss.cpu().item())
			
			optimizer.step()
			optimizer.zero_grad()
		ave_train = np.mean(np.array(losses))
		current_train_loss += ave_train
		print('epoch {} user training loss: {}'.format(epoch, ave_train))

		# train item, aspect embedding
		mode = 'i_f'
		losses = []
		for i_f_pair in i_f_train_loader:
			i_f_pair.to(device)
			pre_score, gt_score = model(i_f_pair, mode)
			loss = crit(pre_score, gt_score)
			loss.backward()

			gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			losses.append(loss.cpu().item())

			optimizer.step()
			optimizer.zero_grad()
		ave_train = np.mean(np.array(losses))
		current_train_loss += ave_train
		print('epoch {} item training loss: {}'.format(epoch, ave_train))

		if current_train_loss < best_loss:
			best_loss = current_train_loss
			best_model_wts = copy.deepcopy(model.state_dict())
			best_epoch_num = epoch

	print('best epoch num： {}'.format(best_epoch_num))
	model.load_state_dict(best_model_wts)

	if not os.path.isdir(os.path.join(args.input_dir, 'a2cf_pretrain')):
		os.mkdir(os.path.join(args.input_dir, 'a2cf_pretrain'))
	torch.save(model.state_dict(), os.path.join(args.input_dir, 'a2cf_pretrain', 'pretrain.pt'))

	# fill user attention matrix
	zero_indices = np.where(X == 0)
	pre_scores = np.array([])
	pairs = np.concatenate((np.expand_dims(zero_indices[0], axis=1), np.expand_dims(zero_indices[1], axis=1)), axis=1)
	u_f_pre_loader = DataLoader(dataset=A2CFFillMatrixDataset(pairs), batch_size=10000, shuffle=False)
	for batch in tqdm(u_f_pre_loader):
		pre_score, _ = model(batch, mode='u_f')
		pre_score = pre_score.cpu().detach().numpy().squeeze()
		pre_scores = np.concatenate((pre_scores, pre_score), axis=0)
	X[zero_indices] = pre_scores

	# fill item quality matrix
	zero_indices = np.where(Y == 0)
	pre_scores = np.array([])
	pairs = np.concatenate((np.expand_dims(zero_indices[0], axis=1), np.expand_dims(zero_indices[1], axis=1)), axis=1)
	i_f_pre_loader = DataLoader(dataset=A2CFFillMatrixDataset(pairs), batch_size=10000, shuffle=False)
	for batch in tqdm(i_f_pre_loader):
		pre_score, _ = model(batch, mode='i_f')
		pre_score = pre_score.cpu().detach().numpy().squeeze()
		pre_scores = np.concatenate((pre_scores, pre_score), axis=0)
	Y[zero_indices] = pre_scores

	X.dump(os.path.join(args.input_dir, 'a2cf_user_matrix_pretrained.pickle'))
	Y.dump(os.path.join(args.input_dir, 'a2cf_item_matrix_pretrained.pickle'))


