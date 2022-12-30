from functions import *
from engine import *
from dataset import AgnosticDataset
from evaluate import print_metrics_with_rank_cutoff
from hard_match_model import HardMatchModel

import random, sys, time
import argparse
import copy
import json
from tqdm import tqdm
import os
import logging
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_

np.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--data_dir', type=str, required=True)
	parser.add_argument('--item_scores_dir', type=str, required=True)
	parser.add_argument('--save_checkpoint_and_score', type=int, default=1)

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_user', type=int, default=3151)
	parser.add_argument('--num_item', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)
	parser.add_argument('--num_relation', type=int, default=3)
	parser.add_argument('--use_TransE', type=int, default=0)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--momentum', type=float, default=5e-1)
	parser.add_argument('--l2', type=float, default=0)
	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--use_cuda', type=int, default=1)
	parser.add_argument('--uia_path', type=str, default=" ")

	parser.add_argument('--use_gt', type=int, default=0)
	parser.add_argument('--use_kl', type=int, default=0)
	parser.add_argument('--num_neg', type=int, default=9)
	parser.add_argument('--use_temperature', type=int, default=0)
	parser.add_argument('--temperature', type=float, default=5e-4)

	# hard match model parameters
	parser.add_argument('--black_box_output', type=int, default=0)	
	parser.add_argument('--meta_weights', type=str, default='0.4,0.3,0.3')

	parser.add_argument('--model_name', type=str, default='')

	args = parser.parse_args()
	config = vars(args)

	logging.basicConfig(filename=os.path.join(args.data_dir, 'hardmatch.log'), level=logging.INFO)
	logging.info('learning rate {}'.format(args.lr))
	logging.info('l2 regularization {}'.format(args.l2))
	logging.info('latent dim {}'.format(args.latent_dim))
	logging.info('use gt {}'.format(args.use_gt))
	logging.info('\n')
	logging.info('train start')

	engine = HMEngine(config)
	dataset = AgnosticDataset(args.data_dir, args.dataset, config)
	interact_hist = dataset.interact_hist
	item_pool = set([i for i in range(args.num_item)])
	gt = np.load(os.path.join(args.data_dir, 'test.pickle'), allow_pickle=True)


	if args.use_gt:
		args.model_name += "gt"
		train_loader = dataset.instance_gt_train_loader(args.batch_size, shuffle=True, num_workers=4, item_pool=item_pool)
	if args.use_kl:
		args.model_name += "bbx"
		print(20*"-", "use kl train bbx train loader", 20*"-")
		train_loader = dataset.instance_kl_bbx_train_loader(args.batch_size, shuffle=True, num_workers=4)


	test_loader = dataset.instance_a_test_loader(args.batch_size, shuffle=False, num_workers=4)
	
	hr_holder, checkpoint_holder = [], []
	if args.use_gt:
		args.model_name += "_softmax"
	if args.use_kl:
		print(20*"-", "hard-match-kl model", 20*"-")
		args.model_name += "_kl"

	for epoch in range(args.epoch):
		if args.use_gt:
			epoch_loss = engine.train_hard_match_softmax_epoch(train_loader, epoch_id=epoch)
		elif args.use_kl:
			epoch_loss = engine.train_hard_match_kl_epoch(train_loader, epoch_id=epoch)
		ranklist = engine.output_ranklist(test_loader, interact_hist, item_pool)
		recall, precision, ndcg, hr, mrr = engine.evaluate(ranklist, gt, 20)

		hr_holder.append(hr)
		checkpoint_holder.append(engine.model.state_dict())
		logging.info('Epoch {}'.format(epoch))
		logging.info('epoch loss {:.4f}'.format(epoch_loss))
		logging.info('hit rate {:.4f}'.format(hr))
		logging.info('ndcg {:.4f}'.format(ndcg))
		logging.info('\n')

	logging.info('training finished')
	best_index = hr_holder.index(max(hr_holder))
	engine.model.load_state_dict(checkpoint_holder[best_index])

	if args.save_checkpoint_and_score:
		args.model_name = "best_" + args.model_name
		save_best_checkpoint(engine.model, args.data_dir, args.model_name)
		ranklist = engine.output_ranklist(test_loader, interact_hist, item_pool)
		item_scores = engine.output_item_scores(test_loader, dataset.interact_hist, item_pool)

		if not os.path.isdir(os.path.join(args.data_dir, args.model_name+'_item_scores')):
			os.mkdir(os.path.join(args.data_dir, args.model_name+'_item_scores'))
		if not os.path.isdir(os.path.join(args.data_dir, args.model_name+'_ranklist')):
			os.mkdir(os.path.join(args.data_dir, args.model_name+'_ranklist'))

		save_ranklist(ranklist, os.path.join(args.data_dir, args.model_name+'_ranklist'))
		save_item_scores(item_scores, os.path.join(args.data_dir, args.model_name+'_ranklist'))