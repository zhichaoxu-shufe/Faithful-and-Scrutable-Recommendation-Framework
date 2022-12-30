from dataset import *
from model import *
from evaluate import *
from engine import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, os, json, copy, logging
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default='Electronics')
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--pretrain_path', type=str, required=True)
	
	parser.add_argument('--num_user', type=int, default=3151)
	parser.add_argument('--num_item', type=int, default=3253)
	parser.add_argument('--num_aspect', type=int, default=200)

	parser.add_argument('--epoch', type=int, default=1)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--lr', type=float, default=5e-5)
	parser.add_argument('--l2', type=float, default=0)
	parser.add_argument('--momentum', type=float, default=0)
	parser.add_argument('--use_cuda', type=int, default=1)
	parser.add_argument('--epsilon', type=float, default=8.)
	parser.add_argument('--num_negative', type=int, default=7)
	parser.add_argument('--alpha', type=float, default=0.5)

	parser.add_argument('--ranklist_per_epoch', type=int, default=1)

	args = parser.parse_args()
	config = vars(args)

	logging.basicConfig(filename=os.path.join(args.input_dir, 'a2cf.log'), level=logging.INFO)
	logging.info('learning rate {}'.format(args.lr))
	logging.info('l2 regularization {}'.format(args.l2))
	logging.info('latent dim {}'.format(args.latent_dim))
	logging.info('\n')
	logging.info('train start')

	engine = A2CFEngine(config)
	dataset = A2CFDataset(args.input_dir, args.dataset)
	gt = np.load(os.path.join(args.input_dir, 'test.pickle'), allow_pickle=True)
	item_pool = set([i for i in range(args.num_item)])

	interact_hist = dataset.build_interaction_history()

	test_loader = dataset.instance_a_test_loader(1, True, 4)

	print('initialization finished')
	ranklist = engine.output_ranklist(test_loader, dataset.history, item_pool)
	recall, precision, ndcg, hr, mrr = engine.evaluate(ranklist, gt, 20)
	
	hr_holder = []
	checkpoint_holder = []

	for epoch in range(args.epoch):
		print('Epoch {}'.format(epoch+1))
		train_loader = dataset.instance_pack_train_loader(1, True, 4, args.num_negative, item_pool)
		epoch_loss = engine.train_batch_epoch(train_loader, epoch)
		if (epoch+1) % args.ranklist_per_epoch == 0:
			ranklist = engine.output_ranklist(test_loader, dataset.history, item_pool)
			recall, precision, ndcg, hr, mrr = engine.evaluate(ranklist, gt, 20)
			hr_holder.append(hr)
			checkpoint_holder.append(engine.model.state_dict())

			logging.info('Epoch {}'.format(epoch))
			logging.info('epoch loss {}'.format(epoch_loss))
			logging.info('hit rate {}'.format(hr))
			logging.info('ndcg {}'.format(ndcg))
			logging.info('\n')
	logging.info('training finished')
	
	best_index = hr_holder.index(max(hr_holder))
	engine.model.load_state_dict(checkpoint_holder[best_index])
	save_checkpoint(best_index+1, engine.model, args.dataset, args.input_dir)

	if not os.path.isdir(os.path.join(args.input_dir, 'a2cf_ranklist')):
		os.mkdir(os.path.join(args.input_dir, 'a2cf_ranklist'))

	ranklist = engine.output_ranklist(test_loader, dataset.history, item_pool)

	logging.info('output_item_scores')
	engine.output_item_scores(test_loader)
