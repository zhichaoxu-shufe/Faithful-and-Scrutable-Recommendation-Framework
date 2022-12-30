from functions import *
from dataset import *
from model import *
from evaluate import *
from engine import *

import argparse
import numpy as np
import pandas as pd
import random, sys, time, math, os, json, copy, logging
from tqdm import tqdm

import torch
from six.moves import xrange
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import normal_


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--dest_dir', type=str, required=True)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--num_users', type=int, required=True)
	parser.add_argument('--num_items', type=int, required=True)
	parser.add_argument('--num_aspect', type=int, required=True)
	parser.add_argument('--latent_dim', type=int, default=128)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--rating_lr', type=float, default=1e-3)
	parser.add_argument('--aspect_lr', type=float, default=1e-3)
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--momentum', type=float, default=5e-1)
	parser.add_argument('--l2', type=float, default=1e-4)
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--use_cuda', type=int, default=0)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--lambda_x', type=float, default=0)
	parser.add_argument('--lambda_y', type=float, default=0)
	parser.add_argument('--mlp_layer', type=int, default=3)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--cutoff', type=int, default=20)

	parser.add_argument('--gt_dir', type=str, default='/raid/brutusxu/agnostic/datasets/Electronics_5_5_3')

	args = parser.parse_args()

	# CDs
	# num_user 14178 num_item 15145 num_aspect 232
	# Kindle
	# num_user 6843 num_item 8581 num_aspect 77
	# Electronics
	# num_user 3151 num_item 3253 num_aspect 200

	config = vars(args)

	engine = EFMEngine(config)
	dataset = EFMDataset(args.dest_dir, args.dataset)

	# interact_hist = dataset.build_interaction_history()
	item_pool = set([i for i in range(args.num_items)])

	gt = np.load(os.path.join(args.gt_dir, 'test.pickle'), allow_pickle=True)
	train_loader = dataset.instance_a_train_loader(args.batch_size, True, 4)
	test_loader = dataset.instance_a_test_loader(1, False, 1)

	logging.basicConfig(filename=os.path.join(args.dest_dir, 'efm.log'), level=logging.INFO)
	logging.info('learning rate {}'.format(args.lr))
	logging.info('l2 regularization {}'.format(args.l2))
	logging.info('latent dim {}'.format(args.latent_dim))
	logging.info('alpha {}'.format(args.alpha))

	logging.info('\n')
	logging.info('train start')

	hr_holder = []
	checkpoint_holder = []

	ranklist = engine.output_ranklist_full(test_loader, dataset.history, item_pool)
	recall, precision, ndcg, hr, mrr = engine.evaluate(ranklist, gt, 20)
	# sys.exit()

	for epoch in range(args.epoch):
		print('Epoch ', epoch)
		epoch_loss = engine.train_an_epoch(train_loader, epoch_id=epoch)

		if epoch==0 or (epoch+1)%5 == 0:
			save_checkpoint(epoch, engine.model, args.dataset, args.dest_dir)
			logging.info('checkpoing saved')
			logging.info('\n')

		ranklist = engine.output_ranklist_full(test_loader, dataset.history, item_pool)
		recall, precision, ndcg, hr, mrr = engine.evaluate(ranklist, gt, 20)

		hr_holder.append(hr)
		checkpoint_holder.append(engine.model.state_dict())

		logging.info('epoch {}'.format(epoch))
		logging.info('epoch loss {}'.format(epoch_loss))
		logging.info('hit rate {}'.format(hr))
		logging.info('ndcg {}'.format(ndcg))
		logging.info('\n')

	logging.info('training finished')
	best_index = hr_holder.index(max(hr_holder))
	engine.model.load_state_dict(checkpoint_holder[best_index])

	ranklist = engine.output_ranklist_full(test_loader, dataset.history, item_pool)
	
	if not os.path.isdir(os.path.join(args.dest_dir, 'efm_ranklist')):
		os.mkdir(os.path.join(args.dest_dir, 'efm_ranklist'))

	with open(os.path.join(args.dest_dir, 'efm_ranklist/ranklist.json'), 'w') as fp:
		json.dump(ranklist, fp)

	item_scores = engine.output_item_scores(test_loader, dataset.history, item_pool)
	
	if not os.path.isdir(os.path.join(args.dest_dir, 'efm_item_scores')):
		os.mkdir(os.path.join(args.dest_dir, 'efm_item_scores'))

	with open(os.path.join(args.dest_dir, 'efm_item_scores/item_scores.json'), 'w') as fp:
		json.dump(item_scores, fp)