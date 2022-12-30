import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import time, sys, argparse, logging
# def save_model(model, optimizer, save_variable_list, args):


# def read_triplets(file_path):
# 	train_triplets = np.load(file_path+'/'+'train_triplets.pickle', allow_pickle=True).tolist()







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--input_dir', type=str)
	parser.add_argument('--output_dir', type=str)
	args = parser.parse_args()

	read_triplets(args.input_dir)
