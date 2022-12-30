import json
import os
from math import log
from six.moves import xrange
import pandas as pd
import numpy as np
import time
import argparse

# compute ndcg
def metrics(doc_list, rel_set):
	dcg = 0.0
	hit_num = 0.0
	reciprocal_rank = 0.0

	for i in xrange(len(doc_list)):
		if doc_list[i] in rel_set:
			# dcg
			dcg += 1/(log(i+2)/log(2))
			hit_num += 1

	for i in xrange(len(doc_list)):
		if doc_list[i] in rel_set:
			reciprocal_rank = 1/(i+1)
			break

	# idcg
	idcg = 0.0
	for i in xrange(min(len(rel_set), len(doc_list))):
		idcg += 1/(log(i+2)/log(2))
	ndcg = dcg/idcg
	recall = hit_num/len(rel_set)
	precision = hit_num/len(doc_list)
	hit = 1.0 if hit_num > 0 else 0.0
	large_rel = 1.0 if len(rel_set)>len(doc_list) else 0.0

	return recall, ndcg, hit, large_rel, precision, reciprocal_rank


def print_metrics_with_rank_cutoff(ranklist, qrel_map, rank_cutoff):
	ndcgs = 0.0
	recalls = 0.0
	hits = 0.0
	large_rels = 0.0
	precisions = 0.0
	count_query = 0
	reciprocal_ranks = 0.0
	for qid in ranklist.keys():
		if qid in qrel_map.keys():
			recall, ndcg, hit, large_rel, precision, reciprocal_rank = metrics(ranklist[qid][:rank_cutoff], qrel_map[qid])
			count_query += 1
			ndcgs += ndcg
			recalls += recall
			hits += hit
			large_rels += large_rel
			precisions += precision
			reciprocal_ranks += reciprocal_rank

	# print("Query Number:" + str(count_query))
	# print("Larger_rel_set@"+str(rank_cutoff) + ":" + str(large_rels/count_query))
	# print("Recall@"+str(rank_cutoff) + ":" + str(recalls/count_query))
	# print("Precision@"+str(rank_cutoff) + ":" + str(precisions/count_query))
	
	print('Hit@{}: {}'.format(rank_cutoff, hits/count_query))
	print('NDCG@{}: {}'.format(rank_cutoff, ndcgs/count_query))

	# print('Reciprocal Rank {}'.format(reciprocal_ranks/count_query))
	return recalls/count_query, precisions/count_query, ndcgs/count_query, hits/count_query, reciprocal_ranks/count_query


if __name__ == "__main__":

	# read ranklist file
	parser = argparse.ArgumentParser()
	parser.add_argument('--gt', type=str, required=True)
	parser.add_argument('--ranklist', type=str, required=True)
	parser.add_argument('--cutoff', type=int, default=20)

	args = parser.parse_args()
	with open(args.ranklist, 'r') as f:
		ranklist = json.load(f)

	testset = np.load(args.gt, allow_pickle=True)
	testset = testset.tolist()
	# print(testset)
	qrel_map = {}
	for entry in testset:
		qrel_map[entry[0]] = entry[1]

	reformed = {}
	for key in qrel_map.keys():
		key = int(key)
		reformed[str(key)]=[]
		for value in qrel_map[key]:
			value = int(value)
			reformed[str(key)].append(str(value))

	qrel_map = reformed
	# print(ranklist)
	print_metrics_with_rank_cutoff(args.cutoff)
