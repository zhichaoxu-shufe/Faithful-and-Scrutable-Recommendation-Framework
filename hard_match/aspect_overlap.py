import numpy as np
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbx_asp_fn', type=str, required=True)
    parser.add_argument('--wbx_asp_fn', type=str, required=True)
    args = parser.parse_args()
	#blackbox_aspect = np.load('datasets/Electronics_5_5_3/common_aspects/efm_top_aspects.pickle', allow_pickle=True)
	#whitebox_aspect = np.load('datasets/Electronics_5_5_3/common_aspects/best_sm_wbx_slice_kl.pickle', allow_pickle=True)
    
    blackbox_aspect = np.load(args.bbx_asp_fn, allow_pickle=True)
    whitebox_aspect = np.load(args.wbx_asp_fn, allow_pickle=True)

    blackbox_holder = {}
    whitebox_holder = {}
    for i, row in enumerate(blackbox_aspect):
        if row[0] not in blackbox_holder.keys():
            blackbox_holder[row[0]] = set()
        blackbox_holder[row[0]].add(row[1])

    for i, row in enumerate(whitebox_aspect):
        if row[0] not in whitebox_holder.keys():
            whitebox_holder[row[0]] = set()
        whitebox_holder[row[0]].add(row[1])

    count = 0
    for u in blackbox_holder.keys():
        bbx = blackbox_holder[u]
        wbx = whitebox_holder[u]
        count += len(bbx.intersection(wbx))

    print('overlapped items: ', count/len(blackbox_holder))

    blackbox_holder = {}
    whitebox_holder = {}
    for i, row in enumerate(blackbox_aspect):
        if row[0] not in blackbox_holder.keys():
            blackbox_holder[row[0]] = set()
        for aspect in row[2]:
            blackbox_holder[row[0]].add(aspect)

    for i, row in enumerate(whitebox_aspect):
        if row[0] not in whitebox_holder.keys():
            whitebox_holder[row[0]] = set()
        for aspect in row[2]:
            whitebox_holder[row[0]].add(aspect)

    count = 0
    for u in blackbox_holder.keys():
        bbx = blackbox_holder[u]
        wbx = whitebox_holder[u]
        count += len(bbx.intersection(wbx))

    print('overlapped aspects: ', count/len(blackbox_holder))

    blackbox_holder = {}
    whitebox_holder = {}
    for i, row in enumerate(blackbox_aspect):
        blackbox_holder[(row[0], row[1])] = row[2]
    for i, row in enumerate(whitebox_aspect):
        whitebox_holder[(row[0], row[1])] = row[2]

    count = 0
    hit = 0
    for k, v in blackbox_holder.items():
        if k in whitebox_holder.keys():
            hit += 1
            overlapped_item_aspects = set(blackbox_holder[k]).intersection(set(whitebox_holder[k]))
            # overlapped_item_aspects
            count += len(overlapped_item_aspects)
    print('overlapped aspects per hit item: ', count/hit)
