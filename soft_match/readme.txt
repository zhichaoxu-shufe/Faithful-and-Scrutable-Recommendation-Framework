cmd to run bbx-kl mode and evaluate its performance
CUDA_VISIBLE_DEVICES=1 python soft_match/train.py 
    --dataset electronics 
    --data_dir datasets/Electronics/ 
    --item_scores_dir datasets/Electronics/a2cf_item_scores 
    --use_kl 1 
    --lr 2e-3 
    --epoch 20

cmd to run softmax gt
python soft_match/train.py
    --dataset electronics
    --data_dir datasets/Electronics/
    --item_scores_dir datasets/Electronics/a2cf_item_scores 
    --use_gt 1
    --lr 2e-5
    --epoch 20

cmd to generate aspect file
python soft_match/aspect_reader.py
    --ranklist {ranklist.json}
    --checkpoint {checkpoint.pt}
    --data_dir datasets/Electronics/
    --check_wbx_perf 1

cmd to evaluate aspect overlap
python aspect_overlap.py:
    --bbx_asp_fn {aspect_from_bbx.pickle}
    --wbx_asp_fn {aspect_from_wbx.pickle}