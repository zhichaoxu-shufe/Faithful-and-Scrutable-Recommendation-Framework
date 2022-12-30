cmd to run softmax gt:
python hard_match/train.py 
    --dataset Electronics 
    --data_dir datasets/Electronics_5_5_3 
    --item_scores_dir datasets/Electronics_5_5_3/efm_item_scores
    --use_gt 1 
    --use_temperature 1 
    --lr 2e-5 
    --epoch 20

-------------------------------------------------------------------------------------------------------

cmd to run kl bbx mode:
python hard_match/train.py 
    --dataset Electronics 
    --data_dir datasets/Electronics_5_5_3 
    --item_scores_dir datasets/Electronics_5_5_3/efm_item_scores 
    --lr 2e-3 
    --epoch 150 
    --batch_size 32 
    --black_box_output 1 
    --use_kl 1

-------------------------------------------------------------------------------------------------------

cmd to run bbx-kl mode and evaluate its performance:
(1) Train model:
python hard_match/train.py 
    --dataset Electronics 
    --data_dir datasets/Electronics_5_5_3 
    --item_scores_dir datasets/Electronics_5_5_3/efm_item_scores 
    --lr 5e-3 
    --epoch 25 
    --batch_size 32 
    --black_box_output 1 
    --use_kl 1 


(2) generate aspect file:
python hard_match/aspect_reader.py 
    --ranklist datasets/Electronics_5_5_3/hardmatch_ranklist/best_bbx_kl.json 
    --checkpoint datasets/Electronics_5_5_3/best_bbx_kl_checkpoints/wbx_epoch_24.pt 
    --data_dir datasets/Electronics_5_5_3 
    --check_wbx_perf 1

(3) evluate performance
python aspect_overlap.py:
    --bbx_asp_fn {aspect_from_bbx.pickle}
    --wbx_asp_fn {aspect_from_wbx.pickle}


-------------------------------------------------------------------------------------------------------

cmd to run gt-softmax mode and evaluate its performance:

(1) Train model:
python hard_match/train.py 
    --dataset Electronics 
    --data_dir datasets/Electronics_5_5_3 
    --item_scores_dir datasets/Electronics_5_5_3/efm_item_scores 
    --use_gt 1 
    --use_temperature 1 
    --lr 2e-5 
    --epoch 15

(2) generate aspect file
python hard_match/aspect_reader.py 
    --ranklist datasets/Electronics_5_5_3/hardmatch_ranklist/best_gt_softmax.json 
    --checkpoint datasets/Electronics_5_5_3/best_gt_softmax_checkpoints/wbx_epoch_13.pt 
    --data_dir datasets/Electronics_5_5_3

(3) evaluate performance
python aspect_overlap.py:
    --bbx_asp_fn {aspect_from_bbx.pickle}
    --wbx_asp_fn {aspect_from_wbx.pickle}
