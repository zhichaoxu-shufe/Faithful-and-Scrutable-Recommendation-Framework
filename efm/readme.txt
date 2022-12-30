cmd to train efm
python efm/train_efm.py 
    --dataset Electronics 
    --dest_dir ./datasets/Electronics/ 
    --num_users 3151 
    --num_items 3253 
    --num_aspect 200
    --epoch {your_training_epochs, recommend >30}

cmd to generate aspects
python efm/aspect_reader.py
    --ranklist {path_to_ranklist.json}
    --checkpoint {path_to_efm_checkpoint}
    --data_dir ./datasets/Electronics
    --num_users 3151 
    --num_items 3253 
    --num_aspect 200