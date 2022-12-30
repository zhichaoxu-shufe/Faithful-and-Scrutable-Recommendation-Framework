cmd to train a2cf

1. pretrain a2cf matrix
python a2cf/pretrain.py
    --input_dir ./datasets/Electronics/
    --epoch {your_training_epochs, recommend>50}

2. train a2cf model
python a2cf/train_a2cf.py 
    --input_dir ./datasets/Electronics/ 
    --epoch {your_training_epochs, recommend>5}

cmd to generate aspects
python a2cf/aspect_reader.py
    --ranklist
    --checkpoint
    --input_dir
    --num_user
    --num_item
    --num_aspect

