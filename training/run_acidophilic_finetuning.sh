#!/bin/bash

# Path to the extracted acidophilic dataset
DATASET_PATH="/tmp/acidophilic_data/acidophilic_dataset_for_aaron_converted"

# Path to a pre-trained model to use as starting point (optional)
# PRETRAINED_MODEL="/home/maiwald/ProteinMPNN/vanilla_model_weights/v_48_020.pt"

# Run finetuning
python finetuning.py \
  --path_for_training_data $DATASET_PATH \
  --path_for_outputs "./exp_acidophilic_%Y%m%d_%H%M%S" \
  --num_epochs 100 \
  --batch_size 1000 \
  --num_examples_per_epoch 50 \
  --max_protein_length 1000 \
  --hidden_dim 128 \
  --num_encoder_layers 3 \
  --num_neighbors 48 \
  --dropout 0.1 \
  --backbone_noise 0.2 \
  --mixed_precision \
  --debug
  
# Remove --debug flag for full training