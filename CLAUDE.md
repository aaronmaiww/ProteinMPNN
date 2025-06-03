# ProteinMPNN Usage Guide

## Acidophilic Dataset Finetuning

### Dataset Structure
The acidophilic dataset is organized as follows:
```
acidophilic_dataset_converted.zip
├── acidophilic_dataset_for_aaron_converted/
│   ├── train_proteins.txt       # List of protein IDs for training
│   ├── validation_proteins.txt  # List of protein IDs for validation
│   ├── test_proteins.txt        # List of protein IDs for testing
│   ├── train/                   # Training data directory
│   │   └── pdb/                 # PDB files organized by first two letters of ID
│   │       └── XX/              # First two letters of protein ID
│   │           └── XXXXX/       # Protein ID directory
│   │               ├── XXXXX.pt       # Metadata file
│   │               └── XXXXX_A.pt     # Chain data file
│   ├── validation/              # Validation data directory (same structure)
│   └── test/                    # Test data directory (same structure)
```

### How to Run Finetuning
```bash
# Extract the dataset
mkdir -p /tmp/acidophilic_data
unzip /home/maiwald/ProteinMPNN/acidophilic_dataset_converted.zip -d /tmp/acidophilic_data

# Run finetuning with debug mode (fast, for testing)
cd /home/maiwald/ProteinMPNN/training
./run_acidophilic_finetuning.sh

# For full training, edit the script to remove the --debug flag
```

### Key Modifications for Acidophilic Dataset
1. Custom training clusters building from protein ID lists
2. Custom loader for the acidophilic dataset directory structure
3. Special handling for the tensor shapes in the loader
4. Modified data processing pipeline to work with the acidophilic data format

### Example Command for Full Training
```bash
python finetuning.py \
  --path_for_training_data /path/to/acidophilic_dataset_for_aaron_converted \
  --path_for_outputs "./exp_acidophilic_%Y%m%d_%H%M%S" \
  --num_epochs 100 \
  --batch_size 1000 \
  --num_examples_per_epoch 1000 \
  --max_protein_length 1000 \
  --hidden_dim 128 \
  --num_encoder_layers 3 \
  --num_neighbors 48 \
  --dropout 0.1 \
  --backbone_noise 0.2 \
  --mixed_precision
```