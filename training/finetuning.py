import argparse
import os.path

# Custom utilities for acidophilic dataset
def build_custom_training_clusters(params, debug=False):
    """Build training clusters from protein lists instead of list.csv"""
    # Read protein lists
    import os
    import numpy as np
    
    with open(params['VAL'], 'r') as f:
        val_proteins = [line.strip() for line in f.readlines()]
    
    with open(params['TEST'], 'r') as f:
        test_proteins = [line.strip() for line in f.readlines()]
    
    train_file = os.path.join(params['DIR'], 'train_proteins.txt')
    with open(train_file, 'r') as f:
        train_proteins = [line.strip() for line in f.readlines()]
        
    print(f"Found {len(train_proteins)} train proteins, {len(val_proteins)} validation proteins, {len(test_proteins)} test proteins")
    
    # Create dictionaries in the expected format
    train = {}
    valid = {}
    test = {}
    
    # For train proteins, add them to train dict
    for protein in train_proteins:
        if debug and len(train) >= 20:  # Limit in debug mode
            break
        train[len(train)] = [[f"{protein}_A", protein]]
        
    # For validation proteins, add them to valid dict
    for protein in val_proteins:
        if debug and len(valid) >= 10:  # Limit in debug mode
            break
        valid[len(valid)] = [[f"{protein}_A", protein]]
        
    # For test proteins, add them to test dict
    for protein in test_proteins:
        if debug and len(test) >= 10:  # Limit in debug mode
            break
        test[len(test)] = [[f"{protein}_A", protein]]
        
    if debug:
        print("Train sample:", list(train.items())[:5])
        print("Valid sample:", list(valid.items())[:5])
        
    return train, valid, test

def custom_loader_pdb(item, params):
    """Custom loader for acidophilic dataset structure"""
    import os
    import torch
    import numpy as np
    
    pdbid, chid = item[0].split('_')
    
    # Determine if it's a train, validation, or test protein
    dataset_type = None
    train_file = os.path.join(params['DIR'], 'train_proteins.txt')
    val_file = os.path.join(params['DIR'], 'validation_proteins.txt')
    test_file = os.path.join(params['DIR'], 'test_proteins.txt')
    
    with open(train_file, 'r') as f:
        train_proteins = [line.strip() for line in f.readlines()]
    with open(val_file, 'r') as f:
        val_proteins = [line.strip() for line in f.readlines()]
    with open(test_file, 'r') as f:
        test_proteins = [line.strip() for line in f.readlines()]
    
    if pdbid in train_proteins:
        dataset_type = "train"
    elif pdbid in val_proteins:
        dataset_type = "validation"
    elif pdbid in test_proteins:
        dataset_type = "test"
    else:
        print(f"Protein {pdbid} not found in any dataset!")
        return {'seq': np.zeros(5)}
    
    # Extract first two characters for directory structure
    first_two = pdbid[:2]
    
    PREFIX = f"{params['DIR']}/{dataset_type}/pdb/{first_two}/{pdbid}/{pdbid}"
    
    # Check if file exists
    if not os.path.isfile(f"{PREFIX}.pt"):
        print(f"File not found: {PREFIX}.pt")
        return {'seq': np.zeros(5)}
    
    # Load metadata
    meta = torch.load(f"{PREFIX}.pt")
    
    # Check if chain file exists
    chain_file = f"{PREFIX}_{chid}.pt"
    if not os.path.isfile(chain_file):
        print(f"Chain file not found: {chain_file}")
        return {'seq': np.zeros(5)}
    
    # Load chain data
    chain = torch.load(chain_file)
    L = len(chain['seq'])
    
    # Return data in the expected format
    return {'seq': chain['seq'],
            'xyz': chain['xyz'],
            'idx': torch.zeros(L).int(),
            'masked': torch.Tensor([0]).int(),
            'label': item[0]}

def load_data_from_loader(data_loader, max_length, num_examples):
    """Load data directly from loader_pdb output and convert to expected format"""
    pdb_dict_list = []
    count = 0
    
    for batch in data_loader:
        # Handle the batch format - extract values from lists
        # batch contains lists because DataLoader wraps everything in batches
        
        # Extract single values from batch lists
        label = batch['label'][0] if isinstance(batch['label'], list) else batch['label']
        seq = batch['seq'][0] if isinstance(batch['seq'], list) else batch['seq']
        xyz = batch['xyz']
        idx = batch['idx'][0] if batch['idx'].dim() > 1 else batch['idx']
        masked = batch['masked'][0] if batch['masked'].dim() > 1 else batch['masked']
        
        print(f"DEBUG: Processing {label}, seq_len={len(seq)}, xyz_shape={xyz.shape}")
        
        if len(seq) <= max_length:
            # Convert to format expected by featurize
            chain_id = label.split('_')[-1]  # Extract chain ID (e.g., 'A')
            
            # Remove His-tags from sequence
            sequence = seq
            if sequence[-6:] == "HHHHHH":
                sequence = sequence[:-6]
            if sequence[0:6] == "HHHHHH":
                sequence = sequence[6:]
            # Add other His-tag removal patterns if needed
            
            if len(sequence) < 4:
                continue
                
            # Get coordinates and ensure proper shape
            # Handle various possible shapes of coordinates
            if xyz.dim() == 4 and xyz.shape[0] == 1 and xyz.shape[2] == 4:  # [1, L, 4, 3]
                all_atoms = xyz.squeeze(0)
            elif xyz.dim() == 5 and xyz.shape[0] == 1 and xyz.shape[1] == 1:  # [1, 1, L, 4, 3]
                all_atoms = xyz.squeeze(0).squeeze(0)
            else:
                print(f"Unable to handle coordinate shape: {xyz.shape}")
                continue
            
            print(f"DEBUG: {label} - reshaped coords to shape: {all_atoms.shape}, seq length: {len(sequence)}")
            
            # Adjust coordinates to match sequence length after His-tag removal
            if len(all_atoms) > len(sequence):
                print(f"DEBUG: Trimming coordinates from {len(all_atoms)} to {len(sequence)}")
                all_atoms = all_atoms[:len(sequence)]
            elif len(all_atoms) < len(sequence):
                print(f"Warning: coordinates shorter than sequence for {label}: coords={len(all_atoms)}, seq={len(sequence)}")
                continue
            
            # Create coordinate dictionary
            coords_dict = {
                f'N_chain_{chain_id}': all_atoms[:,0,:].tolist(),
                f'CA_chain_{chain_id}': all_atoms[:,1,:].tolist(),
                f'C_chain_{chain_id}': all_atoms[:,2,:].tolist(),
                f'O_chain_{chain_id}': all_atoms[:,3,:].tolist(),
            }
            
            # Determine masking
            if masked.dim() > 0:
                masked_values = masked.tolist()
            else:
                masked_values = [masked.item()]
            
            if 0 in masked_values:
                masked_list = [chain_id]
                visible_list = []
            else:
                masked_list = []
                visible_list = [chain_id]
            
            # Create final structure in format expected by featurize
            converted_item = {
                f'seq_chain_{chain_id}': sequence,
                f'coords_chain_{chain_id}': coords_dict,
                'masked_list': masked_list,
                'visible_list': visible_list,
                'num_of_chains': 1,
                'seq': sequence,
                'name': label
            }
            
            pdb_dict_list.append(converted_item)
            count += 1
            print(f"DEBUG: Successfully converted {label}")
            
            if count >= num_examples:
                break
    
    return pdb_dict_list

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess 
    from utils import worker_init_fn, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "VAL": f"{data_path}/validation_proteins.txt",
        "TEST": f"{data_path}/test_proteins.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut, #resolution cutoff for PDBs
        "HOMO": 0.70 #min seq.id. to detect homo chains
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 4}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    print("Building training clusters...")
    train, valid, test = build_custom_training_clusters(params, args.debug)

    print("Creating training dataset...")
    train_set = PDB_dataset(list(train.keys()), custom_loader_pdb, train, params)
    
    # Test sample
    print("Sample training item:", train_set[0])

    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), custom_loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    print("Length train_loader:", len(train_loader))

    # Check a batch
    for b in train_loader:
        print("Batch Trainloader example:", {k: type(v) for k, v in b.items()})
        break
    
    # Create the model
    model = ProteinMPNN(node_features=args.hidden_dim, 
                    edge_features=args.hidden_dim, 
                    hidden_dim=args.hidden_dim, 
                    num_encoder_layers=args.num_encoder_layers, 
                    num_decoder_layers=args.num_encoder_layers, 
                    k_neighbors=args.num_neighbors, 
                    dropout=args.dropout, 
                    augment_eps=args.backbone_noise)
    model.to(device)

    # Load pre-trained weights if specified
    if PATH:
        print(f"Loading weights from {PATH}")
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
            optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        total_step = 0
        epoch = 0
        optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    # Load data directly from loaders
    print("Loading training data...")
    pdb_dict_train = load_data_from_loader(train_loader, args.max_protein_length, args.num_examples_per_epoch)
    print(f"Loaded {len(pdb_dict_train)} training examples")
    
    print("Loading validation data...")  
    pdb_dict_valid = load_data_from_loader(valid_loader, args.max_protein_length, args.num_examples_per_epoch)
    print(f"Loaded {len(pdb_dict_valid)} validation examples")
      
    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
    
    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    
    # Training loop
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.
        
        for _, batch in enumerate(loader_train):
            start_batch = time.time()
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            elapsed_featurize = time.time() - start_batch
            optimizer.zero_grad()
            mask_for_loss = mask*chain_M
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
        
                scaler.scale(loss_av_smoothed).backward()
                  
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()
            
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
        
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()

            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            total_step += 1

        # Validation
        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            validation_acc = 0.
            for _, batch in enumerate(loader_valid):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                
                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
        
        # Calculate metrics
        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)
        
        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
        
        # Log results
        log_message = f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}'
        with open(logfile, 'a') as f:
            f.write(f'{log_message}\n')
        print(log_message)
        
        # Save checkpoint
        checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'
        torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)

        # Save periodic checkpoints
        if (e+1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
            torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename)

        # Reload data periodically if needed
        if (e+1) % args.reload_data_every_n_epochs == 0 and e+1 < args.num_epochs:
            print("Reloading training data...")
            pdb_dict_train = load_data_from_loader(train_loader, args.max_protein_length, args.num_examples_per_epoch)
            dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
            loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, required=True, help="path for loading acidophilic dataset") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_acidophilic", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000, help="number of training examples to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=1000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=1000, help="maximum length of the protein complex")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", action='store_true', help="run in debug mode with minimal data loading")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", action='store_true', help="train with mixed precision")
 
    args = argparser.parse_args()    
    main(args)   
