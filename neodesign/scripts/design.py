#!/usr/bin/env python
"""
Peptide generation using MCMC sampling on energy map
Input: MHC pseudo-seq and/or TCR sequence
Output: Top # peptides with lowest energy
Users can flexibly specify peptide length and number to generate
"""
import os
import torch
import pandas as pd
import numpy as np
import random
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from neodesign.model.model import NeoDesignModel
from neodesign.config.default import get_default_config
from neodesign.utils.constant import RESTYPES, RESTYPES_WITH_X

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global mappings
RESTYPE_TO_IDX = {aa: i for i, aa in enumerate(RESTYPES_WITH_X)}
IDX_TO_RESTYPE = {v: k for k, v in RESTYPE_TO_IDX.items()}


# ===============================
# Sequence Encoding Functions
# ===============================

def encode_sequence(seq, device='cuda'):
    """Encode a sequence to one-hot and index tensors."""
    if isinstance(device, str):
        device = torch.device(device)
        
    if pd.isna(seq) or seq == '':
        return torch.zeros(1, 1, 21, device=device), torch.zeros(1, 1, dtype=torch.long, device=device)

    # Create tensors on specified device
    idxs = [RESTYPE_TO_IDX.get(aa, len(RESTYPES_WITH_X)-1) for aa in seq]
    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(0)
    one_hot = torch.zeros(1, len(seq), 21, device=device)
    
    # Fill one-hot encoding
    for i, idx_val in enumerate(idxs):
        one_hot[0, i, idx_val if idx_val < 21 else 20] = 1.0
    
    return one_hot, idx_tensor


def create_base_batch(mhc_seq, seq_len, device='cuda', l_v_seq='', l_j_seq='', 
                      h_v_seq='', cdr3_b_seq='', h_j_seq=''):
    """Create input batch with zeros for epitope sequence (only positional info)."""
    if isinstance(device, str):
        device = torch.device(device)
        
    # Handle missing sequences
    l_v_seq = l_v_seq if pd.notna(l_v_seq) else ''
    l_j_seq = l_j_seq if pd.notna(l_j_seq) else ''
    h_v_seq = h_v_seq if pd.notna(h_v_seq) else ''
    cdr3_b_seq = cdr3_b_seq if pd.notna(cdr3_b_seq) else ''
    h_j_seq = h_j_seq if pd.notna(h_j_seq) else ''
    mhc_seq = mhc_seq if pd.notna(mhc_seq) else ''

    logger.debug(f"Sequence lengths - MHC: {len(mhc_seq)}, L_V: {len(l_v_seq)}, L_J: {len(l_j_seq)}")
    logger.debug(f"H_V: {len(h_v_seq)}, CDR3_B: {len(cdr3_b_seq)}, H_J: {len(h_j_seq)}")
   
    # Encode fixed sequences
    mhc_encoded, mhc_aatype = encode_sequence(mhc_seq, device)
    l_v_encoded, l_v_aatype = encode_sequence(l_v_seq, device)
    l_j_encoded, l_j_aatype = encode_sequence(l_j_seq, device)
    h_v_encoded, h_v_aatype = encode_sequence(h_v_seq, device)
    cdr3_b_encoded, cdr3_b_aatype = encode_sequence(cdr3_b_seq, device)
    h_j_encoded, h_j_aatype = encode_sequence(h_j_seq, device)
    
    # Create empty epitope encoding since we want to predict it
    epitope_encoded = torch.zeros(1, seq_len, 21, device=device)
    epitope_aatype = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    
    # Create residue indices
    mhc_ri = torch.arange(mhc_encoded.size(1), device=device)[None, :]
    l_v_ri = torch.arange(l_v_encoded.size(1), device=device)[None, :]
    l_j_ri = torch.arange(l_j_encoded.size(1), device=device)[None, :]
    h_v_ri = torch.arange(h_v_encoded.size(1), device=device)[None, :]
    cdr3_b_ri = torch.arange(cdr3_b_encoded.size(1), device=device)[None, :]
    h_j_ri = torch.arange(h_j_encoded.size(1), device=device)[None, :]
    epitope_ri = torch.arange(seq_len, device=device)[None, :]
    
    # Create type tensors
    mhc_types = torch.zeros((1, mhc_encoded.size(1)), device=device)
    l_v_types = torch.ones((1, l_v_encoded.size(1)), device=device)
    l_j_types = torch.full((1, l_j_encoded.size(1)), 2, device=device)
    h_v_types = torch.full((1, h_v_encoded.size(1)), 3, device=device)
    cdr3_b_types = torch.full((1, cdr3_b_encoded.size(1)), 4, device=device)
    h_j_types = torch.full((1, h_j_encoded.size(1)), 5, device=device)
    epitope_types = torch.full((1, seq_len), 6, device=device)
    
    return {
        'mhc_encoded': mhc_encoded,
        'mhc_aatype': mhc_aatype,
        'mhc_ri': mhc_ri,
        'mhc_types': mhc_types,
        'l_v_encoded': l_v_encoded,
        'l_v_aatype': l_v_aatype,
        'l_v_ri': l_v_ri,
        'l_v_types': l_v_types,
        'l_j_encoded': l_j_encoded,
        'l_j_aatype': l_j_aatype,
        'l_j_ri': l_j_ri,
        'l_j_types': l_j_types,
        'h_v_encoded': h_v_encoded,
        'h_v_aatype': h_v_aatype,
        'h_v_ri': h_v_ri,
        'h_v_types': h_v_types,
        'cdr3_b_encoded': cdr3_b_encoded,
        'cdr3_b_aatype': cdr3_b_aatype,
        'cdr3_b_ri': cdr3_b_ri,
        'cdr3_b_types': cdr3_b_types,
        'h_j_encoded': h_j_encoded,
        'h_j_aatype': h_j_aatype,
        'h_j_ri': h_j_ri,
        'h_j_types': h_j_types,
        'epitope_encoded': epitope_encoded,
        'epitope_aatype': epitope_aatype,
        'epitope_ri': epitope_ri,
        'epitope_types': epitope_types,
        'mhc_lengths': torch.tensor([mhc_encoded.size(1)], device=device),
        'l_v_lengths': torch.tensor([l_v_encoded.size(1)], device=device),
        'l_j_lengths': torch.tensor([l_j_encoded.size(1)], device=device),
        'h_v_lengths': torch.tensor([h_v_encoded.size(1)], device=device),
        'cdr3_b_lengths': torch.tensor([cdr3_b_encoded.size(1)], device=device),
        'h_j_lengths': torch.tensor([h_j_encoded.size(1)], device=device),
        'epitope_lengths': torch.tensor([seq_len], device=device)
    }


# ===============================
# Energy Calculation Functions
# ===============================

def calculate_total_energy(aatype, seq_log_probs, pair_log_probs):
    """Calculate total energy using pre-computed log probabilities."""
    batch_size, seq_len = aatype.shape
    
    # Single site energy
    single_energy = -torch.gather(
        seq_log_probs,
        dim=-1,
        index=aatype[0].unsqueeze(-1)
    ).sum()
    
    # Pair energy
    aa_indices = aatype[0]
    pair_energy = -pair_log_probs[0]
    
    pair_energy = pair_energy[
        torch.arange(seq_len)[:,None,None],
        torch.arange(seq_len)[None,:,None],
        aa_indices[:,None,None],
        aa_indices[None,:,None]
    ].squeeze(-1)
    
    pair_energy = pair_energy.sum() / 2
    
    return single_energy + pair_energy


def sequence_accuracy(predicted_seq, target_seq):
    """Calculate sequence accuracy between predicted and target sequences."""
    if len(predicted_seq) != len(target_seq):
        logger.warning(f"Length mismatch - predicted: {len(predicted_seq)}, target: {len(target_seq)}")
        return 0.0
    
    matches = sum(1 for p, t in zip(predicted_seq, target_seq) if p == t)
    return matches / len(target_seq)


# ===============================
# MCMC Sampling Functions
# ===============================

def sample_from_cached_probs(seq_log_probs, pair_log_probs, initial_seq, test_seq=None, 
                           seq_len=None, min_steps=2000, patience=1000, min_temp=0.1, 
                           max_temp=20.0, n_replicas=5, device='cuda'):
    """MCMC sampling using pre-computed probabilities."""
    logger.info("Initializing replicas with cached probabilities...")
    
    if seq_len is None:
        seq_len = len(initial_seq)
    
    # Initialize replicas
    replicas = []
    for i in range(n_replicas):
        # Sample initial sequence from single-site probabilities
        replica_aatype = torch.zeros((1, seq_len), dtype=torch.long, device=device)
        for pos in range(seq_len):
            logits = seq_log_probs[pos]
            probs = F.softmax(logits / max_temp, dim=-1)
            sampled_aa = torch.multinomial(probs, num_samples=1)[0]
            replica_aatype[0, pos] = sampled_aa

        # Calculate initial energy
        initial_energy = calculate_total_energy(replica_aatype, seq_log_probs, pair_log_probs)
        logger.info(f"Replica {i+1} initial energy: {initial_energy.item():.4f}")
        
        replicas.append({
            'aatype': replica_aatype,
            'energy': initial_energy.item(),
            'best_energy': initial_energy.item(),
            'best_seq': ''.join(IDX_TO_RESTYPE[i] for i in replica_aatype[0].cpu().numpy()),
            'steps_without_improvement': 0
        })

    all_samples = []
    all_energies = []
    step = 0
    best_overall_energy = float('inf')
    steps_without_global_improvement = 0

    # Sampling function
    def get_adaptive_temp(base_temp, energy):
        """Calculate adaptive temperature based on current energy"""
        if energy > 100:
            return base_temp * 2.0
        elif energy > 50:
            return base_temp
        else:
            return base_temp * 0.5

    logger.info("Starting sampling with cached probabilities...")
    while True:
        # Calculate current temperature with annealing schedule
        base_temp = max_temp * (min_temp/max_temp)**(step/min_steps)
        if step % 100 < 20:  # Temperature cycling
            base_temp = max_temp
        
        # Update each replica
        for replica_idx, replica in enumerate(replicas):
            temp = get_adaptive_temp(base_temp, replica['energy'])
            
            # Sampling strategy based on energy
            block_size = min(
                8 if replica['energy'] > 100 else 
                4 if replica['energy'] > 50 else 2,
                seq_len
            )
            random_mutation_prob = (
                0.3 if replica['energy'] > 100 else
                0.2 if replica['energy'] > 50 else 0.1
            )

            # Randomize positions to update
            positions = list(range(seq_len))
            random.shuffle(positions)
            
            # Process positions in blocks
            for start_idx in range(0, seq_len, block_size):
                block_positions = positions[start_idx:start_idx + block_size]
                if not block_positions:
                    continue

                current_aas = replica['aatype'][0, block_positions].clone()
                current_energy = replica['energy']

                # Try multiple moves
                best_block_energy = float('inf')
                best_block_aas = None
                n_tries = 5 if replica['energy'] <= 50 else 3
                
                for _ in range(n_tries):
                    # Update block using cached probabilities
                    for pos in block_positions:
                        if random.random() < random_mutation_prob:
                            new_aa = random.randint(0, 20)
                        else:
                            energies = seq_log_probs[pos].clone()
                            for j in range(seq_len):
                                if j not in block_positions:
                                    other_aa = replica['aatype'][0, j].item()
                                    energies += pair_log_probs[0, pos, j, :, other_aa]
                            
                            probs = F.softmax(energies / temp, dim=-1)
                            new_aa = torch.multinomial(probs, num_samples=1)[0]
                        
                        replica['aatype'][0, pos] = new_aa

                    # Evaluate energy
                    new_energy = calculate_total_energy(
                        replica['aatype'], 
                        seq_log_probs, 
                        pair_log_probs
                    ).item()
                    
                    # Track best move
                    if new_energy < best_block_energy:
                        best_block_energy = new_energy
                        best_block_aas = replica['aatype'][0, block_positions].clone()
                    
                    # Revert changes for next try
                    replica['aatype'][0, block_positions] = current_aas

                # Accept or reject based on energy difference
                if best_block_aas is not None:
                    energy_diff = best_block_energy - current_energy
                    acceptance_prob = np.exp(-energy_diff / temp)
                    
                    if energy_diff < 0 or random.random() < acceptance_prob:
                        replica['aatype'][0, block_positions] = best_block_aas
                        replica['energy'] = best_block_energy

                        if best_block_energy < replica['best_energy']:
                            replica['best_energy'] = best_block_energy
                            current_seq = ''.join(
                                IDX_TO_RESTYPE[i] for i in replica['aatype'][0].cpu().numpy()
                            )
                            replica['best_seq'] = current_seq
                            replica['steps_without_improvement'] = 0
                            all_samples.append((current_seq, best_block_energy, step))
                            all_energies.append(best_block_energy)
                        else:
                            replica['steps_without_improvement'] += 1

        # Update stopping criteria and check for improvements
        curr_best = min(replica['energy'] for replica in replicas)
        if curr_best < best_overall_energy:
            best_overall_energy = curr_best
            steps_without_global_improvement = 0
        else:
            steps_without_global_improvement += 1

        if step % 10 == 0:
            logger.info(f"Step {step}, Best Energy: {min(r['best_energy'] for r in replicas):.4f}")
            if test_seq:
                best_seq = min(replicas, key=lambda r: r['best_energy'])['best_seq']
                accuracy = sequence_accuracy(best_seq, test_seq)
                logger.info(f"Best sequence accuracy: {accuracy:.2f}")

        # Stopping criteria
        if step >= min_steps and steps_without_global_improvement >= patience:
            break
        step += 1

    logger.info("Sampling complete.")
    return all_samples, all_energies


# ===============================
# Main Functionality
# ===============================

def generate_peptides(model_path, input_data, output_file, n_runs=3, n_sequences=100):
    """Generate peptides using model and MCMC sampling."""
    # Load model 
    config = get_default_config()
    model = NeoDesignModel(config)
    
    checkpoint = torch.load(model_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    logger.info("Model loaded successfully")

    # Load input data
    if isinstance(input_data, str):
        input_data = pd.read_csv(input_data)
    
    # Process each input sequence
    results = []
    for i, row in input_data.iterrows():
        try:
            logger.info(f"Processing input {i+1}/{len(input_data)}")
            
            # Get sequence information
            mhc_seq = row['mhc']
            peptide_len = row.get('peptide_length', 9)
            
            # Get TCR sequences if available
            l_v_seq = row.get('l_v', '')
            l_j_seq = row.get('l_j', '')
            h_v_seq = row.get('h_v', '')
            cdr3_b_seq = row.get('cdr3_b', '')
            h_j_seq = row.get('h_j', '')
            
            # Optional target sequence for comparison
            target_seq = row.get('peptide', None)

            batch_seq_len = peptide_len if target_seq is None else len(target_seq)
            base_batch = create_base_batch(
                mhc_seq=mhc_seq,
                seq_len=batch_seq_len,
                device='cuda',
                l_v_seq=l_v_seq,
                l_j_seq=l_j_seq,
                h_v_seq=h_v_seq,
                cdr3_b_seq=cdr3_b_seq,
                h_j_seq=h_j_seq
            )

            with torch.no_grad():
                outputs = model(base_batch, compute_loss=False, compute_energy=True)
                seq_logits = outputs['seqhead_pred']
                pair_logits = outputs['pairhead_pred']
                
                # Compute log probabilities
                seq_log_probs = F.log_softmax(seq_logits.squeeze(0).squeeze(0), dim=-1)
                N = pair_logits.size(1)
                pair_log_probs = F.log_softmax(pair_logits.view(1, N, N, 21, 21), dim=-1)

                # Calculate reference energy if target exists
                if target_seq is not None:
                    test_encoded, test_aatype = encode_sequence(target_seq, 'cuda')
                    test_energy = calculate_total_energy(test_aatype, seq_log_probs, pair_log_probs)
                    logger.info(f"Target sequence energy: {test_energy:.4f}")

            # Run sampling
            all_samples = []
            all_energies = []
            for run in range(n_runs):
                logger.info(f"Sampling run {run+1}/{n_runs}")
                random_initial = ''.join(random.choice(RESTYPES) for _ in range(batch_seq_len))
                
                samples, energies = sample_from_cached_probs(
                    seq_log_probs=seq_log_probs,
                    pair_log_probs=pair_log_probs,
                    initial_seq=random_initial,
                    test_seq=target_seq,
                    min_steps=1000,
                    patience=500
                )
                all_samples.extend(samples)
                all_energies.extend(energies)

            # Process results
            sorted_samples = sorted(all_samples, key=lambda x: x[1])
            unique_sequences = []
            seen = set()
            
            for seq, energy, iter_num in sorted_samples:
                if seq not in seen:
                    unique_sequences.append((seq, energy, iter_num))
                    seen.add(seq)
                if len(unique_sequences) >= n_sequences:
                    break

            logger.info(f"Found {len(unique_sequences)} unique sequences")
            
            # Store results
            result_entry = {
                'input_id': i,
                'mhc_seq': mhc_seq,
                'l_v_seq': l_v_seq,
                'l_j_seq': l_j_seq,
                'h_v_seq': h_v_seq,
                'cdr3_b_seq': cdr3_b_seq,
                'h_j_seq': h_j_seq
            }
            
            if target_seq is not None:
                result_entry['target_seq'] = target_seq

            for rank, (seq, energy, _) in enumerate(unique_sequences, 1):
                result_entry[f'gen_seq_{rank}'] = seq
                result_entry[f'energy_{rank}'] = energy
                if target_seq is not None:
                    result_entry[f'accuracy_{rank}'] = sequence_accuracy(seq, target_seq)

            results.append(result_entry)
        except Exception as e:
            logger.error(f"Error processing input {i}: {e}", exc_info=True)

    # Save results
    results_df = pd.DataFrame(results)
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    return results_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate peptides using MCMC sampling')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file with MHC and TCR sequences')
    parser.add_argument('--output', type=str, default='generated_peptides.csv',
                        help='Path to output CSV file')
    parser.add_argument('--peptide-length', type=int, default=9,
                        help='Length of peptides to generate')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of sampling runs per input')
    parser.add_argument('--n-sequences', type=int, default=100,
                        help='Number of sequences to generate per input')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of sampling results')
    
    return parser.parse_args()


# ===============================
# Main Entry Point
# ===============================

if __name__ == "__main__":
    args = parse_args()
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = generate_peptides(
        model_path=args.model,
        input_data=args.input,
        output_file=args.output,
        n_runs=args.n_runs,
        n_sequences=args.n_sequences
    )
    
    logger.info(f"Generated {len(results)} peptide sets")
