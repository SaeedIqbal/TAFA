import torch
import numpy as np

def tafa_aggregation(decoder_updates, trust_scores, global_stats):
    """Algorithm 1: Trust-Aware Federated Aggregation."""
    # Clip trust scores
    lower_bound = global_stats['trust_mean'] - 3 * global_stats['trust_std']
    upper_bound = global_stats['trust_mean'] + 3 * global_stats['trust_std']
    clipped_scores = np.clip(trust_scores, lower_bound, upper_bound)
    
    # Normalize
    weights = clipped_scores / np.sum(clipped_scores)
    
    # Weighted average
    global_decoder = {}
    for key in decoder_updates[0].keys():
        global_decoder[key] = torch.zeros_like(decoder_updates[0][key])
        for i, local_update in enumerate(decoder_updates):
            global_decoder[key] += weights[i] * local_update[key]
    
    # Update global stats
    new_mean = np.mean(clipped_scores)
    new_std = np.std(clipped_scores)
    
    return global_decoder, {'trust_mean': new_mean, 'trust_std': new_std}