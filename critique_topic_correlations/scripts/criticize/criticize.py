import sys
import os
import math
import json
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
import seaborn as sns

parser = argparse.ArgumentParser(description='Compute Latent NLL.')
parser.add_argument('--critic', type=str, required=True,
                    help='Folder containing critic checkpoints')
parser.add_argument('--real_data_lambda', type=str, required=True,
                    help='CTM\'s inference results on real data.')
parser.add_argument('--LM_generations_lambda', type=str, required=True,
                    help='CTM\'s inference results on language model generations.')
parser.add_argument('--visualize_cov_path', type=str, default=None,
                    help='Visualize the covariance matrices and save to this path.')
parser.add_argument('--hierarchical_clustering', action='store_true',
                    help='If set, rearrange topic ids using hierarchical clustering (on real data) to facilitate visual comparison.')
args = parser.parse_args()

def load_matrix(filename, num_cols):
    with open(filename) as fin:
        nums = [float(line.strip()) for line in fin.readlines()]
        array = torch.Tensor(nums).view(-1, num_cols)
    return array

def load_params(param_filename):
    params = {}
    with open(param_filename) as fin:
        for line in fin:
            key, val = line.strip().split()
            params[key] = val
    return params

def eval_latent_NLL(critic_distribution, lamb):
    total_llh = 0
    for l in lamb:
        total_llh += critic_distribution.log_prob(l)
    return -total_llh / lamb.size(0)

def compute_cov(lamb):
    N, M = lamb.size()
    avg_lamb = lamb.mean(0)
    return (lamb.view(N, M, 1) * lamb.view(N, 1, M)).mean(0) - avg_lamb.view(-1, 1) * avg_lamb.view(1, -1)

def main(args):
    # Load model
    params = load_params(os.path.join(args.critic, 'final-param.txt'))
    num_topics = int(params['num_topics'])
    mu_file = os.path.join(args.critic, 'final-mu.dat')
    mu = load_matrix(mu_file, num_topics-1).view(-1) # in CTM the last lambda is set to 0 so mu only has num_topics-1 components
    inv_cov_file = os.path.join(args.critic, 'final-inv-cov.dat')
    inv_cov = load_matrix(inv_cov_file, num_topics-1)
    critic_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, precision_matrix=inv_cov)

    # Load posterior inference results
    real_data_lamb = load_matrix(args.real_data_lambda, num_topics)[:, :num_topics-1] 
    LM_generations_lamb = load_matrix(args.LM_generations_lambda, num_topics)[:, :num_topics-1]

    # Criticise text
    real_data_latent_NLL = eval_latent_NLL(critic_distribution, real_data_lamb)
    print (f'Latent PPL (Data): {real_data_latent_NLL}')
    LM_generations_latent_NLL = eval_latent_NLL(critic_distribution, LM_generations_lamb)
    print (f'Latent PPL (LM): {LM_generations_latent_NLL}')

    # Visualize covariance matrices
    if args.visualize_cov_path is not None:
        real_data_cov = compute_cov(real_data_lamb)
        LM_generations_cov = compute_cov(LM_generations_lamb)
        if args.hierarchical_clustering:
            clustergrid = sns.clustermap(real_data_cov.data.numpy(), cmap='jet', vmin=-5, vmax=5)
            reordered_id = clustergrid.dendrogram_row.reordered_ind
            reordered_id = torch.LongTensor(reordered_id).view(-1)
        else:
            reordered_id = None
        def reorder(X):
            if reordered_id is None:
                return X
            return X[reordered_id, :][:, reordered_id]

        fig = plt.figure(figsize=(9.75, 3))
    
        axes = ImageGrid(fig, 111,
                         nrows_ncols=(1,2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        im0 = axes[0].imshow(reorder(real_data_cov).data.numpy(), cmap='jet', vmin=-5, vmax=5, interpolation='nearest')
        im1 = axes[1].imshow(reorder(LM_generations_cov).data.numpy(), cmap='jet', vmin=-5, vmax=5, interpolation='nearest')
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[1].cax.colorbar(im1)
        axes[1].cax.toggle_label(True)
        plt.savefig(args.visualize_cov_path, bbox_inches='tight', pad_inches=0.1, dpi=200,)
        print (f'Covariance matrix plots written to {args.visualize_cov_path}')

if __name__ == '__main__':
    main(args)
