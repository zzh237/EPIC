"""
Run experiments against libero
"""
import os
# I think you can make this true when not debugging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
# torch.set_default_dtype(torch.float32)

from torch import nn


from libero.lifelong.metric import evaluate_loss, evaluate_success
import hydra
from tqdm import trange
import numpy as np
import yaml
from easydict import EasyDict
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.utils import get_task_embs, create_experiment_dir, safe_device
from omegaconf import OmegaConf
from epic_libero.algorithm import EPICAlgorithm, MyLifelongAlgo
import epic_libero.policy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from bayesian_torch.models.dnn_to_bnn import bnn_conv_layer, bnn_linear_layer, bnn_lstm_layer
from functools import reduce

hydra.initialize_config_module("libero.configs")
hydra_cfg = hydra.compose(config_name="config")
yml_cfg = OmegaConf.to_yaml(hydra_cfg)
cfg = EasyDict(yaml.safe_load(yml_cfg))

cfg.device = "cuda"
cfg.folder = get_libero_path("datasets")
cfg.bddl_folder = get_libero_path("bddl_files")
cfg.init_states_folder = get_libero_path("init_states")
cfg.eval.num_procs = 1
cfg.eval.n_eval = 5

cfg.train.n_epochs = 25

task_order = cfg.data.task_order_index
cfg.benchmark_name = "libero_10"
benchmark = get_benchmark(cfg.benchmark_name)(task_order)

datasets = []
descriptions = []
shape_meta = None
n_tasks = benchmark.n_tasks

for i in range(n_tasks):
    task_i_dataset, shape_meta = get_dataset(
        dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=(i == 0),
        seq_len=cfg.data.seq_len,
    )
    descriptions.append(benchmark.get_task(i).language)
    datasets.append(task_i_dataset)

task_embs = get_task_embs(cfg, descriptions)
benchmark.set_task_embs(task_embs)

datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
n_demos = [data.n_demos for data in datasets]
n_sequences = [data.total_num_sequences for data in datasets]


# TODO determine if it is safe to do this, specifically does the algorithm use the name
# to reinitialize the policy
cfg.policy.policy_type = "MyTransformerPolicy"
cfg.lifelong.algo = "MyLifelongAlgo"

create_experiment_dir(cfg)
cfg.shape_meta = shape_meta

print(f"experiment directory is: {cfg.experiment_dir}")

def dnn_to_bnn(module: nn.Module, bnn_prior_parameters):
    for name, m in module.named_modules():
        if "Conv" in m.__class__.__name__:
            module.set_submodule(name, bnn_conv_layer(
                    bnn_prior_parameters,
                    m))
        elif "Linear" in m.__class__.__name__:
            module.set_submodule(name, bnn_linear_layer(
                bnn_prior_parameters,
                m))
        elif "LSTM" in m.__class__.__name__:
            module.set_submodule(name, bnn_lstm_layer(bnn_prior_parameters, m))


def policy_maker(cfg, shape_meta):
    policy = BCTransformerPolicy(cfg, shape_meta)

    # dnn_to_bnn(policy, {
    #     "prior_mu": 0.0,
    #     "prior_sigma": 1.0,
    #     "posterior_mu_init": 0.0,
    #     "posterior_rho_init": -3.0,
    #     "type": "Reparameterization",
    #     "moped_enable": False,
    #     "moped_delta": 0.5
    # })

    return policy

algo = safe_device(MyLifelongAlgo(n_tasks, cfg), cfg.device)

result_summary = {
    'L_conf_mat': np.zeros((n_tasks, n_tasks)),   # loss confusion matrix
    'S_conf_mat': np.zeros((n_tasks, n_tasks)),   # success confusion matrix
    'L_fwd'     : np.zeros((n_tasks,)),           # loss AUC, how fast the agent learns
    'S_fwd'     : np.zeros((n_tasks,)),           # success AUC, how fast the agent succeeds
}

gsz = cfg.data.task_group_size

if (cfg.train.n_epochs < 50):
    print("NOTE: the number of epochs used in this example is intentionally reduced to 30 for simplicity.")
if (cfg.eval.n_eval < 20):
    print("NOTE: the number of evaluation episodes used in this example is intentionally reduced to 5 for simplicity.")

# skip the first task, which has a weird shape?
for i in trange(n_tasks):
    algo.train()
    d = datasets[i]
    s_fwd, l_fwd = algo.learn_one_task(d, i, benchmark, result_summary)
    # s_fwd is success rate AUC, when the agent learns the {0, e, 2e, ...} epochs
    # l_fwd is BC loss AUC, similar to s_fwd
    result_summary["S_fwd"][i] = s_fwd
    result_summary["L_fwd"][i] = l_fwd

    if cfg.eval.eval:
        algo.eval()
        # we only evaluate on the past tasks: 0 .. i
        L = evaluate_loss(cfg, algo, benchmark, datasets[:i+1]) # (i+1,)
        S = evaluate_success(cfg, algo, benchmark, list(range((i+1)*gsz))) # (i+1,)
        result_summary["L_conf_mat"][i][:i+1] = L
        result_summary["S_conf_mat"][i][:i+1] = S

        torch.save(result_summary, os.path.join(cfg.experiment_dir, f'result.pt'))


