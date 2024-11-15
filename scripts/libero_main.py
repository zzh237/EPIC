"""
Run experiments against libero
"""
import os
# I think you can make this true when not debugging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

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
from omegaconf import DictConfig, OmegaConf
from epic_libero.algorithm import EPICAlgorithm, MyLifelongAlgo
import epic_libero.policy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
import hydra
import imageio

from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@hydra.main(config_name="user_config", config_path="../libero_config", version_base=None)
def main(hydra_cfg: DictConfig):
    yml_cfg = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yml_cfg))

    print(f"Beginning experiment run for policy: {cfg.policy.policy_type}, algorithm: {cfg.lifelong.algo}")

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    task_order = cfg.data.task_order_index
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

    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    print(f"experiment directory is: {cfg.experiment_dir}")

    algo = safe_device(MyLifelongAlgo(n_tasks, cfg), cfg.device)

    result_summary = {
        'L_conf_mat': np.zeros((n_tasks, n_tasks)),   # loss confusion matrix
        'S_conf_mat': np.zeros((n_tasks, n_tasks)),   # success confusion matrix
        'L_fwd'     : np.zeros((n_tasks,)),           # loss AUC, how fast the agent learns
        'S_fwd'     : np.zeros((n_tasks,)),           # success AUC, how fast the agent succeeds
    }

    gsz = cfg.data.task_group_size

    if (cfg.train.n_epochs < 50):
        print(f"NOTE: the number of epochs used in this example is intentionally reduced to {cfg.train.n_epochs} for simplicity.")
    if (cfg.eval.n_eval < 20):
        print(f"NOTE: the number of evaluation episodes used in this example is intentionally reduced to {cfg.train.n_epochs} for simplicity.")

    # save 1 task for eval
    n_train_tasks = (n_tasks - 1) if not cfg.dev_mode else 0
    for i in trange(n_train_tasks):
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

            torch.save(result_summary, os.path.join(cfg.experiment_dir, 'result.pt'))
    
    print(bcolors.OKCYAN + "Starting video eval" + bcolors.ENDC)
    # single trial rollout on a new task
    env_num = 1
    task_id = n_tasks
    task = benchmark.get_task(task_id - 1)
    task_emb = benchmark.get_task_emb(task_id - 1)

    algo.eval()
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }

    env = DummyVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    )
    init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)

    env.reset()

    init_state = init_states[0:1]
    dones = [False]

    algo.reset()

    obs = env.set_init_state(init_state)
    # assume that all tasks in the life have the same actiondim
    action_dim = shape_meta["ac_dim"]

    # manually open gripper to make it consistent with demonstration
    dummy_actions = np.zeros((env_num, action_dim))
    for _ in range(5):
        obs, _, _, _ = env.step(dummy_actions)

    steps = 0
    obs_tensors = [[]] * env_num

    while steps < cfg.eval.max_steps:
        steps += 1
        data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
        action = algo.policy.get_action(data)
        obs, reward, done, info = env.step(action)

        for k in range(env_num):
            dones[k] = dones[k] or done[k]
            obs_tensors[k].append(obs[k]["agentview_image"])
        if all(dones):
            break

    images = [img[::-1] for img in obs_tensors[0]]
    fps = 30
    outfile_name = os.path.join(cfg.experiment_dir, "agentview_eval.mp4")
    writer = imageio.get_writer(outfile_name, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()

    print(bcolors.OKCYAN + "Video saved to: " + outfile_name + bcolors.ENDC)

    env.close()


if __name__ == "__main__":
    main()
