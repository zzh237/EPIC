"""
Bayesian policy for libero framework
"""

import einops
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from einops import rearrange, repeat
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from bayesian_torch.models.dnn_to_bnn import (
    bnn_conv_layer,
    bnn_linear_layer,
    bnn_lstm_layer,
)


def dnn_to_bnn(module: nn.Module, bnn_prior_parameters):
    for name, m in module.named_modules():
        if "Conv" in m.__class__.__name__:
            module.set_submodule(name, bnn_conv_layer(bnn_prior_parameters, m))
        elif "Linear" in m.__class__.__name__:
            module.set_submodule(name, bnn_linear_layer(bnn_prior_parameters, m))
        elif "LSTM" in m.__class__.__name__:
            module.set_submodule(name, bnn_lstm_layer(bnn_prior_parameters, m))


class EpicBayesianPolicy(BCTransformerPolicy):
    """
    Bayesian version of the TransformerPolicy.
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        dnn_to_bnn(
            self,
            {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": "Reparameterization",  # Flipout or Reparameterization
                "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
                "moped_delta": 0.5,
            }
        )
        self.compile(mode="reduce-overhead")
        
    def preprocess_input(self, data, train_mode=True):
        # the data on disk has a different channel order than expected by
        # e.g. torchvision

        # but we don't need to permute it again
        obs = data["obs"]

        if len(obs["eye_in_hand_rgb"].shape) == 5:
            obs["eye_in_hand_rgb"] = einops.rearrange(
                obs["eye_in_hand_rgb"], "b n h w c -> b n c h w"
            ).type(torch.float32)
        if len(obs["agentview_rgb"].shape) == 5:
            obs["agentview_rgb"] = einops.rearrange(
                obs["agentview_rgb"], "b n h w c -> b n c h w"
            ).type(torch.float32)
        obs["joint_states"] = obs["joint_states"].type(torch.float32)
        obs["gripper_states"] = obs["gripper_states"].type(torch.float32)

        return super().preprocess_input(data, train_mode)


class MyTransformerPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. encode image
        embed_size = policy_cfg.embed_size
        transformer_input_sizes = []
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = (
                    policy_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        ### 2. encode language
        policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        ### 4. define temporal transformer
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
            policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        self.temporal_transformer = TransformerDecoder(
            input_size=embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
        )

        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = embed_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs,
        )

        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

        self.compile(mode="reduce-overhead")

    def temporal_encode(self, x):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        # 1. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)

        # 2. encode language, treat it as action token
        B, T = extra.shape[:2]
        text_encoded = self.language_encoder(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded = [text_encoded, extra]

        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data):
        x = self.spatial_encode(data)
        x = self.temporal_encode(x)
        dist = self.policy_head(x)
        return dist

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)
            self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
            x = self.temporal_encode(x)
            dist = self.policy_head(x[:, -1])
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []

    def preprocess_input(self, data, train_mode=True):
        # the data on disk has a different channel order than expected by
        # e.g. torchvision

        # but we don't need to permute it again
        obs = data["obs"]

        if len(obs["eye_in_hand_rgb"].shape) == 5:
            obs["eye_in_hand_rgb"] = einops.rearrange(
                obs["eye_in_hand_rgb"], "b n h w c -> b n c h w"
            ).type(torch.float32)
        if len(obs["agentview_rgb"].shape) == 5:
            obs["agentview_rgb"] = einops.rearrange(
                obs["agentview_rgb"], "b n h w c -> b n c h w"
            ).type(torch.float32)
        obs["joint_states"] = obs["joint_states"].type(torch.float32)
        obs["gripper_states"] = obs["gripper_states"].type(torch.float32)

        return super().preprocess_input(data, train_mode)


class ExtraModalityTokens(nn.Module):
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)

        extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
        )

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        self.extra_encoders = {}

        def generate_proprio_mlp_fn(modality_name, extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True),
                    ]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]

            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}

        for proprio_dim, use_modality, modality_name in [
            (joint_states_dim, self.use_joint, "joint_states"),
            (gripper_states_dim, self.use_gripper, "gripper_states"),
            (ee_dim, self.use_ee, "ee_states"),
        ]:
            if use_modality:
                generate_proprio_mlp_fn(modality_name, proprio_dim)

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.extra_encoders.values()]
        )

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []

        for use_modality, modality_name in [
            (self.use_joint, "joint_states"),
            (self.use_gripper, "gripper_states"),
            (self.use_ee, "ee_states"),
        ]:
            if use_modality:
                tensor_list.append(
                    self.extra_encoders[modality_name]["encoder"](
                        obs_dict[modality_name]
                    )
                )

        x = torch.stack(tensor_list, dim=-2)
        return x
