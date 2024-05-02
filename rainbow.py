# TMRL imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.actor import TorchActorModule
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training import TrainingAgent
from tmrl.training_offline import TrainingOffline
from tmrl.util import cached_property
from tmrl.util import partial

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch import Tensor

# Other imports
import os
from argparse import ArgumentParser
from copy import deepcopy
import math
import numpy as np


class ConfigParameters:
    def __init__(self):
        # Useful parameters
        self.epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
        self.rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
        self.steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
        self.start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
        self.max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
        self.update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
        self.update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
        self.device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
        self.memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
        self.batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

        # Wandb parameters
        self.wandb_run_id = cfg.WANDB_RUN_ID
        self.wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
        self.wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]
        self.wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]
        os.environ['WANDB_API_KEY'] = self.wandb_key

        self.max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

        # Networking parameters
        self.server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
        self.server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
        self.server_port = cfg.PORT
        self.password = cfg.PASSWORD
        self.security = cfg.SECURITY

        # Advanced parameters
        self.memory_base_cls = cfg_obj.MEM
        self.sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
        self.sample_preprocessor = None
        self.dataset_path = cfg.DATASET_PATH
        self.obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

        # Competition fixed parameters (we don't care)
        self.env_cls = cfg_obj.ENV_CLS
        self.device_worker = 'cuda' if cfg.CUDA_INFERENCE else 'cpu'

        # Environment parameters
        self.window_width = cfg.WINDOW_WIDTH
        self.window_height = cfg.WINDOW_HEIGHT
        self.img_width = cfg.IMG_WIDTH
        self.img_height = cfg.IMG_HEIGHT
        self.img_grayscale = cfg.GRAYSCALE
        self.imgs_buf_len = cfg.IMG_HIST_LEN
        self.act_buf_len = cfg.ACT_BUF_LEN

        # Custom stuff
        self.possible_actions = np.array(
            [np.array([0, 0, 0]), np.array([0, 0, -1]), np.array([0, 0, 1]), np.array([0, 1, 0]),
             np.array([0, 1, -1]), np.array([0, 1, 1]), np.array([1, 0, 0]), np.array([1, 0, -1]),
             np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 1, -1]), np.array([1, 1, 1])])

        self.possible_actions_tensor_trainer = torch.tensor(self.possible_actions, dtype=torch.float32).to(
            device=self.device_trainer)
        self.possible_actions_tensor_worker = torch.tensor(self.possible_actions, dtype=torch.float32).to(
            device=self.device_worker)

        self.V_MIN = -5
        self.V_MAX = 5
        self.atoms = 500


params = ConfigParameters()

memory_cls = partial(params.memory_base_cls,
                     memory_size=params.memory_size,
                     batch_size=params.batch_size,
                     sample_preprocessor=params.sample_preprocessor,
                     dataset_path=params.dataset_path,
                     imgs_obs=params.imgs_buf_len,
                     act_buf_len=params.act_buf_len,
                     crc_debug=False)


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input, log=False):
        if log:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)

        return F.linear(input, self.weight_mu, self.bias_mu)


class PreprocessLayer(nn.Module):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def _process_input(self, input_tensors: tuple[Tensor]) -> list[Tensor]:
        processed_tensors = []

        for tensor in input_tensors:
            if tensor.dim() < 1:
                tensor = tensor.expand((1, 3))
            processed_tensors.append(tensor)

        return processed_tensors

    def forward(self, input_tensors: tuple[Tensor]) -> Tensor:
        processed_tensors = self._process_input(input_tensors)
        return torch.cat(processed_tensors, dim=1)


class RainbowActorModule(TorchActorModule):

    def __init__(self, observation_space, action_space, atoms=500, hidden_size=512, noisy_std=0.1, activation=nn.ReLU):
        super().__init__(observation_space, action_space)

        dim_act = (action_space.shape[0] * 4)  # 2 actions options per action +1 for straight steering
        act_limit = action_space.high[0]

        self.process = PreprocessLayer()

        self.device = params.device_worker

        # TODO: make this a parameter
        self.atoms = atoms

        self.action_space = action_space
        # print(action_space)

        self.support = torch.linspace(params.V_MIN, params.V_MAX, self.atoms).to(device=self.device)  # Support (range) of z

        self.dense = nn.Sequential(nn.Linear(83, 256), activation(),
                                   nn.Linear(256, 512), activation(),
                                   nn.Linear(512, 512), activation())
        self.dense_output_size = 512
        self.fc_h_v = NoisyLinear(self.dense_output_size, hidden_size, std_init=noisy_std)
        self.fc_h_a = NoisyLinear(self.dense_output_size, hidden_size, std_init=noisy_std)
        self.fc_z_v = NoisyLinear(hidden_size, self.atoms, std_init=noisy_std)
        self.fc_z_a = NoisyLinear(hidden_size, dim_act * self.atoms, std_init=noisy_std)

        self.act_limit = act_limit
        self.dim_act = dim_act

    def forward(self, obs, log=False) -> tuple[Tensor, Tensor]:

        # Process observation
        processed_tensor = self.process(obs)

        # Feed into dense layer
        x = self.dense(processed_tensor)
        x = x.view(-1, self.dense_output_size)

        # Value and Advantage streams
        v = self.fc_z_v(F.relu(self.fc_h_v(x, log)), log)
        a = self.fc_z_a(F.relu(self.fc_h_a(x, log)), log)

        # Reshape v and a for atoms dimension
        v = v.view(-1, 1, self.atoms)
        a = a.view(-1, self.dim_act, self.atoms)

        # Combine streams
        q = v + a - a.mean(1, keepdim=True)

        # Return action probabilities and log-probabilities
        return F.softmax(q, dim=2), F.log_softmax(q, dim=2)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def act(self, obs, test=False):
        with torch.no_grad():
            action_probs, _ = self.forward(obs=obs, log=False)

            support_weighted = action_probs * self.support
            summed = support_weighted.sum(2)

            best_action_index = summed.argmax(1).item()

            best_action = params.possible_actions[best_action_index]
            return best_action


class DQN(nn.Module):

    def __init__(self, observation_space, action_space):
        super().__init__()

        self.actor = RainbowActorModule(observation_space, action_space)

    def reset_noise(self):
        self.actor.reset_noise()


class RAINTrainingAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=DQN,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 lr_entropy=1e-3,
                 learn_entropy_coef=True,
                 target_entropy=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef = learn_entropy_coef
        self.target_entropy = target_entropy
        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor, eps=1.5e-4)

        self.V_min = params.V_MIN
        self.V_max = params.V_MAX
        self.atoms = params.atoms

        self.iter = 0
        self.batch_size = params.batch_size
        self.norm_clip = 10

        # TODO: figure out what this should be
        self.n = 1

        self.support = torch.linspace(self.V_min, self.V_max, self.atoms).to(device=self.device)
        self.delta_z = (self.V_max - self.V_min) / (self.atoms - 1)
        torch.autograd.set_detect_anomaly(True)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        # Unpack batch
        previous_obs, actions, rewards, new_obs, terminated, truncated = batch

        action_indices = self._get_action_indices(actions)

        # Calculate current state probabilities
        action_probs, log_ps = self.model.actor(previous_obs, log=True)
        log_ps_a = log_ps[range(self.batch_size), action_indices]

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns, _ = self.model.actor(new_obs)
            dns = self.support.expand_as(pns) * pns
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.model_target.reset_noise()
            pns, _ = self.model_target.actor(new_obs)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # TODO: figure out self.n
            Tz = rewards.unsqueeze(1) + terminated.unsqueeze(1) * (self.gamma ** self.n) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)

            # Compute L2 project of Tz onto fixed support z
            b = (Tz - self.V_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] = -1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = previous_obs[0].new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.model.actor.zero_grad()

        # TODO: figure out weights
        loss.mean().backward()
        # (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.model.actor.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimizer.step()

        # TODO: figure out priorities
        # mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

        ret_dict = dict(
            loss_actor=loss.mean().item(),
            loss_critic=0,
        )

        return ret_dict

    def _get_action_indices(self, actions: Tensor) -> Tensor:
        diff = actions.unsqueeze(1) - params.possible_actions_tensor_trainer
        distances = torch.norm(diff, dim=2)
        indices = torch.argmin(distances, dim=1)
        return indices


training_agent_cls = partial(RAINTrainingAgent,
                             model_cls=DQN,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.02,
                             lr_actor=1e-3,
                             lr_critic=1e-3,
                             lr_entropy=1e-3,
                             learn_entropy_coef=True,
                             target_entropy=None)

training_cls = partial(
    TrainingOffline,
    env_cls=params.env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=params.epochs,
    rounds=params.rounds,
    steps=params.steps,
    update_buffer_interval=params.update_buffer_interval,
    update_model_interval=params.update_model_interval,
    max_training_steps_per_env_step=params.max_training_steps_per_env_step,
    start_training=params.start_training,
    device=params.device_trainer)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=params.server_ip_for_trainer,
                             server_port=params.server_port,
                             password=params.password,
                             security=params.security)
        my_trainer.run()
    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=params.env_cls,
                           actor_module_cls=RainbowActorModule,
                           sample_compressor=params.sample_compressor,
                           device=params.device_worker,
                           server_ip=params.server_ip_for_worker,
                           server_port=params.server_port,
                           password=params.password,
                           security=params.security,
                           max_samples_per_episode=params.max_samples_per_episode,
                           obs_preprocessor=params.obs_preprocessor,
                           standalone=args.test)
        rw.run()
    elif args.server:
        import time

        serv = Server(port=params.server_port,
                      password=params.password,
                      security=params.security)
        while True:
            time.sleep(1.0)
