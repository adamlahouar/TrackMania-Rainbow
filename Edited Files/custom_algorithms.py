# standard library imports
import itertools
from copy import deepcopy
from dataclasses import dataclass

# third-party imports
import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from torch.nn.utils import clip_grad_norm_

# local imports
import tmrl.custom.custom_models as core
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property, partial
from tmrl.training import TrainingAgent
import tmrl.config.config_constants as cfg

import logging


# Soft Actor-Critic ====================================================================================================


@dataclass(eq=0)
class SpinupSacAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically
    optimizer_actor: str = "adam"  # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"  # one of ["adam", "adamw", "sgd"]
    betas_actor: tuple = None  # for Adam and AdamW
    betas_critic: tuple = None  # for Adam and AdamW
    l2_actor: float = None  # weight decay
    l2_critic: float = None  # weight decay

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function:

        self.optimizer_actor = self.optimizer_actor.lower()
        self.optimizer_critic = self.optimizer_critic.lower()
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:
            logging.warning(f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd")
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:
            logging.warning(f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd")
        if self.optimizer_actor == "adam":
            pi_optimizer_cls = Adam
        elif self.optimizer_actor == "adamw":
            pi_optimizer_cls = AdamW
        else:
            pi_optimizer_cls = SGD
        pi_optimizer_kwargs = {"lr": self.lr_actor}
        if self.optimizer_actor in ["adam, adamw"] and self.betas_actor is not None:
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)
        if self.l2_actor is not None:
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor

        if self.optimizer_critic == "adam":
            q_optimizer_cls = Adam
        elif self.optimizer_critic == "adamw":
            q_optimizer_cls = AdamW
        else:
            q_optimizer_cls = SGD
        q_optimizer_kwargs = {"lr": self.lr_critic}
        if self.optimizer_critic in ["adam, adamw"] and self.betas_critic is not None:
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)
        if self.l2_critic is not None:
            q_optimizer_kwargs["weight_decay"] = self.l2_critic

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), **q_optimizer_kwargs)

        # entropy coefficient:

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        # Run one gradient descent step for Q1 and Q2

        # loss_q:

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Next run one gradient descent step for actor.

        # loss_pi:

        # pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # FIXME: remove debug info
        with torch.no_grad():

            if not cfg.DEBUG_MODE:
                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                )
            else:
                q1_o2_a2 = self.model.q1(o2, a2)
                q2_o2_a2 = self.model.q2(o2, a2)
                q1_targ_pi = self.model_target.q1(o, pi)
                q2_targ_pi = self.model_target.q2(o, pi)
                q1_targ_a = self.model_target.q1(o, a)
                q2_targ_a = self.model_target.q2(o, a)

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()

                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                    # debug:
                    debug_log_pi=logp_pi.detach().mean().item(),
                    debug_log_pi_std=logp_pi.detach().std().item(),
                    debug_logp_a2=logp_a2.detach().mean().item(),
                    debug_logp_a2_std=logp_a2.detach().std().item(),
                    debug_q_a1=q_pi.detach().mean().item(),
                    debug_q_a1_std=q_pi.detach().std().item(),
                    debug_q_a1_targ=q_pi_targ.detach().mean().item(),
                    debug_q_a1_targ_std=q_pi_targ.detach().std().item(),
                    debug_backup=backup.detach().mean().item(),
                    debug_backup_std=backup.detach().std().item(),
                    debug_q1=q1.detach().mean().item(),
                    debug_q1_std=q1.detach().std().item(),
                    debug_q2=q2.detach().mean().item(),
                    debug_q2_std=q2.detach().std().item(),
                    debug_diff_q1=diff_q1_backup.mean().item(),
                    debug_diff_q1_std=diff_q1_backup.std().item(),
                    debug_diff_q2=diff_q2_backup.mean().item(),
                    debug_diff_q2_std=diff_q2_backup.std().item(),
                    debug_diff_r_q1=diff_q1_backup_r.mean().item(),
                    debug_diff_r_q1_std=diff_q1_backup_r.std().item(),
                    debug_diff_r_q2=diff_q2_backup_r.mean().item(),
                    debug_diff_r_q2_std=diff_q2_backup_r.std().item(),
                    debug_diff_q1pt_qpt=diff_q1pt_qpt.mean().item(),
                    debug_diff_q2pt_qpt=diff_q2pt_qpt.mean().item(),
                    debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean().item(),
                    debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean().item(),
                    debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean().item(),
                    debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean().item(),
                    debug_diff_q1_q1t_a=diff_q1_q1t_a.mean().item(),
                    debug_diff_q2_q2t_a=diff_q2_q2t_a.mean().item(),
                    debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std().item(),
                    debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std().item(),
                    debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std().item(),
                    debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std().item(),
                    debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std().item(),
                    debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std().item(),
                    debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std().item(),
                    debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std().item(),
                    debug_r=r.detach().mean().item(),
                    debug_r_std=r.detach().std().item(),
                    debug_d=d.detach().mean().item(),
                    debug_d_std=d.detach().std().item(),
                    debug_a_0=a[:, 0].detach().mean().item(),
                    debug_a_0_std=a[:, 0].detach().std().item(),
                    debug_a_1=a[:, 1].detach().mean().item(),
                    debug_a_1_std=a[:, 1].detach().std().item(),
                    debug_a_2=a[:, 2].detach().mean().item(),
                    debug_a_2_std=a[:, 2].detach().std().item(),
                    debug_a1_0=pi[:, 0].detach().mean().item(),
                    debug_a1_0_std=pi[:, 0].detach().std().item(),
                    debug_a1_1=pi[:, 1].detach().mean().item(),
                    debug_a1_1_std=pi[:, 1].detach().std().item(),
                    debug_a1_2=pi[:, 2].detach().mean().item(),
                    debug_a1_2_std=pi[:, 2].detach().std().item(),
                    debug_a2_0=a2[:, 0].detach().mean().item(),
                    debug_a2_0_std=a2[:, 0].detach().std().item(),
                    debug_a2_1=a2[:, 1].detach().mean().item(),
                    debug_a2_1_std=a2[:, 1].detach().std().item(),
                    debug_a2_2=a2[:, 2].detach().mean().item(),
                    debug_a2_2_std=a2[:, 2].detach().std().item(),
                )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# REDQ-SAC =============================================================================================================

@dataclass(eq=0)
class REDQSACAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.REDQMLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning
    learn_entropy_coef: bool = True
    target_entropy: float = None  # if None, the target entropy is set automatically
    n: int = 10  # number of REDQ parallel Q networks
    m: int = 2  # number of REDQ randomly sampled target networks
    q_updates_per_policy_update: int = 1  # in REDQ, this is the "UTD ratio" (20), this interplays with lr_actor

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer_list = [Adam(q.parameters(), lr=self.lr_critic) for q in self.model.qs]
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)

        self.i_update = 0  # for UTD ratio

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        o, a, r, o2, d, _ = batch

        if update_policy:
            pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        loss_alpha = None
        if self.learn_entropy_coef and update_policy:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        q_prediction_list = [q(o, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)  # * self.n  # averaged for homogeneity with SAC

        for q in self.q_optimizer_list:
            q.zero_grad()
        loss_q.backward()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            qs_pi = [q(o, pi) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            for q in self.model.qs:
                q.requires_grad_(True)

        for q_optimizer in self.q_optimizer_list:
            q_optimizer.step()

        if update_policy:
            self.pi_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            self.loss_pi = loss_pi.detach()
        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# Taste the RAINBOW =============================================================================================================

@dataclass(eq=0)
class RAINBOWAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.DQN
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning
    learn_entropy_coef: bool = True
    target_entropy: float = None  # if None, the target entropy is set automatically

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        model = self.model_cls(observation_space, action_space, device=device)
        logging.debug(f" device RAIN: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        with torch.no_grad():
            for param in self.model_target.parameters():
                param.requires_grad = False
        self.criterion = torch.nn.MSELoss()
        self.optimiser = Adam(self.model.actor.parameters(), lr=self.lr_actor, eps=1.5e-4)

        self.V_min = -50 # Minimum reward
        self.V_max = 700  # Maximum reward
        self.atoms = 500
        self.iter = 0
        self.batch_size=256

        self.support = torch.linspace(self.V_min, self.V_max, self.atoms).to(device=device)  # Support (range) of z
        self.delta_z = (self.V_max - self.V_min) / (self.atoms - 1)
        torch.autograd.set_detect_anomaly(True)

    def get_actor(self):
        return self.model_nograd.actor
    
    def train(self, batch):
        #batch data
        o, a, r, o2, d, _ = batch
        # o = prev observation
        # a = action
        # r = reward
        # o2 = new observation
        # d = terminated

        arrays = [
                np.array([0, 0, 0]),
                np.array([0, 0, -1]),
                np.array([0, 0, 1]),
                np.array([0, 1, 0]),
                np.array([0, 1, -1]),
                np.array([0, 1, 1]),
                np.array([1, 0, 0]),
                np.array([1, 0, -1]),
                np.array([1, 0, 1]),
                np.array([1, 1, 0]),
                np.array([1, 1, -1]),
                np.array([1, 1, 1])
            ]
        
        # Convert the list of arrays to a single NumPy array
        arrays_np = np.array(arrays, dtype=np.float32)

        # Convert the NumPy array to a PyTorch tensor
        arrays_tensor = torch.tensor(arrays_np).to(device=self.device)

        def batched_find_nearest_array_indices(input_tensor, arrays_tensor, batch_size=256):
            num_input = input_tensor.shape[0]
            nearest_indices = []
            
            with torch.no_grad():
                for start in range(0, num_input, batch_size):
                    end = min(start + batch_size, num_input)
                    batch_input = input_tensor[start:end]
                    
                    # Compute squared distances using broadcasting
                    distances = torch.sum((arrays_tensor.unsqueeze(0) - batch_input.unsqueeze(1))**2, dim=2)
                    
                    # Find nearest indices for each element in the batch
                    batch_nearest_indices = torch.argmin(distances, dim=1)
                    nearest_indices.append(batch_nearest_indices)
            
            # Concatenate all nearest indices from batches
            return torch.cat(nearest_indices)

        a_i = batched_find_nearest_array_indices(a, arrays_tensor)
        # print(a)
        # print(a_i)


        #get actor's log prob
        _, log_pi = self.model.actor(o)

        # log_pi_gas = log_pi[range(self.batch_size),a[:,0]]
        # log_pi_brake = log_pi[range(self.batch_size),a[:,1]]
        # log_pi_steer = log_pi[range(self.batch_size),a[:,2]+1]

        log_pi_a = log_pi[range(self.batch_size),a_i[:]]


        _, tlog_pi = self.model_target.actor(o)

        # tlog_pi_gas = tlog_pi[range(self.batch_size),a[:,0]]
        # tlog_pi_brake = tlog_pi[range(self.batch_size),a[:,1]]
        # tlog_pi_steer = tlog_pi[range(self.batch_size),a[:,2]+1]

        tlog_pi_a = tlog_pi[range(self.batch_size),a_i[:]]

        # print(log_pi.shape)
        # print(log_pi_brake.shape)
        # print(log_pi_gas.shape)
        # print(log_pi_steer.shape)

        with torch.no_grad():
            pns, _ = self.model.actor(o2)
            dns = self.support.expand_as(pns) * pns
            summed = dns.sum(2)

            # print(summed.shape)

            # Extract argmax indices along specific ranges of `summed` tensor
            argmax_indices_a = torch.argmax(summed[:], dim=1)

            # argmax_indices_gas = torch.argmax(summed[:, :2], dim=1)
            # argmax_indices_brake = torch.argmax(summed[:, 2:4], dim=1)
            # argmax_indices_steer = torch.argmax(summed[:, 4:], dim=1)

            # print(argmax_indices_brake)


            self.model_target.reset_noise()
            pns, _ = self.model_target.actor(o2)
            pns_a = pns[range(self.batch_size),argmax_indices_a]
            # pns_gas = pns[range(self.batch_size),argmax_indices_gas]
            # pns_brake = pns[range(self.batch_size),argmax_indices_brake+2]
            # pns_steer = pns[range(self.batch_size),argmax_indices_steer+4]

            discount = 0.99
            n = 0

            nonterminals = (d == 0).to(torch.float32).to(device=self.device)

            # print(nonterminals)

            # print(self.support.shape)
            # print(self.support.unsqueeze(0))
            # print(self.support.shape)

            
            discount_factor = discount**n
            discounted_values = nonterminals.float() * discount_factor
            #  # Compute Tz (Bellman operator T applied to z)
            expand_r = r.unsqueeze(1)
            expand_support = self.support.unsqueeze(0)


            # print(expand_r.shape)
            # print(discounted_values.shape)
            # print(expand_support.shape)


            Tz = expand_r + discounted_values.unsqueeze(1) * expand_support  # Tz = R^n + (γ^n)z (accounting for terminal states)
            
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)  # Clamp between supported values

            # print(Tz.shape)

            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.V_min) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # print(u)
            # Distribute probability of Tz

            m = torch.zeros(self.batch_size, self.atoms, device=self.device, dtype = o[1].dtype)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(device=self.device)

            # Helper function to perform index_add_ operation
            def update_tensor_with_indices(tensor, indices, values):
                # Flatten the tensor to apply index_add_
                flat_tensor = tensor.view(-1)
                # Convert indices to int64 dtype
                indices = indices.view(-1).to(torch.int64)
                # Perform index_add_ operation
                flat_tensor.index_add_(0, indices, values.view(-1))

            update_tensor_with_indices(m, (l + offset), (pns_a * (u.float() - b)))
            update_tensor_with_indices(m, (u + offset), (pns_a * (b - l.float())))

            # # Update `m_gas` tensor with indices and values
            # update_tensor_with_indices(m, (l + offset), (pns_gas * (u.float() - b)))
            # update_tensor_with_indices(m, (u + offset), (pns_gas * (b - l.float())))

            # # Update `m_brake` tensor with indices and values
            # update_tensor_with_indices(m, (l + offset), (pns_brake * (u.float() - b)))
            # update_tensor_with_indices(m, (u + offset), (pns_brake * (b - l.float())))

            # # Update `m_steer` tensor with indices and values
            # update_tensor_with_indices(m, (l + offset), (pns_steer * (u.float() - b)))
            # update_tensor_with_indices(m, (u + offset), (pns_steer * (b - l.float())))
        
        # loss = (-torch.sum(m * log_pi_gas, 1)) + (-torch.sum(m * log_pi_brake, 1))  + (-torch.sum(m * log_pi_steer, 1)) # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        # tloss = (-torch.sum(m * tlog_pi_gas, 1)) + (-torch.sum(m * tlog_pi_brake, 1))  + (-torch.sum(m * tlog_pi_steer, 1))
        loss = (-torch.sum(m * log_pi_a, 1))
        tloss = (-torch.sum(m * tlog_pi_a, 1))
        
        # print(loss.mean())
        self.model.actor.zero_grad()
        loss.mean().backward()

        norm_clip = 10
        clip_grad_norm_(self.model.actor.parameters(), norm_clip)
        self.optimiser.step()

        self.iter+=1
        if((self.iter%1000)==0):
            #update target
            self.model_target.actor.load_state_dict(self.model.actor.state_dict())
            self.iter=0

        ret_dict = dict(
                    loss_actor=loss.mean().item(),
                    loss_critic=tloss.mean().item(),
                )
        return ret_dict
        