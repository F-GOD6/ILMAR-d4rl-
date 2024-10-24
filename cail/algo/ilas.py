import os
import gym
import torch
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from .base import Algorithm, Expert
from cail.buffer import Buffer
from cail.utils import soft_update, disable_gradient
from cail.network import StateDependentPolicy, TwinnedStateActionFunction , StateActionFunction , StateIndependentPolicy,StateActionConcatedFunction,StateFunction,StateIndependentState
from cail.buffer import SerializedBuffer
EPS = np.finfo(np.float32).eps
EPS2 = 1e-3
def minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.):
    """
    Implements the Minimax discriminator loss function.
    
    Args:
        expert_cost_val (torch.Tensor): The discriminator's output for real samples.
        union_cost_val (torch.Tensor): The discriminator's output for generated samples.
        label_smoothing (float, optional): The amount of label smoothing to apply. Defaults to 0.
    
    Returns:
        torch.Tensor: The Minimax discriminator loss.
    """
    expert_loss = -torch.mean(torch.log(torch.clamp(expert_cost_val - label_smoothing, min=1e-12, max=1.0)))
    union_loss = -torch.mean(torch.log(torch.clamp(1. - union_cost_val - label_smoothing, min=1e-12, max=1.0)))
    return expert_loss + union_loss
class ILAS(Algorithm):
    """
    Implementation of BC



    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    batch_size: int
        batch size for sampling in the replay buffer
    rollout_length: int
        rollout length of the buffer
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_alpha: float
        learning rate of log(alpha)
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    start_steps: int
        start steps. Training starts after collecting these steps in the environment.
    tau: float
        tau coefficient
    """
        # def __init__(self, state_dim, action_dim, hidden_size=256, actor_lr=1e-4, critic_lr=1e-4,
                #  grad_reg_coef=1.0, stochastic_policy=True, tau=0.0, version="v1"):
    def __init__(
            self,
            buffer_exp: SerializedBuffer,
            buffer_union:SerializedBuffer,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float = 0.99,
            batch_size: int = 256,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_cost:float = 1e-4,
            units_actor: tuple = (256, 256),
            units_critic: tuple = (256, 256),
            units_cost: tuple = (256, 256),
            start_steps: int = 10000,
            tau: float = 0.0,
            grad_reg_coef : float=0.1,
            alpha:float=0.0,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.buffer_exp = buffer_exp
        self.buffer_union = buffer_union
        # actor

        self.actor = StateIndependentPolicy(
            state_shape=tuple(dim * 2 for dim in state_shape),
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.statemaker = StateIndependentState(
            state_shape=state_shape,
            action_shape=state_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_statemaker = Adam(self.statemaker.parameters(),lr=lr_critic)
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.grad_reg_coef = grad_reg_coef
        self.device = device
        self.non_expert_regularization = alpha + 1
    def is_update(self, step: int) -> bool:
        """
        Whether the time is for update

        Parameters
        ----------
        step: int
            current training step

        Returns
        -------
        update: bool
            whether to update. SAC updates when the step is larger
            than the start steps and the batch size
        """
        return step >= max(self.start_steps, self.batch_size)

    def step(self, env: gym.wrappers.TimeLimit, state: np.array, t: int, step: int):
        """
        Sample one step in the environment

        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            environment
        state: np.array
            current state
        t: int
            current time step in the episode
        step: int
            current total steps

        Returns
        -------
        next_state: np.array
            next state
        t: int
            time step
        """
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)[0]
            
        # next_state, reward, done, _ = env.step(action) #新版更改
        next_state, reward, terminated, truncated,_ = env.step(action)
        # mask = True if t == env.max_episode_steps else done #新版更改
        mask = terminated or truncated
        self.buffer.append(state, action, reward, mask, next_state)

        # if done or t == env.max_episode_steps: #新版更改
        if mask :
            t = 0
            next_state = env.reset()[0]

    def exploit(self, state: np.array) -> np.array:
        """
        Act with deterministic policy

        Parameters
        ----------
        state: np.array
            current state

        Returns
        -------
        action: np.array
            action to take
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = self.statemaker(state.unsqueeze_(0))
        inputs = torch.cat([state, next_state], -1)
        with torch.no_grad():
            action = self.actor(inputs)
        return action.cpu().numpy()[0]
    def update(self, writer: SummaryWriter):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1
        expert_states, expert_actions, rewards, dones, expert_next_states,_ = \
            self.buffer_exp.sample(self.batch_size)
        union_states,union_actions, rewards, dones, union_next_states,init_states= \
            self.buffer_union.sample(self.batch_size)
        expert_inputs = torch.cat([expert_states, expert_next_states], -1)
        union_inputs = torch.concat([union_states, union_next_states], -1)
        
        pi_loss = - (self.actor.evaluate_log_pi(union_inputs, union_actions)).mean()
        print(union_actions[0])
        state_loss = - (self.statemaker.evaluate_log_pi(expert_states, expert_next_states)).mean()
        print(expert_next_states[0])
        self.update_actor(pi_loss,writer)
        self.update_statemaker(state_loss,writer)

    

    def update_actor(self, pi_loss,writer: SummaryWriter):
        """
        Update the actor for one step

        Parameters
        ----------
        states: torch.Tensor
            sampled states
        writer: SummaryWriter
            writer for logs
        """

        self.optim_actor.zero_grad()
        pi_loss.backward(retain_graph=False)
        self.optim_actor.step()

        

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', pi_loss.item(), self.learning_steps)

    def update_statemaker(self, pi_loss,writer: SummaryWriter):
        """
        Update the actor for one step

        Parameters
        ----------
        states: torch.Tensor
            sampled states
        writer: SummaryWriter
            writer for logs
        """

        self.optim_statemaker.zero_grad()
        pi_loss.backward(retain_graph=False)
        self.optim_statemaker.step()

        

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/state', pi_loss.item(), self.learning_steps)
        print(pi_loss)
        import time 
        time.sleep(10)
    def update_cost(
            self,
            loss,
            writer: SummaryWriter
    ):
        """
        Update the cost for one step

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
 

        self.optim_cost.zero_grad()
        loss.backward(retain_graph=False)
        self.optim_cost.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/cost', loss.item(), self.learning_steps)

    def update_critic(
            self,
            loss,
            writer: SummaryWriter
    ):
        """
        Update the cost for one step

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
 

        self.optim_critic.zero_grad()
        loss.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic', loss.item(), self.learning_steps)
    def save_models(self, save_dir: str):
        """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
        super().save_models(save_dir)
        # we only save actor to reduce workloads
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )


class ILASExpert(Expert):
    """
    Well-trained SAC agent

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    path: str
        path to the well-trained weights
    units_actor: tuple
        hidden units of the actor
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            path: str,
            units_actor: tuple = (256, 256)
    ):
        super(ILASExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)
