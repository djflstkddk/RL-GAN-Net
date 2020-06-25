import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from torch.distributions.normal import Normal
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)  # 400
        self.l2 = nn.Linear(400, 400)
        self.l2_additional = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(400, action_dim)

        self.max_action = max_action

    def forward(self, x, deterministic=False, repara_trick=False, with_logprob=True):
        x = F.relu(self.l1(x))
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        x = F.relu(self.l2(x))
        x = F.relu(self.l2_additional(x))
        mu = self.max_action * torch.tanh(self.l3(x))
        pi_distribution = Normal(mu, std)

        if deterministic:
            pi_action = mu
        elif repara_trick:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = pi_distribution.sample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis = -1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.max_action*pi_action

        return pi_action, logp_pi





class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3_additional = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3_additional(x)
        x = self.l3(x)
        return x


class SoftAC(object):
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actor_out = self.actor(state, deterministic=deterministic)
        return actor_out[0].cpu().data.numpy().flatten(), actor_out[1].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001, alpha = 0.1):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            action_tilde, log_pi_tilde = self.actor(next_state, repara_trick=False)
            log_pi_tilde = log_pi_tilde.unsqueeze(1)
            target_Q = self.critic_target(next_state, action_tilde)
            target_Q = reward + (done * discount * (target_Q - alpha * log_pi_tilde)).detach() # added entropy

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            tmp_action, tmp_log_pi = self.actor(state, repara_trick=True)
            tmp_log_pi = tmp_log_pi.unsqueeze(1)
            actor_loss = (-self.critic(state, tmp_action) + alpha * tmp_log_pi).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))