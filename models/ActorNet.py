
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

__all__= ['actor_net']

class ActorNet(nn.Module):
    def __init__(self, args):
        super(ActorNet, self).__init__()
        state_dim = args.state_dim
        action_dim = args.z_dim
        max_action = args.max_action
        self.args =args
        self.l1 = nn.Linear(state_dim, 400)  # 400
        self.l2 = nn.Linear(400, 400)
        self.l2_additional = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.FloatTensor(x.reshape(1, -1)).to(self.args.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l2_additional(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x.cpu().data.numpy().flatten()


class ActorNetInSAC(nn.Module):
    def __init__(self, args):
        super(ActorNetInSAC, self).__init__()
        self.args = args
        state_dim = args.state_dim
        action_dim = args.z_dim
        max_action = args.max_action
        self.l1 = nn.Linear(state_dim, 400)  # 400
        self.l2 = nn.Linear(400, 400)
        self.l2_additional = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(400, action_dim)

        self.max_action = max_action

    def forward(self, x, deterministic=False, repara_trick=False, with_logprob=True):
        x = torch.FloatTensor(x.reshape(1, -1)).to(self.args.device)
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

        return pi_action.cpu().data.numpy().flatten()


def actor_net(args,data=None):

    if args.policy_name == "DDPG":
        model = ActorNet(args)
    elif args.policy_name == "SoftAC":
        model = ActorNetInSAC(args)
    else:
        pass

    model.load_state_dict(data)

    return model