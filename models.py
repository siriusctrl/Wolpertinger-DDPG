import numpy as np
import torch
from torch._C import device
import torch.nn as nn
from torch.optim import Adam

from explore import OUActionNoise
from memory import SequentialMemory
from action_space import Space, Discrete_space

class Actor(nn.Module):
    def __init__(self, n_state, n_actions) -> None:
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, n_state, n_actions) -> None:
        super(Critic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(256 + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    
    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

action_lr = 1e-4
cirtic_lr = 1e-3
OU_STD = 0.2


class DDPG_Agent():
    def __init__(self, n_states, n_actions) -> None:
        # assuming up to one GPU used
        self.USE_CUDA = torch.cuda.is_available()
        print("Using CUDA", self.USE_CUDA)
        self.device = torch.device("cuda" if self.USE_CUDA else "cpu")

        self.n_states = n_states
        self.n_actions = n_actions

        # action network
        # ! I removed the .double from here, don't know the impact
        self.actor = Actor(self.n_states, self.n_actions).to(self.device)
        self.actor_target = Actor(self.n_states, self.n_actions)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.to(self.device)
        # ! remove decay here
        self.actor_optim = Adam(self.actor.parameters(), lr=action_lr)

        # critic network
        self.critic = Critic(self.n_states, self.n_actions).to(self.device)
        self.critic_target = Critic(self.n_states, self.n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.to(self.device)
        # ! remove decay here
        self.critic_optim = Adam(self.critic.parameters(), lr=cirtic_lr)

        # create replay buffer
        self.buffer = SequentialMemory(limit=100000, window_length=1)
        # setup explorer sampler
        self.explorer = OUActionNoise(mean=np.zeros(1), std_deviation=OU_STD * np.ones(1))

        self.batch_size = 64
        # target running average
        # ! he use 0.001
        self.tau = 0.005
        # the discount factor
        self.gamma = 0.99

        # Linear decay rate of exploration policy
        self.epsilon_decay = 1.0 / 80000
        # initial exploration rate
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.continuous_action_space = False
    
    def eval(self):
        """
        freeze the model for evaluation
        """
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def observe(self, r_t, s_t_next, done):
        if self.is_training:
            self.buffer.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t_next

    def random_action(self):
        return np.random.uniform(-1., 1., self.n_actions)
    
    def reset(self, s_t):
        """
        reset the state and explore
        """
        self.s_t = s_t
        self.explorer.reset()
    
    def save_model(self, destination: str) -> None:
        torch.save(self.actor.state_dict(), '{}/actor.pt'.format(destination))
        torch.save(self.critic.state_dict(), '{}/critic.pt'.format(destination))
    
    def load_model(self, target: str, location=None) -> None:
        """
        load the model to agent, notice that if you want to load model 
        cross device, the location could be either 'cpu' or 'gpu'
        """

        if target is None:
            return
        
        if location is None:
            self.actor.load_state_dict(
                torch.load('{}/actor.pt'.format(target))
            )

            self.critic.load_state_dict(
                torch.load('{}/critic.pt'.format(target))
            )

        elif location == 'cpu':
            device = torch.device('cpu')
            self.actor.load_state_dict(
                torch.load('{}/actor.pt'.format(target),
                    map_location=device
                )
            )

            self.critic.load_state_dict(
                torch.load('{}/critic.pt'.format(target),
                    map_location=device
                ),
            )
        elif location == 'gpu':
            device = torch.device('cuda')
            self.actor.load_state_dict(
                torch.load('{}/actor.pt'.format(target),
                    map_location=device
                )
            )

            self.critic.load_state_dict(
                torch.load('{}/critic.pt'.format(target),
                    map_location=device
                ),
            )

            self.actor.to(device)
            self.critic.to(device)

        else:
            raise NameError('The target device should be either cpu or gpu')
    
    def seed(self, seed):
        torch.manual_seed(seed)


class Wolpertinger_Agent(DDPG_Agent):

    def __init__(self, space_type, max_actions, 
                        action_low, actions_high, 
                        n_states, n_actions, k_ratio=0.1) -> None:

        super().__init__(n_states, n_actions)

        if space_type is 'continuous':
            self.action_space = Space(action_low, actions_high, max_actions)
            self.k_nearest_neighbors = max(1, int(max_actions*k_ratio))
        elif space_type is 'discrete':
            self.action_space = Discrete_space(max_actions)
            self.k_nearest_neighbors = max(1, max_actions * k_ratio)
        elif space_type is 'binary':
            raise NameError('Not supported yet')
        else:
            raise NameError('Unknown space type')
    
    def get_name(self):
        from datetime import date
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, str(date.today()))
    
    def get_action_space(self):
        return self.action_space
    
    def wolp_action(self, s_t, proto_action):
        raise NotImplementedError

    def select_action(self, s_t, decay_epsilon=True):
        raise NotImplementedError

    def select_target_action(self, s_t):
        raise NotImplementedError

    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update_policy(self):
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.buffer.sample_and_split(self.batch_size)

        next_state_batch = torch.FloatTensor(next_state_batch, device=device)
        next_wolp_action_batch = torch.FloatTensor(
            self.select_target_action(next_state_batch), 
            device=device)

        next_q_values = self.critic_target([
            next_state_batch,
            next_wolp_action_batch
        ])

        target_q_batch = torch.FloatTensor(reward_batch, device=device) + self.gamma * \
                         torch.FloatTensor(terminal_batch.astype(np.float32), device=device) * \
                         next_q_values
        
        self.critic_optim.zero_grad()

        state_batch = torch.FloatTensor(state_batch, device=device)
        action_batch = torch.FloatTensor(action_batch, device=device)
        q_batch = self.critic([state_batch, action_batch])

        value_loss = nn.MSELoss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        
        actor_loss = -self.critic([state_batch, self.actor(state_batch)])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        

