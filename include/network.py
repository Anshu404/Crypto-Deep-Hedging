# from include.settings import getSettings

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Actor(nn.Module):
#     def __init__(self, state_dim, settings = getSettings()):
#         super(Actor, self).__init__()
        
#         self.lrelu_alpha = settings['lrelu_alpha']
#         actor_nn = settings['actor_nn']

#         self.input = nn.Linear(state_dim, actor_nn)
#         self.hidden = nn.Linear(actor_nn, actor_nn)
#         self.output = nn.Linear(actor_nn, 1)

#     def forward(self, state):
#         x = F.leaky_relu(self.input(state), self.lrelu_alpha)
#         x = F.leaky_relu(self.hidden(x), self.lrelu_alpha)
#         x = torch.tanh(self.output(x))

#         return x

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, settings = getSettings()):
#         super(Critic, self).__init__()
        
#         self.lrelu_alpha = settings['lrelu_alpha']
#         critic_nn = settings['critic_nn']
                
#         self.input = nn.Linear(state_dim + action_dim, critic_nn)
#         self.hidden1 = nn.Linear(critic_nn, critic_nn)
#         self.hidden2 = nn.Linear(critic_nn, critic_nn)
#         self.output = nn.Linear(critic_nn, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = F.leaky_relu(self.input(x), self.lrelu_alpha)
#         x = F.leaky_relu(self.hidden1(x), self.lrelu_alpha)
#         x = F.leaky_relu(self.hidden2(x), self.lrelu_alpha)
#         x = self.output(x)
        
#         return x


from include.settings import getSettings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Device setup for GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, settings = getSettings()):
        super(Actor, self).__init__()
        
        self.lrelu_alpha = settings['lrelu_alpha']
        actor_nn = settings['actor_nn']

        # UPGRADE 1: Deeper Network with Shock Absorbers (LayerNorm)
        # Agent ke 4 pure inputs yahan aayenge
        self.input = nn.Linear(state_dim, actor_nn)
        self.ln1 = nn.LayerNorm(actor_nn)
        
        self.hidden1 = nn.Linear(actor_nn, actor_nn)
        self.ln2 = nn.LayerNorm(actor_nn)
        
        self.hidden2 = nn.Linear(actor_nn, actor_nn)
        self.ln3 = nn.LayerNorm(actor_nn)
        
        # Output will be exactly 1 action (delta position)
        self.output = nn.Linear(actor_nn, 1)

    def forward(self, state):
        # LayerNorm is applied BEFORE activation to stabilize extreme crypto jumps
        x = F.leaky_relu(self.ln1(self.input(state)), self.lrelu_alpha)
        x = F.leaky_relu(self.ln2(self.hidden1(x)), self.lrelu_alpha)
        x = F.leaky_relu(self.ln3(self.hidden2(x)), self.lrelu_alpha)
        
        # Tanh keeps the output between -1 and 1 (100% short to 100% long)
        x = torch.tanh(self.output(x))

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, settings = getSettings()):
        super(Critic, self).__init__()
        
        self.lrelu_alpha = settings['lrelu_alpha']
        critic_nn = settings['critic_nn']
                
        # UPGRADE 2: Highly robust Critic to judge actions accurately in Heston
        self.input = nn.Linear(state_dim + action_dim, critic_nn)
        self.ln1 = nn.LayerNorm(critic_nn)
        
        self.hidden1 = nn.Linear(critic_nn, critic_nn)
        self.ln2 = nn.LayerNorm(critic_nn)
        
        # Funneling down architecture to extract pure value
        self.hidden2 = nn.Linear(critic_nn, int(critic_nn / 2)) 
        self.ln3 = nn.LayerNorm(int(critic_nn / 2))
        
        self.hidden3 = nn.Linear(int(critic_nn / 2), int(critic_nn / 2))
        self.ln4 = nn.LayerNorm(int(critic_nn / 2))
        
        # Output is the predicted Q-Value (Expected Future Reward)
        self.output = nn.Linear(int(critic_nn / 2), 1)

    def forward(self, state, action):
        # State aur Action ko jod kar evaluate karte hain
        x = torch.cat([state, action], 1)
        
        x = F.leaky_relu(self.ln1(self.input(x)), self.lrelu_alpha)
        x = F.leaky_relu(self.ln2(self.hidden1(x)), self.lrelu_alpha)
        x = F.leaky_relu(self.ln3(self.hidden2(x)), self.lrelu_alpha)
        x = F.leaky_relu(self.ln4(self.hidden3(x)), self.lrelu_alpha)
        
        x = self.output(x)
        
        return x