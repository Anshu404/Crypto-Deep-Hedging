import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device setup for Apple Silicon / CUDA / CPU
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ─────────────────────────────────────────
# 1. CAUSAL TRANSFORMER ENCODER (The Temporal Eye)
# ─────────────────────────────────────────
class PreNormTransformerLayer(nn.Module):
    """Pre-norm: LayerNorm BEFORE attention/FFN for stable RL gradients."""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # GELU outperforms ReLU in transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        # Pre-norm attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=False)
        x = x + attn_out
        
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x

class CausalTransformerEncoder(nn.Module):
    """
    Processes the [Moneyness, TTM, Inventory, IV] window and extracts 
    a single context vector (h_T) representing the market history.
    """
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2, d_ff=256, dropout=0.1, seq_len=20):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Project raw 4-feature state into d_model space
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learned positional encodings
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Build causal mask once to save computation
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        
        # Stack of pre-norm transformer layers
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, 4) — observation window
        returns: (batch, d_model) — context vector h_T
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        
        h = self.input_proj(x) + self.pos_embedding(positions)     # (B, T, d_model)
        
        for layer in self.layers:
            h = layer(h, self.causal_mask[:T, :T])
            
        h = self.final_norm(h)
        
        # Extract the last token (h_T) - this summarizes the entire sequence
        return h[:, -1, :]
    


# ─────────────────────────────────────────
# 2. TD3 ACTOR (The Policy Network)
# ─────────────────────────────────────────
class TransformerActor(nn.Module):
    """
    Receives the extracted market history (h_T) from the Transformer,
    and decides the optimal Hedge Ratio (Delta) between -1 and 1.
    Includes LayerNorm to absorb extreme market shocks.
    """
    def __init__(self, d_model=64, hidden=128, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden, int(hidden/2)),
            nn.LayerNorm(int(hidden/2)),
            nn.LeakyReLU(0.01),
            
            nn.Linear(int(hidden/2), 1),
            nn.Tanh()
        )

    def forward(self, h_T):
        # Tanh bounds output strictly to [-1, 1] (100% Short to 100% Long)
        return self.max_action * self.net(h_T)

# ─────────────────────────────────────────
# 3. TD3 TWIN CRITICS (The Evaluators)
# ─────────────────────────────────────────
class TransformerCritic(nn.Module):
    """
    Evaluates how good an action is given the market history (h_T).
    Uses Twin Q-Networks to prevent overestimation bias (a core TD3 feature).
    """
    def __init__(self, d_model=64, action_dim=1, hidden=128):
        super().__init__()
        
        # Q1 Architecture
        self.q1 = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden, int(hidden/2)),
            nn.LayerNorm(int(hidden/2)),
            nn.LeakyReLU(0.01),
            
            nn.Linear(int(hidden/2), 1)
        )
        
        # Q2 Architecture
        self.q2 = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden, int(hidden/2)),
            nn.LayerNorm(int(hidden/2)),
            nn.LeakyReLU(0.01),
            
            nn.Linear(int(hidden/2), 1)
        )

    def forward(self, h_T, action):
        # Concatenate history embedding (d_model) and action (1)
        x = torch.cat([h_T, action], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, h_T, action):
        # Used specifically during Actor updates to save computation
        x = torch.cat([h_T, action], dim=-1)
        return self.q1(x)
    


import copy

# ─────────────────────────────────────────
# 4. FULL TRANSFORMER-TD3 AGENT (The Master Wrapper)
# ─────────────────────────────────────────
class TransformerTD3Agent:
    """
    Wraps the Encoders, Actor, and Critics into a single Agent.
    Handles the TD3 training loop, delayed policy updates, and soft target updates.
    """
    def __init__(self, state_dim=4, d_model=64, seq_len=20, max_action=1.0, lr=3e-4):
        self.max_action = max_action
        self.seq_len = seq_len

        # 1. LIVE NETWORKS (What the agent currently uses)
        self.encoder_actor = CausalTransformerEncoder(input_dim=state_dim, d_model=d_model, seq_len=seq_len).to(device)
        self.actor = TransformerActor(d_model=d_model, max_action=max_action).to(device)
        
        self.encoder_critic = CausalTransformerEncoder(input_dim=state_dim, d_model=d_model, seq_len=seq_len).to(device)
        self.critic = TransformerCritic(d_model=d_model).to(device)

        # 2. TARGET NETWORKS (Slowly moving copies for stable training)
        self.encoder_actor_target = copy.deepcopy(self.encoder_actor)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.encoder_critic_target = copy.deepcopy(self.encoder_critic)
        self.critic_target = copy.deepcopy(self.critic)

        # 3. OPTIMIZERS (Notice how we update the encoder + network together)
        self.actor_optimizer = torch.optim.Adam(
            list(self.encoder_actor.parameters()) + list(self.actor.parameters()), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.encoder_critic.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.total_it = 0

    def select_action(self, sequence_window):
        """
        sequence_window: numpy array of shape (seq_len, state_dim)
        Used during environment interaction.
        """
        with torch.no_grad():
            # Add batch dimension and move to device
            seq_tensor = torch.FloatTensor(sequence_window).unsqueeze(0).to(device)
            
            # Extract history vector h_T and get action
            h_T = self.encoder_actor(seq_tensor)
            action = self.actor(h_T)
            
        return action.cpu().data.numpy().flatten()

    def train_step(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005, 
                   policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        The core TD3 update loop.
        """
        self.total_it += 1

        # Sample a batch of SEQUENCES from our optimized buffer
        windows, actions, rewards, next_windows, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Target Policy Smoothing (Adds clipped noise to target actions)
            noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
            
            h_next = self.encoder_actor_target(next_windows)
            next_action = (self.actor_target(h_next) + noise).clamp(-self.max_action, self.max_action)

            # Compute Target Q-Values using Twin Critics (Min Trick)
            h_next_c = self.encoder_critic_target(next_windows)
            target_Q1, target_Q2 = self.critic_target(h_next_c, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * discount * target_Q

        # Get Current Q-Values
        h_curr_c = self.encoder_critic(windows)
        current_Q1, current_Q2 = self.critic(h_curr_c, actions)

        # CRITIC UPDATE
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient Clipping to prevent math explosions during extreme market crashes
        torch.nn.utils.clip_grad_norm_(list(self.encoder_critic.parameters()) + list(self.critic.parameters()), 1.0)
        self.critic_optimizer.step()

        # DELAYED ACTOR UPDATE (Updates only every 'policy_freq' steps)
        if self.total_it % policy_freq == 0:
            
            h_curr_a = self.encoder_actor(windows)
            # Critic evaluates the Actor's proposed action. We detach the critic's encoder so it doesn't get updated here.
            h_curr_c_detached = self.encoder_critic(windows).detach()
            actor_loss = -self.critic.Q1(h_curr_c_detached, self.actor(h_curr_a)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder_actor.parameters()) + list(self.actor.parameters()), 1.0)
            self.actor_optimizer.step()

            # SOFT UPDATES (Slowly update target networks)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.encoder_critic.parameters(), self.encoder_critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.encoder_actor.parameters(), self.encoder_actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename):
        if not filename.endswith('.pth'):
            filename += ".pth"
        torch.save({
            'encoder_actor': self.encoder_actor.state_dict(),
            'actor': self.actor.state_dict(),
            'encoder_critic': self.encoder_critic.state_dict(),
            'critic': self.critic.state_dict(),
        }, filename)

    def load(self, filename):
        if not filename.endswith('.pth'):
            filename += ".pth"
        checkpoint = torch.load(filename, map_location=device)
        self.encoder_actor.load_state_dict(checkpoint['encoder_actor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.encoder_critic.load_state_dict(checkpoint['encoder_critic'])
        self.critic.load_state_dict(checkpoint['critic'])