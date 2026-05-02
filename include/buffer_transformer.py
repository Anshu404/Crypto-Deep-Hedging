import numpy as np
import torch

class EfficientSequenceBuffer:
    """
    Memory-efficient Replay Buffer for Sequence Data (Transformers).
    Instead of duplicating (seq_len, state_dim) arrays,
    we store flat states and slice sequences on the fly using indices.
    Prevents Out-Of-Memory (OOM) crashes on 8GB RAM Macs.
    """
    def __init__(self, capacity=50000, seq_len=20, state_dim=4, device='mps'):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Store flat data - Memory footprint is O(N)
        self.states  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1),         dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),         dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),         dtype=np.float32)
        
        # Track episodes so we don't slice across different episodes
        self.episode_starts = np.zeros(capacity, dtype=bool)

    def push(self, state, action, reward, done, is_start=False):
        self.states[self.ptr]  = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr]   = done
        self.episode_starts[self.ptr] = is_start
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_sequence(self, idx):
        # Extract a sequence of length seq_len ending at idx
        seq = np.zeros((self.seq_len, self.states.shape[1]), dtype=np.float32)
        start_idx = idx - self.seq_len + 1
        
        valid_start = start_idx
        # Check backward for episode starts OR negative indices
        for i in range(idx, start_idx - 1, -1):
            if i < 0 or self.episode_starts[i]:
                valid_start = max(0, i) # Stop at 0 to avoid wrap-around indexing
                break
                
        real_len = idx - valid_start + 1
        seq[-real_len:] = self.states[valid_start : idx + 1]
        
        return seq

    def sample(self, batch_size=256):
        # Sample indices where we have history
        idxs = np.random.randint(0, self.size - 1, size=batch_size)
        
        windows = np.zeros((batch_size, self.seq_len, self.states.shape[1]), dtype=np.float32)
        next_windows = np.zeros((batch_size, self.seq_len, self.states.shape[1]), dtype=np.float32)
        actions = np.zeros((batch_size, 1), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)

        for i, idx in enumerate(idxs):
            windows[i] = self._get_sequence(idx)
            # Next window is the sequence ending at idx + 1
            next_windows[i] = self._get_sequence(idx + 1)
            
            actions[i] = self.actions[idx]
            rewards[i] = self.rewards[idx]
            dones[i]   = self.dones[idx]

        to_tensor = lambda arr: torch.FloatTensor(arr).to(self.device)
        
        return to_tensor(windows), to_tensor(actions), to_tensor(rewards), to_tensor(next_windows), to_tensor(dones)

    def __len__(self):
        return self.size