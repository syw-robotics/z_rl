"""Device-resident time/environment replay with explicit continuity boundaries."""

import torch


class SequenceReplayBuffer:
    def __init__(self, capacity_steps, num_envs, device):
        self.capacity = capacity_steps
        self.num_envs = num_envs
        self.device = device
        self.data = {}
        self.size = 0
        self.position = 0

    @torch.no_grad()
    def add(self, transition):
        if not self.data:
            self.data = {
                k: torch.empty_like(v.expand(self.capacity, *v.shape), device=self.device)
                for k, v in transition.items()
            }
        for key, destination in self.data.items():
            destination[self.position].copy_(transition[key])
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_windows(self, batch_size, length, fields=None):
        if not 1 <= length <= self.size:
            raise ValueError("Replay must contain at least one complete window")
        start = torch.randint(self.size - length + 1, (batch_size, 1), device=self.device)
        env = torch.randint(self.num_envs, (batch_size, 1), device=self.device)
        rows = (self.position - self.size + start + torch.arange(length, device=self.device)) % self.capacity
        keys = self.data.keys() if fields is None else fields
        window = {key: self.data[key][rows, env] for key in keys}
        alive = (~self.data["segment_end"][rows, env]).float()
        window["mask"] = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1).cumprod(1)
        return window

    def sample_starts(self, batch_size):
        return {key: value[:, 0] for key, value in self.sample_windows(batch_size, 1).items() if key != "mask"}
