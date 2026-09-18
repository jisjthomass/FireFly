import torch
import torch.nn as nn
from typing import Optional, Tuple

class SparseSynapticLayer(nn.Module):
    def __init__(self, sparse_adjacency: torch.Tensor) -> None:
        super().__init__()
        self.num_neurons = sparse_adjacency.shape[0]
        adj_coalesced = sparse_adjacency.coalesce()
        self.register_buffer("edge_indices", adj_coalesced.indices())
        nnz = adj_coalesced._nnz()
        self.edge_weights = nn.Parameter(
            torch.randn(nnz) * (2.0 / self.num_neurons) ** 0.5
        )
        self._cached_W = None
        self._last_weight_sum = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current_weight_sum = self.edge_weights.sum().item()

        if self._cached_W is None or self._last_weight_sum != current_weight_sum:
            self._cached_W = torch.sparse_coo_tensor(
                self.edge_indices,
                self.edge_weights,
                size=(self.num_neurons, self.num_neurons),
                device=x.device,
            )
            self._last_weight_sum = current_weight_sum

        return torch.sparse.mm(self._cached_W.t(), x.t()).t()

class FlyBrainNet(nn.Module):
    def __init__(self, input_features, num_neurons, num_classes, sparse_adjacency):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_classes = num_classes
        self.register_buffer("adjacency_mask", sparse_adjacency.coalesce())
        self.sensory_layer = nn.Sequential(nn.Linear(input_features, num_neurons), nn.LayerNorm(num_neurons))
        self.synaptic_layer = SparseSynapticLayer(sparse_adjacency)
        self.motor_layer = nn.Sequential(nn.Linear(num_neurons, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x, previous_brain_state=None):
        batch_size = x.size(0)
        sensory_input = self.sensory_layer(x)
        
        if previous_brain_state is None:
            previous_brain_state = sensory_input
            
        recurrent_input = self.synaptic_layer(previous_brain_state)
        current_brain_state = torch.tanh(sensory_input + recurrent_input)
        prediction = self.motor_layer(current_brain_state)
        
        return prediction, current_brain_state.detach()
