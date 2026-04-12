# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from z_rl.models.mlp_model import MLPModel
from z_rl.modules import CNN, HiddenState, MLP
from z_rl.utils import resolve_nn_activation


class CNNModel(MLPModel):
    """CNN-based neural model.

    Data flow: ``obs groups -> (split 1D/2D) -> (concat 1D -> normalization) + (CNN encodes for 2D) -> optional projection MLP -> concat latent -> head -> (distribution) -> output``.

    This model uses one or more convolutional neural network (CNN) encoders to process one or more 2D observation groups
    before passing the resulting latent to an MLP. Any 1D observation groups are directly concatenated with the CNN
    latent and passed to the MLP. 1D observations can be normalized before being passed to the MLP. The output of the
    model can be either deterministic or stochastic, in which case a distribution module is used to sample the outputs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnn_projection_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        cnn_projectors: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the CNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the CNN and MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            cnn_cfg: Configuration of the CNN encoder(s).
            cnn_projection_cfg: Optional configuration of projection MLP(s) applied after the flattened CNN output(s).
            cnns: CNN modules to use, e.g., for sharing CNNs between actor and critic. If None, new CNNs are created.
            cnn_projectors: Projection modules to use, e.g., for sharing projected CNN branches between actor and critic.
        """
        # Resolve observation groups and dimensions
        self._get_obs_dim(obs, obs_groups, obs_set)

        # Create or validate CNN encoders
        if cnns is not None:
            # Check compatibility if CNNs are provided
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
            # Create CNNs for each 2D observation
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    **cnn_cfg[obs_group],
                )

        if cnn_projectors is not None:
            if set(cnn_projectors.keys()) != set(self.obs_groups_2d):
                raise ValueError("The projected CNN branches must match the configured 2D observation groups.")
        else:
            cnn_projectors = self._build_cnn_projectors(cnns, cnn_projection_cfg)

        # Compute latent dimension of the CNNs
        self.cnn_latent_dim = 0
        for obs_group, cnn in cnns.items():
            if cnn.output_channels is not None:
                raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
            if isinstance(cnn_projectors[obs_group], nn.Identity):
                self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore[arg-type]
            else:
                self.cnn_latent_dim += self._get_projector_output_dim(cnn_projectors[obs_group])

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # Register CNN encoders
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)
        if isinstance(cnn_projectors, nn.ModuleDict):
            self.cnn_projectors = cnn_projectors
        else:
            self.cnn_projectors = nn.ModuleDict(cnn_projectors)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by combining normalized 1D and CNN-encoded 2D observation groups."""
        # Concatenate 1D observation groups and normalize
        latent_1d = super().get_latent(obs)
        # Process 2D observation groups with CNNs
        latent_cnn_list = [
            self.cnn_projectors[obs_group](self.cnns[obs_group](obs[obs_group])) for obs_group in self.obs_groups_2d
        ]
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        # Concatenate 1D and CNN latents
        return torch.cat([latent_1d, latent_cnn], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCNNModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        # Iterate through active observation groups and separate 1D and 2D observations
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. If this is intentional, use the MLP model instead.")

        # Store active 2D observation groups and dimensions directly as attributes
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
        # Return active 1D observation groups and dimension for parent class
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.cnn_latent_dim

    def _build_cnn_projectors(
        self,
        cnns: nn.ModuleDict | dict[str, nn.Module],
        cnn_projection_cfg: dict[str, dict] | dict[str, Any] | None,
    ) -> dict[str, nn.Module]:
        if cnn_projection_cfg is None:
            return {obs_group: nn.Identity() for obs_group in self.obs_groups_2d}

        if not all(isinstance(v, dict) for v in cnn_projection_cfg.values()):
            cnn_projection_cfg = {group: cnn_projection_cfg for group in self.obs_groups_2d}
        if len(cnn_projection_cfg) != len(self.obs_groups_2d):
            raise ValueError("The number of CNN projection configurations must match the number of 2D observation groups.")

        projectors = {}
        for obs_group in self.obs_groups_2d:
            projector_cfg = dict(cnn_projection_cfg[obs_group])
            output_dim = projector_cfg.pop("output_dim")
            hidden_dims = projector_cfg.pop("hidden_dims", [])
            activation = projector_cfg.pop("activation", "elu")
            last_activation = projector_cfg.pop("last_activation", None)
            if projector_cfg:
                raise ValueError(f"Unsupported CNN projection configuration keys: {list(projector_cfg.keys())}")
            cnn_output_dim = int(cnns[obs_group].output_dim)  # type: ignore[arg-type]
            projectors[obs_group] = self._make_projection_mlp(
                input_dim=cnn_output_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                last_activation=last_activation,
            )
        return projectors

    @staticmethod
    def _make_projection_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        last_activation: str | None,
    ) -> nn.Module:
        if len(hidden_dims) > 0:
            return MLP(input_dim, output_dim, hidden_dims, activation, last_activation)

        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        if last_activation is not None:
            layers.append(resolve_nn_activation(last_activation))
        return nn.Sequential(*layers)

    @staticmethod
    def _get_projector_output_dim(projector: nn.Module) -> int:
        for module in reversed(list(projector.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("Unable to determine the output dimension of the CNN projector.")


class _TorchCNNModel(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: CNNModel) -> None:
        """Create a TorchScript-friendly copy of a CNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.cnn_projectors = nn.ModuleList([copy.deepcopy(model.cnn_projectors[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(self.cnn_projectors[i](cnn(obs_2d[i])))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxCNNModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: CNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.cnn_projectors = nn.ModuleList([copy.deepcopy(model.cnn_projectors[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(self.cnn_projectors[i](cnn(obs_2d[i])))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
