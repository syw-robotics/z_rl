"""Differentiable local-increment rigid robot dynamics; quaternions use xyzw."""

import torch
from torch import nn
from z_rl.modules.mlp import MLP
from z_rl.modules.normalization import EmpiricalNormalization


def quat_mul(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_inv(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quat_apply(q, v):
    qv, v = torch.broadcast_tensors(q[..., :3], v)
    uv = torch.linalg.cross(qv, v)
    return v + 2 * (q[..., 3:] * uv + torch.linalg.cross(qv, uv))


def rotvec_to_quat(x):
    angle = x.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return torch.cat((0.5 * torch.sinc(angle / (2 * torch.pi)) * x, torch.cos(angle / 2)), -1)


def rotation_error(a, b):
    """Squared geodesic angle, with finite derivatives at coincident rotations."""
    delta = quat_mul(quat_inv(a), b)
    norm = delta[..., :3].square().sum(-1).clamp_min(1e-12).sqrt()
    return (2 * torch.atan2(norm, delta[..., 3].abs().clamp_min(1e-8))).square()


class LocalResidualDynamics(nn.Module):
    """Learn one control-period transition on clean physical state and executed action.

    Flat state: position(3), quaternion(4), world link twist(6), q(J), qd(J).
    Input removes world xy and yaw. Geometry and integration always stay fp32.
    """

    def __init__(
        self,
        num_joints,
        action_dim,
        context_dim,
        hidden_dims=(2048, 2048, 2048),
        activation="swish",
        layer_norm="pre_activation",
        input_normalization=True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.input_normalization = input_normalization
        width = 10 + 2 * num_joints + action_dim + context_dim
        self.input_norm = EmpiricalNormalization(width)
        self.net = MLP(width, 12 + 2 * num_joints, hidden_dims, activation, layer_norm=layer_norm)
        output = [layer for layer in self.net.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        scales = [0.25] * 3 + [0.5] * 3 + [1.0] * 3 + [3.14] * 3 + [0.5] * num_joints + [5.0] * num_joints
        self.register_buffer("error_scale", torch.tensor(scales))

    def capture_config(self):
        return (self.num_joints, self.input_normalization)

    def features(self, state):
        inv = quat_inv(state[..., 3:7])
        gravity = torch.zeros_like(state[..., :3])
        gravity[..., 2] = -1
        return torch.cat(
            (
                state[..., 2:3],
                quat_apply(inv, gravity),
                quat_apply(inv, state[..., 7:10]),
                quat_apply(inv, state[..., 10:13]),
                state[..., 13:],
            ),
            -1,
        )

    replay_fields = ("state", "next_state", "action", "context")

    def inputs(self, state, action, context):
        return torch.cat((self.features(state), action, context), -1)

    @torch.no_grad()
    def update_normalization(self, state, action, context):
        if self.input_normalization:
            self.input_norm.update(self.inputs(state, action, context))

    def step(self, state, action, context):
        x = self.inputs(state, action, context)
        if self.input_normalization:
            x = self.input_norm(x)
        delta = self.net(x).float()
        dp, dr, dv, dw, dq = delta.split((3, 3, 3, 3, 2 * self.num_joints), -1)
        rotation = state[..., 3:7]
        next_rotation = quat_mul(rotation, rotvec_to_quat(dr))
        next_rotation = nn.functional.normalize(next_rotation, dim=-1)
        return torch.cat(
            (
                state[..., :3] + quat_apply(rotation, dp),
                next_rotation,
                state[..., 7:10] + quat_apply(rotation, dv),
                state[..., 10:13] + quat_apply(rotation, dw),
                state[..., 13:] + dq,
            ),
            -1,
        )

    def state_error(self, predicted, target):
        inv = quat_inv(target[..., 3:7])
        rotation = quat_mul(inv, predicted[..., 3:7])
        # Chordal tangent error is sign-invariant after squaring and smooth near zero.
        return (
            torch.cat(
                (
                    quat_apply(inv, predicted[..., :3] - target[..., :3]),
                    2 * rotation[..., :3],
                    quat_apply(inv, predicted[..., 7:10] - target[..., 7:10]),
                    quat_apply(inv, predicted[..., 10:13] - target[..., 10:13]),
                    predicted[..., 13:] - target[..., 13:],
                ),
                -1,
            )
            / self.error_scale
        )

    def loss(self, window):
        state = window["state"][:, 0]
        errors = []
        for t in range(window["action"].shape[1]):
            state = self.step(state, window["action"][:, t], window["context"][:, t])
            errors.append(self.state_error(state, window["next_state"][:, t]).square().mean(-1))
        errors = torch.stack(errors, 1)
        return (errors * window["mask"]).sum() / window["mask"].sum().clamp_min(1)
