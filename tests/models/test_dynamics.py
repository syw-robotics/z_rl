import torch
from z_rl.models.dynamics import LocalResidualDynamics, quat_apply, quat_mul, rotvec_to_quat


def states(batch=8, joints=2):
    s = torch.zeros(batch, 13 + 2 * joints)
    s[:, 2] = 0.8
    s[:, 6] = 1
    return s


def test_features_ignore_world_xy_and_yaw():
    wm = LocalResidualDynamics(2, 2, 1, hidden_dims=[16], input_normalization=False)
    s = states()
    s[:, 7:13] = torch.randn(8, 6)
    yaw = rotvec_to_quat(torch.tensor([[0.0, 0.0, 1.1]])).expand(8, -1)
    other = s.clone()
    other[:, :3] = quat_apply(yaw, s[:, :3]) + torch.tensor([5.0, -3.0, 0.0])
    other[:, 3:7] = quat_mul(yaw, s[:, 3:7])
    other[:, 7:10] = quat_apply(yaw, s[:, 7:10])
    other[:, 10:13] = quat_apply(yaw, s[:, 10:13])
    torch.testing.assert_close(wm.features(s), wm.features(other))


def test_zero_rotation_increment_has_finite_gradient():
    x = torch.zeros(4, 3, requires_grad=True)
    q = rotvec_to_quat(x)
    q[:, :3].sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.5))
    wm = LocalResidualDynamics(2, 2, 1, hidden_dims=[16])
    out = wm.step(states(), torch.zeros(8, 2), torch.zeros(8, 1))
    torch.testing.assert_close(out, states())
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)


def test_autoregressive_fit_reduces_held_out_error():
    torch.manual_seed(4)
    wm = LocalResidualDynamics(2, 2, 1, hidden_dims=[32, 32], input_normalization=False)
    s = states(64)
    a = torch.randn(64, 3, 2) * 0.2
    targets = []
    current = s
    for t in range(3):
        current = current.clone()
        current[:, 13:15] += a[:, t] * 0.1
        targets.append(current)
    window = dict(
        state=s[:, None].expand(-1, 3, -1),
        next_state=torch.stack(targets, 1),
        action=a,
        context=torch.zeros(64, 3, 1),
        mask=torch.ones(64, 3),
    )
    training = {key: value[:48] for key, value in window.items()}
    held_out = {key: value[48:] for key, value in window.items()}
    initial = wm.loss(held_out).item()
    optimizer = torch.optim.Adam(wm.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        wm.loss(training).backward()
        optimizer.step()
    assert wm.loss(held_out).item() < initial * 0.1
