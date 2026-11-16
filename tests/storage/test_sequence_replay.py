import torch
from z_rl.storage.sequence_replay import SequenceReplayBuffer


def test_wrapped_windows_include_terminal_step_but_never_reset_target():
    replay = SequenceReplayBuffer(4, 2, "cpu")
    for t in range(7):
        replay.add(
            dict(
                state=torch.full((2, 1), float(t)),
                next_state=torch.full((2, 1), float(t + 1)),
                segment_end=torch.tensor([t == 4, False]),
                frame=torch.full((2,), t),
            )
        )
    batch = replay.sample_windows(100, 3)
    torch.testing.assert_close(batch["state"][:, 1:, 0], batch["state"][:, :-1, 0] + 1)
    torch.testing.assert_close(batch["next_state"], batch["state"] + 1)
    assert torch.all(batch["mask"][:, 0] == 1)
    ended = batch["segment_end"][:, :-1].cummax(1).values
    assert torch.equal(batch["mask"][:, 1:] == 0, ended)
    assert replay.sample_starts(10)["state"].shape == (10, 1)
