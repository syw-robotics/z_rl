# Third-party source notices

Z-RL's BSD-3-Clause license applies to its original contributions and to upstream
code under compatible terms, subject to the exception below. Preserve existing
file-level copyright and license notices when redistributing source or binaries.
This inventory records known source origins; it is not a complete dependency audit.

## RSL-RL

Z-RL derives from [RSL-RL](https://github.com/leggedrobotics/rsl_rl).
Its ETH Zurich and NVIDIA copyright notices and BSD-3-Clause license are retained
in [LICENSE](LICENSE) and the relevant source files. The Preferred Networks
copyright notice in `z_rl/modules/normalization.py` is also retained.

The SAC algorithm follows the BSD-3-Clause [RSL-RL-SAC implementation](https://github.com/leggedrobotics/rsl_rl_sac/tree/e0d243a).

## Isaac Lab

The Isaac Lab adaptor contains code derived from
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), under BSD-3-Clause.
File-level copyright notices are retained. The full upstream license is included
in [isaaclab-license.txt](licenses/dependencies/isaaclab-license.txt).

## FlashSAC

The FlashSAC variant adapts the PyTorch implementation from [FlashSAC](https://github.com/Holiday-Robot/FlashSAC/tree/87edc90).
The upstream MIT license and Holiday Robotics copyright notice are included in
[flashsac-license.txt](licenses/dependencies/flashsac-license.txt).

