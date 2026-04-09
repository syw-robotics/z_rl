# IsaacLab 的 Z-RL Adaptor

这个包用于把 IsaacLab 的 `ManagerBasedRLEnv` 适配成 Z-RL 期望的接口。

当前主要提供：

- `ZRlVecEnvWrapper`：把 IsaacLab 环境包装成 `z_rl.env.VecEnv`
- `rl_cfg.py` 中的一组训练配置类，例如 `ZRlBaseRunnerCfg`、`ZRlOnPolicyRunnerCfg`
- `distillation_cfg.py` 中的蒸馏配置
- `ZRlSymmetryCfg`：对称增强相关配置
- `export_policy_as_jit()` / `export_policy_as_onnx()`：策略导出工具

## 快速使用

```python
from isaaclab_rl.z_rl import ZRlVecEnvWrapper

env = ZRlVecEnvWrapper(
    env,
    clip_actions=1.0,
    obs_group_concat_mode="term_major",  # 或 "history_major"
)
```

注意：

- `ZRlVecEnvWrapper` 必须是包装链上的最后一层。
- 这个 wrapper 在初始化时会主动调用一次 `env.reset()`，因为 Z-RL runner 本身不会先调 reset。

## `ZRlVecEnvWrapper` 做了什么

它主要解决两类兼容问题：

1. step / action 接口适配
2. observation 布局适配

运行时它会：

- 暴露 `num_envs`、`device`、`max_episode_length`、`num_actions`
- 在 `step()` 前按需裁剪 action
- 把 IsaacLab 的 `(terminated, truncated)` 转成 Z-RL 风格的 `dones`
- 对 infinite-horizon 任务补充 `extras["time_outs"] = truncated`
- 把 observation 包成 `TensorDict`
- 缓存 observation 元信息，例如 `obs_format`、`obs_group_layout_mode_map`、time-slice selector

## Observation group 与 `concatenate_terms`

在 IsaacLab 里，每个 observation group 都由 `ObservationGroupCfg.concatenate_terms` 控制输出形式。

- `concatenate_terms=True`：IsaacLab 直接返回一个拼好的 tensor
- `concatenate_terms=False`：IsaacLab 返回 term 字典，例如：

```python
{
    "policy": {
        "base_lin_vel": ...,
        "joint_pos": ...,
    }
}
```

`ZRlVecEnvWrapper` 只会重新组织那些以字典形式返回的 group。这里的关系非常关键：

- 如果 `concatenate_terms=True`，这个 group 到 wrapper 时已经是 tensor 了，`obs_group_concat_mode` 对它不起作用
- 如果 `concatenate_terms=False`，wrapper 才会把这个 dict group 再转换成单个 tensor

所以，`obs_group_concat_mode` 只影响 IsaacLab 里设置了 `concatenate_terms=False` 的 group。

## `term_major` 和 `history_major`

这是这个 wrapper 最需要说明白的部分。

假设某个 observation group 里有两个 term：

- `a`：单帧维度是 `Da`
- `b`：单帧维度是 `Db`
- 两者 history length 都是 `H`

### `term_major`

`term_major` 保持 IsaacLab 默认的“按 term 组织 history”的布局。

概念上，扁平化顺序是：

```text
[a_t-H+1, a_t-H+2, ..., a_t, b_t-H+1, b_t-H+2, ..., b_t]
```

如果 IsaacLab 已经把每个 term 压成 `(N, H * D)`，那么 wrapper 拼接后的形状就是：

```text
(N, H*Da + H*Db)
```

这是默认模式，也是所有不满足 `history_major` 条件时的回退模式。

### `history_major`

`history_major` 会先把每个 term 还原成 `(N, H, D)`，再在同一个时间步内部拼 feature。

概念上，扁平化顺序变成：

```text
[a_t-H+1, b_t-H+1, a_t-H+2, b_t-H+2, ..., a_t, b_t]
```

最终形状是：

```text
(N, H * (Da + Db))
```

这种布局更适合下游按“帧”切 observation，例如取最后一帧、去掉最后一帧、做时间窗口偏移等。

## `term_major` / `history_major` 与 `self.concatenate_terms` 的准确关系

如果你想让某个 group 真正输出成 `history_major`，下面条件必须同时满足：

1. IsaacLab 侧这个 group 必须设置 `self.concatenate_terms = False`
2. 创建 wrapper 时使用 `obs_group_concat_mode="history_major"`
3. 这个 group 必须满足 time-slice 兼容条件：
   - 每个 term 的 history length 相同且都大于 0
   - 每个 term 的单帧都是向量
   - 这个 group 必须是 dict group，不能已经在 IsaacLab 里拼成单 tensor

只要有任意一条不满足，wrapper 对这个 group 的实际布局就会是 `term_major`。

换句话说：

- `self.concatenate_terms=True` 加上 `obs_group_concat_mode="history_major"` 不会报错，但这个 group 不会变成 `history_major`
- `self.concatenate_terms=False` 只是必要条件，不是充分条件；term 的 shape 和 history 配置也必须兼容

## 几个实际例子

### 例 1：IsaacLab 已经拼好了 term

IsaacLab 配置：

```python
self.concatenate_terms = True
```

Wrapper 配置：

```python
ZRlVecEnvWrapper(env, obs_group_concat_mode="history_major")
```

结果：

- 这个 group 到 wrapper 时已经是 tensor
- wrapper 不会重排它
- 实际布局仍然是 IsaacLab 原本的拼接方式

### 例 2：dict group，使用默认 `term_major`

IsaacLab 配置：

```python
self.concatenate_terms = False
```

Wrapper 配置：

```python
ZRlVecEnvWrapper(env, obs_group_concat_mode="term_major")
```

结果：

- wrapper 按 term 顺序拼接
- 每个 term 的 history 仍然先在 term 内部连续存放

### 例 3：dict group，使用 `history_major`

IsaacLab 配置：

```python
self.concatenate_terms = False
```

Wrapper 配置：

```python
ZRlVecEnvWrapper(env, obs_group_concat_mode="history_major")
```

如果 group 兼容：

- 每个 term 会被视为 `(N, H, D)`
- 同一时间步的各个 term 会先拼在一起
- 最后再展平成 `(N, H * sum(D_i))`

如果 group 不兼容：

- wrapper 会对这个 group 自动回退到 `term_major`

## `get_obs_time_slice()` 的作用

对于兼容的 group，wrapper 会预先缓存几种常用的时间切片：

- `"last"`
- `"exclude_last"`
- `"exclude_first"`

调用方式：

```python
last_obs = env.get_obs_time_slice(obs["policy"], "policy", "last")
```

只有满足以下条件的 group 才支持：

- 所有 term 的 history length 相同且非 0
- 每个 term 的单帧都是向量

对 `history_major` group 来说，这些切片通常是连续区间。
对兼容但仍是 `term_major` 的 group 来说，切片可能是 index tensor，因为“最后一帧”的各段分散在不同 term block 里，不一定连续。

## Wrapper 暴露的缓存元信息

比较有用的属性有：

- `obs_format`
  - 结构：`{group_name: {term_name: (history_length, *single_frame_shape)}}`
- `obs_group_layout_mode_map`
  - 每个 group 最终实际采用的布局：`"term_major"` 或 `"history_major"`
- `obs_group_time_slice_map`
  - `get_obs_time_slice()` 使用的缓存 selector

这些信息对下游模型理解“一个扁平 observation 向量究竟是怎么拼出来的”很有帮助。

## 建议的配置方式

以下情况建议用 `term_major`：

- 你想尽量保持 IsaacLab 默认布局
- 某些 group 已经在 IsaacLab 侧完成拼接
- 不同 term 的 history length 不一致

以下情况建议用 `history_major`：

- 下游算法按帧处理时间窗口
- 你希望便宜地取“最后一帧”或“去掉首尾帧”的连续块
- 目标 group 在 IsaacLab 里配置了 `concatenate_terms=False`
- 这些 group 里的 term 都是向量单帧，并且 history length 相同且非 0

## 文件说明

- `vecenv_wrapper.py`：核心的 IsaacLab -> Z-RL 环境包装器
- `rl_cfg.py`：训练 runner、算法、模型相关配置
- `distillation_cfg.py`：蒸馏相关配置
- `symmetry_cfg.py`：对称增强配置
- `exporter.py`：TorchScript / ONNX 导出工具
