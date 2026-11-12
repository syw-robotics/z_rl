"""Trackio integration tests without remote services."""

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from z_rl.utils.log_writers import TrackioLogWriter
from z_rl.utils.logger import Logger


@pytest.mark.parametrize('use_tracknest', [False, True])
def test_trackio_lifecycle(tmp_path, monkeypatch, use_tracknest):
    client = MagicMock()
    other_client = MagicMock()
    monkeypatch.setitem(sys.modules, 'tracknest' if use_tracknest else 'trackio', client)
    monkeypatch.setitem(sys.modules, 'trackio' if use_tracknest else 'tracknest', other_client)
    client.init.return_value.config = {}
    with patch('torch.utils.tensorboard.SummaryWriter') as summary:
        writer = TrackioLogWriter(str(tmp_path / 'run'), project_name='demo', use_tracknest=use_tracknest)
        writer.store_config({'size': 3}, {'lr': 0.001})
        writer.add_scalar('Loss/value', torch.tensor(0.25), 4)
        writer.add_scalar('Loss/policy', 0.5, 4)
        writer.save_model(str(tmp_path / 'model_4.pt'), 4)
        writer.save_file(str(tmp_path / 'repo.diff'))
        writer.save_video(tmp_path / 'clip.mp4', 4)
        writer.save_video(tmp_path / 'clip.mp4', 5)
        writer.stop()
    client.init.assert_called_once_with(
        project='demo', name='run', config={'log_dir': str(tmp_path / 'run')},
        auto_log_gpu=False, auto_log_cpu=False,
    )
    run = client.init.return_value
    assert run.config == {'train_cfg': {'lr': 0.001}, 'env_cfg': {'size': 3}}
    run.log.assert_any_call({'Loss/value': 0.25}, step=4)
    run.log.assert_any_call({'Loss/policy': 0.5}, step=4)
    run.log_artifact.assert_any_call(str(tmp_path / 'model_4.pt'))
    run.log_artifact.assert_any_call(str(tmp_path / 'repo.diff'))
    client.Video.assert_called_once_with(str(tmp_path / 'clip.mp4'))
    run.log.assert_any_call({'video': client.Video.return_value}, step=4)
    summary.return_value.close.assert_called_once()
    run.finish.assert_called_once()
    other_client.init.assert_not_called()


def test_config_objects(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setitem(sys.modules, 'trackio', client)
    client.init.return_value.config = {}

    @dataclass
    class EnvCfg:
        size: int = 2

    env = MagicMock()
    env.to_dict.return_value = {'size': 3}
    with patch('torch.utils.tensorboard.SummaryWriter'):
        writer = TrackioLogWriter(str(tmp_path), cfg={'trackio_project': 'legacy'})
        writer.store_config(EnvCfg(), {})
        assert writer.run.config['env_cfg'] == {'size': 2}
        writer.store_config(env, {})
        assert writer.run.config['env_cfg'] == {'size': 3}
    assert client.init.call_args.kwargs['project'] == 'legacy'


def test_missing_dependency_and_project(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'trackio', None)
    with pytest.raises(ModuleNotFoundError, match='pip install trackio'):
        TrackioLogWriter(str(tmp_path), project_name='demo')
    monkeypatch.setitem(sys.modules, 'trackio', MagicMock())
    with patch('torch.utils.tensorboard.SummaryWriter'), pytest.raises(KeyError, match='project_name'):
        TrackioLogWriter(str(tmp_path))


def test_missing_tracknest_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'tracknest', None)
    monkeypatch.setitem(sys.modules, 'trackio', MagicMock())
    with pytest.raises(ModuleNotFoundError, match='tracknest-client'):
        TrackioLogWriter(str(tmp_path), project_name='demo', use_tracknest=True)


@pytest.mark.parametrize('config', [
    'trackio',
    {'class_name': 'TrackioLogWriter', 'project_name': 'demo'},
    {'class_name': 'TrackioLogWriter', 'project_name': 'demo', 'use_tracknest': True},
])
def test_logger_config(tmp_path, config):
    logger = Logger(str(tmp_path), {'logger': config, 'trackio_project': 'demo'}, {}, 2, False, 1, 0, 'cpu')
    with patch('z_rl.utils.TrackioLogWriter') as writer, patch.object(logger, '_store_code_state', return_value=[]):
        if isinstance(config, str):
            with pytest.warns(DeprecationWarning):
                logger.init_logging_writer()
        else:
            logger.init_logging_writer()
        kwargs = {'use_tracknest': True} if isinstance(config, dict) and config.get('use_tracknest') else {}
        writer.assert_called_once_with(log_dir=str(tmp_path), project_name='demo', **kwargs)
        assert logger.logger_type == 'TrackioLogWriter'


def test_iteration_step_axis(tmp_path):
    logger = Logger(str(tmp_path), {'num_steps_per_env': 4}, {}, 2, False, 1, 0, 'cpu')
    logger.logger_type = 'TrackioLogWriter'
    logger.writer = MagicMock()
    logger.rewbuffer.append(1.5)
    logger.lenbuffer.append(8)
    logger.log(it=1, start_it=0, total_it=2, collect_time=0.2, learn_time=0.3,
               loss_dict={'value': 0.1}, learning_rate=1e-3, print_minimal=True)
    tags = [call.args[0] for call in logger.writer.add_scalar.call_args_list]
    assert 'Train/mean_reward' in tags
    assert 'Train/mean_episode_length' in tags
    assert 'Train/mean_reward/time' not in tags
    assert 'Train/mean_episode_length/time' not in tags
