"""Comet backend integration tests without network access."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from z_rl.utils.log_writers import CometLogWriter
from z_rl.utils.logger import Logger


def test_comet_lifecycle(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr('z_rl.utils.log_writers.comet_ml', client)
    env = MagicMock()
    env.to_dict.return_value = {'size': 3}
    with patch('torch.utils.tensorboard.SummaryWriter') as summary:
        writer = CometLogWriter(str(tmp_path / 'run'), project_name='demo')
        writer.store_config(env, {'lr': 0.001})
        writer.add_scalar('Loss/value', torch.tensor(0.25), 4)
        writer.save_model(str(tmp_path / 'model_4.pt'), 4)
        writer.save_file(str(tmp_path / 'repo.diff'))
        video = tmp_path / 'clip.mp4'
        writer.save_video(video, 4)
        writer.save_video(video, 5)
        writer.stop()
    run = client.Experiment.return_value
    assert client.Experiment.call_args.kwargs['project_name'] == 'demo'
    assert client.Experiment.call_args.kwargs['auto_metric_logging'] is False
    run.set_name.assert_called_once_with('run')
    run.log_parameters.assert_any_call({'train_cfg': {'lr': 0.001}})
    run.log_parameters.assert_any_call({'env_cfg': {'size': 3}})
    run.log_metric.assert_called_once_with('Loss/value', 0.25, step=4)
    run.log_asset.assert_any_call(str(tmp_path / 'model_4.pt'))
    run.log_asset.assert_any_call(str(tmp_path / 'repo.diff'))
    run.log_video.assert_called_once_with(str(video), step=4)
    summary.return_value.close.assert_called_once()
    run.end.assert_called_once()


def test_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr('z_rl.utils.log_writers.comet_ml', None)
    with pytest.raises(ModuleNotFoundError, match='pip install comet-ml'):
        CometLogWriter(str(tmp_path), project_name='demo')


@pytest.mark.parametrize('config', ['comet', {'class_name': 'CometLogWriter', 'project_name': 'demo'}])
def test_logger_config(tmp_path, config):
    logger = Logger(str(tmp_path), {'logger': config, 'comet_project': 'demo'}, {}, 2, False, 1, 0, 'cpu')
    with patch('z_rl.utils.CometLogWriter') as writer, patch.object(logger, '_store_code_state', return_value=[]):
        if isinstance(config, str):
            with pytest.warns(DeprecationWarning):
                logger.init_logging_writer()
        else:
            logger.init_logging_writer()
        writer.assert_called_once_with(log_dir=str(tmp_path), project_name='demo')
        assert logger.logger_type == 'CometLogWriter'
