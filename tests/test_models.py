import pytest

from llm_scheduler_lab.models import Request, SimulatorConfig


def test_request_validation():
    with pytest.raises(ValueError):
        Request(0, 0.0, 0, 5)
    with pytest.raises(ValueError):
        Request(0, 0.0, 5, 0)


def test_config_validation():
    with pytest.raises(ValueError):
        SimulatorConfig(max_batch_size=0)
