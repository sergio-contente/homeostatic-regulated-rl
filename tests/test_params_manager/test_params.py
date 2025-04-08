# tests/test_params_manager/test_params.py
import torch
import pytest

from utils.get_params import ParameterHandler


@pytest.mark.parametrize("drive_type", [
    'base_drive',
    'interoceptive_drive',
    'elliptic_drive'
])
def test_get_drive(drive_type):
    config_path = "config/config.yaml"
    param_handler = ParameterHandler(config_path)
    drive = param_handler.create_drive(drive_type)
    
    assert hasattr(drive, '_optimal_internal_states')
    assert drive._optimal_internal_states is not None
    assert drive.m is not None
    
    if drive_type == 'base_drive':
        assert drive.n is not None
        assert not hasattr(drive, 'eta')
        assert not hasattr(drive, 'n_vector')
    
    elif drive_type == 'interoceptive_drive':
        assert drive.n is not None
        assert drive.eta is not None
        assert not hasattr(drive, 'n_vector')
    
    elif drive_type == 'elliptic_drive':
        assert hasattr(drive, 'n')
        assert drive.n_vector is not None
        assert not hasattr(drive, 'eta')

@pytest.mark.parametrize("drive_type", [
    'base_drive',
    'interoceptive_drive',
    'elliptic_drive'
])
def test_get_drive_values(drive_type):
    config_path = "config/test_config.yaml"
    param_handler = ParameterHandler(config_path)
    drive = param_handler.create_drive(drive_type)
    
    if isinstance(drive._optimal_internal_states, dict):
        values = list(drive._optimal_internal_states.values())
    else:
        values = drive._optimal_internal_states
    
    if isinstance(values, torch.Tensor):
        values_list = values.tolist()
    else:
        values_list = values
        
    expected = [1.5, 2.5, 1.0, 3.0]
    assert len(values_list) == len(expected), f"Expected size: {len(expected)}, obtained: {len(values_list)}"
    
    for i, (val, exp) in enumerate(zip(values_list, expected)):
        assert abs(val - exp) < 1e-5, f"At index {i}, expected {exp}, obtained {val}"
    
    if drive_type == 'base_drive':
        assert drive.n == 3
        assert drive.m == 4
    
    elif drive_type == 'interoceptive_drive':
        assert drive.n == 3  # base_drive
        assert drive.m == 4  # base_drive
        assert drive.eta == 0.7
    
    elif drive_type == 'elliptic_drive':
        assert torch.all(torch.eq(drive.n_vector, torch.tensor([3, 4, 2, 5]))) if isinstance(drive.n_vector, torch.Tensor) else drive.n_vector == [3, 4, 2, 5]
        assert drive.m == 5  # Overwritten
