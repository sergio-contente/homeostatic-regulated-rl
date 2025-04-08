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
    
    assert drive.optimal_internal_states is not None
    assert drive.m is not None
    
    # Verificações específicas para cada tipo
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
    
    assert drive.optimal_internal_states == [1.5, 2.5] or torch.all(torch.eq(drive.optimal_internal_states, torch.tensor([1.5, 2.5])))
    
    if drive_type == 'base_drive':
        assert drive.n == 3
        assert drive.m == 4
    
    elif drive_type == 'interoceptive_drive':
        assert drive.n == 3  # base_drive
        assert drive.m == 4  # base_drive
        assert drive.eta == 0.7
    
    elif drive_type == 'elliptic_drive':
        assert torch.all(torch.eq(drive.n_vector, torch.tensor([3, 4]))) if isinstance(drive.n_vector, torch.Tensor) else drive.n_vector == [3, 4]
        assert drive.m == 5  # Overwritten
