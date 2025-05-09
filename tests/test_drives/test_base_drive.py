import torch
import pytest

from src.drives.base_drive import BaseDrive

def almost_equal(a, b, tol=1e-5):
    return torch.isclose(a, b, atol=tol)

@pytest.mark.parametrize("optimal,current,m,n,expected", [
    ([1.0, 2.0], [0.0, 0.0], 1, 1, torch.tensor(3.0)),                            # L1 norm
    ([1.0, 2.0], [0.0, 0.0], 2, 2, torch.sqrt(torch.tensor(5.0))),               # L2 norm
    ([1.0, 2.0], [0.0, 0.0], 3, 4, torch.tensor((1**4 + 2**4) ** (1/3))),         # generalized
])
def test_compute_drive(optimal, current, m, n, expected):
    drive = BaseDrive(optimal_internal_states_config=optimal, m=m, n=n)
    result = drive.compute_drive(current)
    assert almost_equal(result, expected), f"Expected {expected}, got {result}"

@pytest.mark.parametrize("optimal,current,new_states,m,n", [
    ([1.0, 1.0], [0.0, 0.0], [0.5, 0.5], 1, 1),
    ([1.0, 1.0], [0.0, 0.0], [0.5, 1.5], 2, 2),
    ([1.0, 1.0], [0.0, 0.0], [1.5, 0.5], 3, 4),
])
def test_compute_reward(optimal, current, new_states, m, n):
    drive = BaseDrive(optimal_internal_states_config=optimal, m=m, n=n)
    
    # Calculate initial drive
    d_t = drive.compute_drive(current)
    drive.update_drive(d_t)
    
    # Calculate new drive
    d_tp1 = drive.compute_drive(new_states)
    
    # Expected reward is the reduction in drive
    expected = d_t - d_tp1
    
    # Get actual reward
    result = drive.compute_reward(d_tp1)
    
    assert almost_equal(result, expected), f"Expected {expected}, got {result}"

@pytest.mark.parametrize("optimal,current,expected_result", [
    ([0.5, 0.5], [0.5, 0.5], True),                  # Exact match
    ([0.5, 0.5], [0.5, 0.501], True),                # Very close (within tolerance)
    ([0.5, 0.5], [0.5, 0.6], False),                 # Too far
])
def test_has_reached_optimal(optimal, current, expected_result):
    drive = BaseDrive(optimal_internal_states_config=optimal, m=1, n=1)
    result = drive.has_reached_optimal(current)
    assert result == expected_result, f"Expected {expected_result}, got {result}"
