import torch
import pytest

from drives.base_drive import BaseDrive

def almost_equal(a, b, tol=1e-5):
    return torch.isclose(a, b, atol=tol)

@pytest.mark.parametrize("optimal,current,m,n,expected", [
    ([1.0, 2.0], [0.0, 0.0], 1, 1, torch.tensor(3.0)),                            # L1 norm
    ([1.0, 2.0], [0.0, 0.0], 2, 2, torch.sqrt(torch.tensor(5.0))),               # L2 norm
    ([1.0, 2.0], [0.0, 0.0], 3, 4, torch.tensor((1**4 + 2**4) ** (1/3))),         # generalized
])
def test_compute_drive(optimal, current, m, n, expected):
    drive = BaseDrive(optimal_internal_states=optimal, m=m, n=n)
    result = drive.compute_drive(current)
    assert almost_equal(result, expected), f"Expected {expected}, got {result}"

@pytest.mark.parametrize("optimal,current,outcome,m,n", [
    ([1.0, 1.0], [0.0, 0.0], [0.5, 0.5], 1, 1),
    ([1.0, 1.0], [0.0, 0.0], [0.5, 1.5], 2, 2),
    ([1.0, 1.0], [0.0, 0.0], [1.5, 0.5], 3, 4),
])
def test_compute_reward(optimal, current, outcome, m, n):
    drive = BaseDrive(optimal_internal_states=optimal, m=m, n=n)
    d_t = drive.compute_drive(current)
    d_tp1 = drive.compute_drive(torch.tensor(current) + torch.tensor(outcome))
    expected = d_t - d_tp1
    result = drive.compute_reward(current, outcome)
    assert almost_equal(result, expected), f"Expected {expected}, got {result}"
