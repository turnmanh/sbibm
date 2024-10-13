""" 
This file provides single functions to evaluate the single injury risks of the
US NCAP rating for a full frontal crash test. 
"""

import torch


def p_hic(hic: torch.tensor) -> torch.tensor:
    """
    Evaluates the cdf of a random variable from a parameterized normal distribution.

    Args:
        hic: HIC value obtained from hardware tests or simulations.
    Returns:
        torch.tensor: Probability of head injury.
    """
    x = ( torch.log(hic) - 7.45231 ) / 0.73998
    return torch.distributions.Normal(0.0, 1.0).cdf(x)

def p_chest(chest: torch.tensor) -> torch.tensor:
    """
    Provides the probability of chest injury based on the chest deflection.

    Args:
        chest: Chest deflection in mm.
    Returns:
        torch.tensor: Probability of chest injury.
    """
    return 1 / (1 + torch.exp(10.5456 - 1.568 * chest ** 0.4612))

def p_femur(femur: torch.tensor) -> torch.tensor:
    """
    Provides the probability of femur injury based on the femur load.
    
    Args:
        femur: Femur load in kN.
    Returns:
        torch.tensor: Probability of femur injury.
    """
    return 1 / (1 + torch.exp(5.795 - 0.5196 * femur))

def p_nij(nij: torch.tensor) -> torch.tensor:
    """
    Provides the probability of neck injury based on the neck injury criterion.

    Args:
        nij: NIJ value obtained during the crash test.

    Returns:
        torch.tensor: Probability of neck injury.
    """
    return 1 / (1 + torch.exp(3.2269 - 1.9688 * nij))

def p_tension(n_tension: torch.tensor) -> torch.tensor:
    """
    Provides the probability of neck injury based on the neck tension.

    Args:
        n_tension: Tension value obtained during the crash test.

    Returns:
        torch.tensor: Probability of neck injury.
    """
    return 1 / (1 + torch.exp(10.9745 - 2.375 * n_tension))

def p_compression(n_compression: torch.tensor) -> torch.tensor:
    """
    Provides the probability of neck injury based on the neck compression.

    Args:
        n_compression: Compression value obtained during the crash test.

    Returns:
        torch.tensor: Probability of neck injury.
    """
    return 1 / (1 + torch.exp(10.9745 - 2.375 * n_compression))