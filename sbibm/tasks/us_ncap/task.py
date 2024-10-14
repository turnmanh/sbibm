import torch

from pathlib import Path
from torch.distributions import Uniform
from typing import List

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.tasks.us_ncap.task_utils import (
    p_chest,
    p_femur,
    p_nij,
    p_tension,
    p_compression,
    p_hic,
)


class US_NCAP(Task):
    def __init__(
        self,
        dim_data: int = 1,
        dim_parameters: int = 6,
        name: str = Path(__file__).parent.name,
        num_observations: int = 10,
        num_posterior_samples: int = 10_000,
        num_reference_posterior_samples: int = 10_000,
        num_simulations: List[int] = [100, 1_000, 10_000, 100_000, 1_000_000],
        path: str = Path(__file__).parent.absolute(),
        prior_lower_bound: List[float] = [200, 5, 2, 0.0, 1.5, 1.0],
        prior_upper_bound: List[float] = [800, 42.5, 8.0, 0.75, 4.0, 4.0],
        base_risk: float = 0.15,
        noise_level: float = 0.1,
    ) -> None:
        """US NCAP Simulator.

        The US New Car Assessment Program (NCAP) is a rating of vehicle safety
        performance. It includes several crash scenarios and utilizes the
        measurements on relevant body parts to provide an overall safety
        assessment of a car.

        This simulator provides the assessment from a from crash with full
        overlap at 56 km/h.

        Args:
            dim_data: Dimensionality of observations.
            dim_parameters: Dimensionality of simulator's parameters.
            name: Name of the task.
            num_observations: Number of different observations provided.
            num_posterior_samples: Number of posterior samples to generate.
            num_reference_posterior_samples: Reference samples to generate.
            num_simulations: Number of simulations to run for this task.
            path: Path to the task's folder.
            prior_lower_bound: Lower bound of the uniform prior.
            prior_upper_bound: Upper bound of the uniform prior.
            base_risk: Base risk of injury.
            noise_level: Noise level to add to the observations.
        """
        self.dim_data = dim_data
        self.dim_parameters = dim_parameters
        self.name = name
        self.num_observations = num_observations
        self.num_posterior_samples = num_posterior_samples
        self.num_reference_posterior_samples = num_reference_posterior_samples
        self.num_simulations = num_simulations
        self.path = path
        self.prior_lower_bound = prior_lower_bound
        self.prior_upper_bound = prior_upper_bound
        self.base_risk = base_risk
        self.noise_level = noise_level

        self.prior_params = {
            "low": torch.tensor(self.prior_lower_bound),
            "high": torch.tensor(self.prior_upper_bound),
        }

        self.prior_dist = Uniform(
            low=torch.tensor(self.prior_lower_bound),
            high=torch.tensor(self.prior_upper_bound),
        )

    def get_prior(self) -> callable:
        """Get function returning samples from the prior distribution."""

        def prior(num_samples: int = 1):
            return self.prior_dist.sample((num_samples,))

        return prior
    
    def get_labels_data(self) -> List[str]:
        """Get labels for the data dimensions."""
        return ["Relative Risk"]
    
    def get_labels_parameters(self) -> List[str]:
        """Get labels for the parameter dimensions."""
        return [
            "HIC",
            "Chest Deflection",
            "Femur Load",
            "NIJ",
            "Neck Compression",
            "Neck Tension"]
    
    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observation for a given index."""
        raise NotImplementedError("This task does not provide observations yet.")

    def get_simulator(self, max_calls: int = None) -> Simulator:
        """Get function returning samples from the simulator given parameters.

        Args:
            max_calls: Maximum number of calls to the simulator. Additional
            calls result in the a SimulationBudgetExceeded exception.

        Returns:
            Simulator callable.
        """

        def simulator(parameters: torch.Tensor) -> torch.Tensor:
            """Simulates the US NCAP rating for a full frontal crash test.

            Args:
                parameters: Parameterization of restrain system.

            Returns:
                torch.tensor: Relative risk of injury.
            """

            # evaluate single probabilities according to provided parameters
            p_hic = p_hic(parameters[:, 0].reshape(-1, 1))
            p_chest = p_chest(parameters[:, 1].reshape(-1, 1))
            p_femur = p_femur(parameters[:, 2].reshape(-1, 1))
            p_nij = p_nij(parameters[:, 3].reshape(-1, 1))
            p_compression = p_compression(parameters[:, 4].reshape(-1, 1))
            p_tension = p_tension(parameters[:, 5].reshape(-1, 1))

            # transform the neck injury probability
            p_neck, _ = torch.max(
                torch.stack([p_nij, p_compression, p_tension], dim=0),
                dim=0,
            )

            joint_prob_inj = 1 - (1 - p_hic) * (1 - p_chest) * (1 - p_femur) * (
                1 - p_neck
            )

            risk_values = 1 - joint_prob_inj / self.base_risk

            # Generating and adding Gaussian white noise to relative risk.
            noise = torch.randn_like(risk_values) * self.noise_level
            risk_values += noise

            return risk_values.reshape(-1, 1)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)
