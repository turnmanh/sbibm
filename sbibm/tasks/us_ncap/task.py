from pathlib import Path

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class US_NCAP(Task):
    def __init__(
        self,
        dim_data: int = 1,
        dim_parameters: int = 6,
        name: str = Path(__file__).parent.name,
        num_observations: int = 10,
        num_posterior_samples: int = 10_000,
        num_reference_posterior_samples: int = 10_000,
        num_simulations: list[int] = [100, 1_000, 10_000, 100_000, 1_000_000],
        path: str = Path(__file__).parent.absolute(),
        prior_lower_bound: list[float] = [200, 5, 2, 0.0, 1.5, 1.0],
        prior_upper_bound: list[float] = [800, 42.5, 8.0, 0.75, 4.0, 4.0],
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
        self.noise_level = noise_level
