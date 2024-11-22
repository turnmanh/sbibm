import torch

from pathlib import Path
from torch.distributions import Uniform
from typing import List

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.tasks.us_ncap.task_utils import FullFrontalCrash, SideImpact, RollOver


# Set the default lower and upper bounds for the measurements.
LOWER_BOUND_FULL_FRONTAL = [200, 5, 2, 0.0, 1.5, 1.5, 200, 5, 2, 0.0, 1.5, 1.5]
UPPER_BOUND_FULL_FRONTAL = [
    1_000,
    45,
    8.0,
    1.2,
    4.0,
    4.0,
    1_000,
    45,
    6.0,
    1.2,
    2.5,
    2.5,
]

LOWER_BOUND_SIDE_IMPACT = [200, 1_000, 200, 5, 1_000, 1_000, 200, 1_000]
UPPER_BOUND_SIDE_IMPACT = [
    1_000,
    5_500,
    1_000,
    45,
    5_500,
    5_500,
    1_000,
    5_500,
]

LOWER_BOUND_ROLL_OVER = [0.01]
UPPER_BOUND_ROLL_OVER = [0.25]


class US_NCAP(Task):
    def __init__(
        self,
        dim_data: int = 1,
        dim_parameters: int = 21,
        name: str = Path(__file__).parent.name,
        num_observations: int = 10,
        num_posterior_samples: int = 10_000,
        num_reference_posterior_samples: int = 10_000,
        num_simulations: List[int] = [100, 1_000, 10_000, 100_000, 1_000_000],
        path: str = Path(__file__).parent.absolute(),
        prior_lower_bound: List[float] = [
            *LOWER_BOUND_FULL_FRONTAL,
            *LOWER_BOUND_SIDE_IMPACT,
            *LOWER_BOUND_ROLL_OVER,
        ],
        prior_upper_bound: List[float] = [
            *UPPER_BOUND_FULL_FRONTAL,
            *UPPER_BOUND_SIDE_IMPACT,
            *UPPER_BOUND_ROLL_OVER,
        ],
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
            "Full Frontal Driver HIC",
            "Full Frontal Driver Chest Deflection",
            "Full Frontal Driver Femur Load",
            "Full Frontal Driver NIJ",
            "Full Frontal Driver Neck Compression",
            "Full Frontal Driver Neck Tension",
            "Full Frontal Passanger HIC",
            "Full Frontal Passanger Chest Deflection",
            "Full Frontal Passanger Femur Load",
            "Full Frontal Passanger NIJ",
            "Full Frontal Passanger Neck Compression",
            "Full Frontal Passanger Neck Tension",
            "Side Pole Front HIC",
            "Side Pole Front Chest (Rib)",
            "Side Pole Front Abdomen",
            "Side Pole Front Pelvis",
            "Side Impact Front HIC",
            "Side Impact Front Chest (Rib)",
            "Side Impact Front Abdomen",
            "Side Impact Front Pelvis",
            "Side Impact Rear HIC",
            "Side Impact Rear Pelvis",
            "Roll Over Probability",
        ]

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

        full_frontal_crash = FullFrontalCrash(base_risk=self.base_risk)
        side_impact_crash = SideImpact(base_risk=self.base_risk)
        roll_over = RollOver(base_risk=self.base_risk)

        def simulator(parameters: torch.Tensor) -> torch.Tensor:
            """Simulates the US NCAP rating for a full frontal crash test.

            Args:
                parameters: Crash test measurements of shape (N, 23).

            Returns:
                torch.tensor: Relative risk of injury.
            """
            assert (
                parameters.shape[1] == self.dim_parameters
            ), f"Invalid number of parameters. Got {parameters.shape[1]}, expected 23."

            # Extracting parameters
            measurements_full_frontal = parameters[:, :12]
            measurements_side_impact = parameters[:, 12:20]
            measurements_roll_over = parameters[:, 20:]

            risk_full_frontal = full_frontal_crash(measurements_full_frontal)
            risk_side_impact = side_impact_crash(measurements_side_impact)
            risk_roll_over = roll_over(measurements_roll_over)

            risk_values = (
                (5 / 12) * risk_full_frontal
                + (4 / 12) * risk_side_impact
                + (3 / 12) * risk_roll_over
            )

            # Generating and adding Gaussian white noise to relative risk.
            noise = torch.randn_like(risk_values) * self.noise_level
            risk_values += noise

            return risk_values  # .reshape(-1, 1)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)
