""" 
This file provides classes to evaluate the rating of a crash test according to
the US NCAP standard. The rating consists of a Full Frontal, Side Pole, Side
Impact and Roll Over test. The Side Pole and Side Impact ratings share the same
logic.

The rating formula can be found (e.g.) in the Safety Companion by carhs.
"""

import torch


class FullFrontalCrash:

    def __init__(self, base_risk: torch.tensor = torch.tensor([0.15])) -> None:
        """
        The FullFrontalCrash class provides functions to the relative risk in the
        case of a full frontal crash test for both driver and passanger.

        The class provides a call method that returns the relative risk for both
        driver and passenger, based on the provided measurements, in the setting
        of a full frontal crash test.
        """
        self.base_risk = base_risk

    @staticmethod
    def p_hic(hic: torch.tensor) -> torch.tensor:
        """
        Evaluates the cdf of a random variable from a parameterized normal
        distribution.

        The formula has the same parameterization for both driver and passenger.

        Args:
            hic: HIC value obtained from hardware tests or simulations.
        Returns:
            torch.tensor: Probability of head injury.
        """
        x = (torch.log(hic) - 7.45231) / 0.73998
        return torch.distributions.Normal(0.0, 1.0).cdf(x)

    @staticmethod
    def p_chest_driver(chest: torch.tensor) -> torch.tensor:
        """
        Provides the probability of chest injury based on the chest deflection.

        As the formula has different parametrization for driver and passenger,
        separate functions are provided.

        Args:
            chest: Chest deflection in mm.
        Returns:
            torch.tensor: Probability of chest injury.
        """
        return 1 / (1 + torch.exp(10.5456 - 1.568 * chest**0.4612))

    @staticmethod
    def p_femur_driver(femur: torch.tensor) -> torch.tensor:
        """
        Provides the probability of femur injury based on the femur load.

        As the formula has different parametrization for driver and passenger,
        separate functions are provided.

        Args:
            femur: Femur load in kN.
        Returns:
            torch.tensor: Probability of femur injury.
        """
        return 1 / (1 + torch.exp(5.795 - 0.5196 * femur))

    @staticmethod
    def p_nij(nij: torch.tensor) -> torch.tensor:
        """
        Provides the probability of neck injury based on the neck injury criterion.

        The formula has the same parameterization for both driver and passenger.

        Args:
            nij: NIJ value obtained during the crash test.

        Returns:
            torch.tensor: Probability of neck injury.
        """
        return 1 / (1 + torch.exp(3.2269 - 1.9688 * nij))

    @staticmethod
    def p_tension_and_compression_driver(n_input: torch.tensor) -> torch.tensor:
        """
        Provides the probability of neck injury based on the neck tension or
        compression.

        The formula for tension and compression has the exact same
        parametrization. Therefore, the same function is used for both.

        As the formula has different parametrization for driver and passenger,
        separate functions are provided.

        Args:
            n_input: Tension value obtained during the crash test.

        Returns:
            torch.tensor: Probability of neck injury.
        """
        return 1 / (1 + torch.exp(10.9745 - 2.375 * n_input))

    @staticmethod
    def p_chest_passenger(chest: torch.tensor) -> torch.tensor:
        """
        Provides the probability of chest injury based on the chest deflection.

        As the formula has different parametrization for driver and passenger,
        separate functions are provided.

        Args:
            chest: Chest deflection in mm.
        Returns:
            torch.tensor: Probability of chest injury.
        """
        return 1 / (1 + torch.exp(10.5456 - 1.721 * chest**0.4612))

    @staticmethod
    def p_femur_passenger(femur: torch.tensor) -> torch.tensor:
        """
        Provides the probability of femur injury based on the femur load.

        As the formula has different parametrization for driver and passenger,
        separate functions are provided.

        Args:
            femur: Femur load in kN.
        Returns:
            torch.tensor: Probability of femur injury.
        """
        return 1 / (1 + torch.exp(5.795 - 0.762 * femur))

    @staticmethod
    def p_tension_and_compression_passenger(n_input: torch.tensor) -> torch.tensor:
        """
        Provides the probability of neck injury based on the neck tension or
        compression.

        The formula for tension and compression has the exact same
        parametrization. Therefore, the same function is used for both.

        As the formula has different parametrization for driver and passenger,
        separate functions are provided.

        Args:
            n_input: Tension value obtained during the crash test.

        Returns:
            torch.tensor: Probability of neck injury.
        """
        return 1 / (1 + torch.exp(10.958 - 3.770 * n_input))

    def _rel_risk_single(self, rel_risk_value: torch.tensor) -> torch.tensor:
        """
        Provides the relative risk of injury for a single crash test.

        The formula to compute the relative risk is the same for both driver and
        passanger. However, they differ in the parameterization used to compute
        the risk. Therefore, the required functions are selected in the
        dedicated function below.

        Args:
            rel_risk_value: Relative risk of injury for a single crash test of
            shape (N,4).
        Returns:
            torch.tensor: Relative risk.
        """
        rel_risk = 1 - (1 - rel_risk_value[:, 0].reshape(-1, 1)) * (
            1 - rel_risk_value[:, 1].reshape(-1, 1)
        ) * (1 - rel_risk_value[:, 2].reshape(-1, 1)) * (
            1 - rel_risk_value[:, 3].reshape(-1, 1)
        )
        return rel_risk / self.base_risk

    def rel_risk_driver(self, parameters_ff_driver: torch.tensor) -> torch.tensor:
        """Computes the relative risk of injury for the driver in a full frontal crash test.

        Args:
            parameters_ff_driver: Crash test measurements for the driver with a required shape of (N,6).

        Returns:
            torch.tensor: Relative risk of injury.
        """
        phic = self.p_hic(parameters_ff_driver[:, 0].reshape(-1, 1))
        pchest = self.p_chest_driver(parameters_ff_driver[:, 1].reshape(-1, 1))
        pfemur = self.p_femur_driver(parameters_ff_driver[:, 2].reshape(-1, 1))
        pnij = self.p_nij(parameters_ff_driver[:, 3].reshape(-1, 1))
        pcompression = self.p_tension_and_compression_driver(
            parameters_ff_driver[:, 4].reshape(-1, 1)
        )
        ptension = self.p_tension_and_compression_driver(
            parameters_ff_driver[:, 5].reshape(-1, 1)
        )

        # Transform the neck injury probability
        pneck, _ = torch.max(
            torch.stack([pnij, pcompression, ptension], dim=0),
            dim=0,
        )

        # Catenate all risk values into a single tensor and provide it to the
        # above function to compute the relative risk.
        risk_values = torch.cat([phic, pchest, pfemur, pneck], dim=1)

        assert risk_values.shape == (
            parameters_ff_driver.shape[0],
            4,
        ), f"Expected shape (N,4), got {risk_values.shape}"

        return self._rel_risk_single(risk_values)

    def rel_risk_passenger(self, parameters_ff_passenger: torch.tensor) -> torch.tensor:
        """Computes the relative risk of injury for the passenger in a full frontal crash test.

        Args:
            parameters_ff_driver: Crash test measurements for the passenger with a required shape of (N,6).

        Returns:
            torch.tensor: Relative risk of injury.
        """
        phic = self.p_hic(parameters_ff_passenger[:, 0].reshape(-1, 1))
        pchest = self.p_chest_driver(parameters_ff_passenger[:, 1].reshape(-1, 1))
        pfemur = self.p_femur_driver(parameters_ff_passenger[:, 2].reshape(-1, 1))
        pnij = self.p_nij(parameters_ff_passenger[:, 3].reshape(-1, 1))
        pcompression = self.p_tension_and_compression_driver(
            parameters_ff_passenger[:, 4].reshape(-1, 1)
        )
        ptension = self.p_tension_and_compression_driver(
            parameters_ff_passenger[:, 5].reshape(-1, 1)
        )

        # Transform the neck injury probability
        pneck, _ = torch.max(
            torch.stack([pnij, pcompression, ptension], dim=0),
            dim=0,
        )

        # Catenate all risk values into a single tensor and provide it to the
        # above function to compute the relative risk.
        risk_values = torch.cat([phic, pchest, pfemur, pneck], dim=1)

        assert risk_values.shape == (
            parameters_ff_passenger.shape[0],
            4,
        ), f"Expected shape (N,4), got {risk_values.shape}"

        return self._rel_risk_single(risk_values)

    def __call__(self, measurements_full_frontal: torch.tensor) -> torch.tensor:
        """
        Provides the relative risk for a full frontal crash test.

        Args:
            measurements_full_frontal: Measurements obtained during the full
            frontal crash test.
        Returns:
            torch.tensor: Relative risk for driver and passenger in a full frontal.
        """
        driver_risk = self.rel_risk_driver(measurements_full_frontal[:, :6])
        passenger_risk = self.rel_risk_passenger(measurements_full_frontal[:, 6:])

        return (driver_risk + passenger_risk) / 2


class SideImpact:

    def __init__(self, base_risk: torch.tensor = torch.tensor([0.15])) -> None:
        """
        The SideImpact class provides functions to the relative risk for the front
        and rear seat passengers in the case of a side impact crash test. The logic
        is shared for a side pole and an side impact test.

        The class provides a call method that returns the relative risk for both
        front and rear, based on the provided measurements, in the setting
        of a side crash test.
        """
        self.base_risk = base_risk

    @staticmethod
    def p_hic(hic: torch.tensor) -> torch.tensor:
        """
        Evaluates the cdf of a random variable from a parameterized normal
        distribution.

        The formula has the same parameterization for both front and rear passenger.

        Args:
            hic: HIC value obtained from hardware tests or simulations.
        Returns:
            torch.tensor: Probability of head injury.
        """
        x = (torch.log(hic) - 7.45231) / 0.73998
        return torch.distributions.Normal(0.0, 1.0).cdf(x)

    @staticmethod
    def p_chest_front(chest: torch.tensor) -> torch.tensor:
        """
        Provides the probability of chest injury based on the max. rib deflection.

        This formula is applicable only for the front passenger.

        Args:
            chest: rib deflection in mm.
        Returns:
            torch.tensor: Probability of chest injury.
        """
        return 1 / (1 + torch.exp(5.3895 - 0.092 * chest))

    @staticmethod
    def p_abdomen_front(abdomen: torch.tensor) -> torch.tensor:
        """
        Provides the probability of abdomen injury based on the abdominal force.

        This formula is applicable only for the front passenger.

        Args:
            abdomen: Abdomen force in N.
        Returns:
            torch.tensor: Probability of abdomen injury.
        """
        return 1 / (1 + torch.exp(6.04 - 0.0021 * abdomen))

    @staticmethod
    def p_pelvis_front(pelvis: torch.tensor) -> torch.tensor:
        """
        Provides the probability of pelvis injury based on the pelvic force.

        This formula is applicable only for the front passenger.

        Args:
            pelvis: Pelvic force in N.
        Returns:
            torch.tensor: Probability of pelvis injury.
        """
        return 1 / (1 + torch.exp(7.597 - 0.001 * pelvis))

    @staticmethod
    def p_pelvis_rear(pelvis: torch.tensor) -> torch.tensor:
        """
        Provides the probability of pelvis injury based on the pelvic force.

        This formula is applicable only for the rear passenger.

        Args:
            pelvis: Pelvic force in N.
        Returns:
            torch.tensor: Probability of pelvis injury.
        """
        return 1 / (1 + torch.exp(6.3055 - 0.00094 * pelvis))

    def rel_risk_front(self, parameters_side_front: torch.tensor) -> torch.tensor:
        """Computes the relative risk of injury for the front passenger in a side crash test.

        Args:
            parameters_side_front: Crash test measurements for the front passenger with a required shape of (N,4).

        Returns:
            torch.tensor: Relative risk of injury.
        """
        phic = self.p_hic(parameters_side_front[:, 0].reshape(-1, 1))
        pchest = self.p_chest_front(parameters_side_front[:, 1].reshape(-1, 1))
        pabdomen = self.p_abdomen_front(parameters_side_front[:, 2].reshape(-1, 1))
        ppelvis = self.p_pelvis_front(parameters_side_front[:, 3].reshape(-1, 1))

        # Catenate all risk values into a single tensor and assert the correct
        # output shape.
        risk_values = torch.cat([phic, pchest, pabdomen, ppelvis], dim=1)
        assert risk_values.shape == (
            parameters_side_front.shape[0],
            4,
        ), f"Expected shape (N,4), got {risk_values.shape}"

        risk_value = 1 - (1 - phic) * (1 - pchest) * (1 - pabdomen) * (1 - ppelvis)

        return risk_value / self.base_risk

    def rel_risk_rear(self, parameters_side_rear: torch.tensor) -> torch.tensor:
        """Computes the relative risk of injury for the rear passenger in a side crash test.

        Args:
            parameters_side_rear: Crash test measurements for the rear passenger with a required shape of (N,2).

        Returns:
            torch.tensor: Relative risk of injury.
        """
        phic = self.p_hic(parameters_side_rear[:, 0].reshape(-1, 1))
        ppelvis = self.p_pelvis_rear(parameters_side_rear[:, 1].reshape(-1, 1))

        # Catenate all risk values into a single tensor and assert the correct
        # output shape.
        risk_values = torch.cat([phic, ppelvis], dim=1)
        assert risk_values.shape == (
            parameters_side_rear.shape[0],
            2,
        ), f"Expected shape (N,2), got {risk_values.shape}"

        risk_value = 1 - (1 - phic) * (1 - ppelvis)

        return risk_value / self.base_risk

    def rel_risk_front_pole(
        self, parameters_side_front_pole: torch.tensor
    ) -> torch.tensor:
        """Computes the relative risk of injury for the front passenger in a side pole crash test.

        The logic to compute the relative risk is the same as for a side impact
        on the rear passenger. 

        Args:
            parameters_side_front_pole: Crash test measurements for the front passenger with a required shape of (N,2).

        Returns:
            torch.tensor: Relative risk of injury.
        """
        return self.rel_risk_rear(parameters_side_front_pole)

    def __call__(self, measurements_side_crash: torch.tensor) -> torch.tensor:
        """Computes the relative risk for a side crash test.

        This includes the side mdb and side pole crash tests.

        Args:
            measurements_side_crash: Measurements for the side crash test of
            shape (N,10) where with the following order: side pole, mdb front, mdb rear.

        Returns:
            torch.tensor: Relative risk for front and rear passengers in pole
            and mdb crash scenarios.
        """
        risk_front_pole = self.rel_risk_front_pole(measurements_side_crash[:, :2])
        risk_front_mdb = self.rel_risk_front(measurements_side_crash[:, 2:6])
        risk_rear_mdb = self.rel_risk_rear(measurements_side_crash[:, 6:])

        # Aggregate the weighted risk for the front passanger
        risk_front = 0.2 * risk_front_pole + 0.8 * risk_front_mdb

        # Return the weighted risk for a side crash
        return (risk_front + risk_rear_mdb) / 2


class RollOver:

    def __init__(self, base_risk: torch.tensor = torch.tensor([0.15])) -> None:
        """
        The RollOver class provides functions to the relative risk for a roll
        over.
        """
        self.base_risk = base_risk

    def __call__(self, p_roll: torch.tensor) -> torch.tensor:
        """
        Provides the relative risk for a roll over event.

        Args:
            p_roll: Probability of roll over.
        Returns:
            torch.tensor: Relative risk.
        """
        return p_roll / self.base_risk
