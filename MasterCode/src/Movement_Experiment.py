from dataclasses import dataclass
import random
import time


@dataclass
class MovementPhase:
    """
    Defines one phase of a movement experiment.

    Examples
    --------
    MovementPhase("stationary", position=3000)

    MovementPhase(
        "centered",
        low=0,
        mid=3000,
        high=6000
    )

    MovementPhase(
        "random",
        low=0,
        high=6000
    )
    """
    type: str

    position: int | None = None
    low: int | None = None
    mid: int | None = None
    high: int | None = None

    speed: str = "1"
    speed_choices: list[str] | None = None


class MovementExperiment:

    def __init__(self, motor, phase_duration=40.0):
        self.motor = motor
        self.phase_duration = phase_duration

        self.start_time = None
        self.motor_log = []

        self.motor.set_position_callback(self._log_motor_position)

    # ---------------------------------------------------------
    # TIME
    # ---------------------------------------------------------

    def now(self):
        """Return elapsed experiment time."""
        if self.start_time is None:
            return time.perf_counter()

        return time.perf_counter() - self.start_time


    def _log_motor_position(self, position):
        timestamp = time.perf_counter() - self.start_time

        self.motor_log.append(
            (timestamp, position)
        )

    def wait_until(self, end_time):
        """Wait until the current phase ends while logging motor position."""
        while self.now() < end_time:
            self.motor.get_position()
            time.sleep(0.05)

    # ---------------------------------------------------------
    # MOVEMENT TYPES
    # ---------------------------------------------------------

    def target_to_zero(self, target, phase_end, speed="1"):
        """
        Move to target, return to zero, then remain there.
        """
        self.motor.move_to(target, speed=speed)

        time.sleep(1.0)

        self.motor.move_to(0, speed=speed)

        time.sleep(1.0)

        self.wait_until(phase_end)

    def two_fixed(self, low, high, phase_end, speed="1"):
        """
        Move between two fixed positions.
        """
        self.motor.move_to(low, speed=speed)

        time.sleep(0.5)

        self.motor.move_to(high, speed=speed)

        time.sleep(1.0)

        self.motor.move_to(low, speed=speed)

        time.sleep(0.2)

        self.wait_until(phase_end)

    def centered(self, low, mid, high, phase_end, speed="1"):
        """
        Move around a central position:

        mid -> high -> mid -> low -> mid
        """
        current_position = self.motor.get_position()

        if current_position != mid:
            self.motor.move_to(mid, speed=speed)

        self.motor.move_to(high, speed=speed)
        self.motor.move_to(mid, speed=speed)
        self.motor.move_to(low, speed=speed)
        self.motor.move_to(mid, speed=speed)

        self.wait_until(phase_end)

    def stationary(self, position, phase_end, speed="1"):
        """
        Move to a position and remain there.
        """
        current_position = self.motor.get_position()

        if current_position != position:
            self.motor.move_to(position, speed=speed)

        self.wait_until(phase_end)

    def random_movement(
        self,
        low,
        high,
        phase_end,
        speed_choices=None,
    ):
        """
        Randomly move between positions until the phase ends.
        """
        if speed_choices is None:
            speed_choices = ["1", "2", "3", "4"]

        while self.now() < phase_end:

            position = random.randint(low, high)
            speed = random.choice(speed_choices)

            self.motor.move_to(
                position,
                speed=speed,
            )

            # Random pause between movements
            sleep_time = random.uniform(0.5, 2.0)

            remaining = phase_end - self.now()

            if remaining <= 0:
                break

            time.sleep(min(sleep_time, remaining))

    # ---------------------------------------------------------
    # FUTURE MOVEMENT TYPES
    # ---------------------------------------------------------

    def sawtooth(self, low, high, phase_end, speed="1"):
        """
        Sawtooth movement.

        TODO:
        Implement continuous/discrete sawtooth trajectory.
        """
        raise NotImplementedError

    def sinusoidal(self, low, high, phase_end, speed="1"):
        """
        Sinusoidal movement.

        TODO:
        Implement sinusoidal trajectory.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # PHASE DISPATCH
    # ---------------------------------------------------------

    def run_phase(self, phase, phase_end):
        """
        Execute one MovementPhase.
        """

        if phase.type == "fixed":

            self.fixed(
                target=phase.position,
                phase_end=phase_end,
                speed=phase.speed,
            )

        elif phase.type == "two_fixed":

            self.two_fixed(
                low=phase.low,
                high=phase.high,
                phase_end=phase_end,
                speed=phase.speed,
            )

        elif phase.type == "centered":

            self.centered(
                low=phase.low,
                mid=phase.mid,
                high=phase.high,
                phase_end=phase_end,
                speed=phase.speed,
            )

        elif phase.type == "stationary":

            self.stationary(
                position=phase.position,
                phase_end=phase_end,
                speed=phase.speed,
            )

        elif phase.type == "random":

            self.random_movement(
                low=phase.low,
                high=phase.high,
                phase_end=phase_end,
                speed_choices=phase.speed_choices,
            )

        elif phase.type == "sawtooth":

            self.sawtooth(
                low=phase.low,
                high=phase.high,
                phase_end=phase_end,
                speed=phase.speed,
            )

        elif phase.type == "sinusoidal":

            self.sinusoidal(
                low=phase.low,
                high=phase.high,
                phase_end=phase_end,
                speed=phase.speed,
            )

        else:
            raise ValueError(
                f"Unknown movement phase type: {phase.type}"
            )

    # ---------------------------------------------------------
    # RUN EXPERIMENT
    # ---------------------------------------------------------

    def run(self, phases, start_time=None):
        """
        Run all phases sequentially.
        """

        if start_time is not None:
            self.start_time = start_time

        if self.start_time is None:
            self.start_time = time.perf_counter()

        for phase in phases:

            phase_start = self.now()
            phase_end = phase_start + self.phase_duration

            print(
                f"Starting phase: {phase.type} "
                f"(start={phase_start:.2f}s, "
                f"end={phase_end:.2f}s)"
            )

            self.run_phase(
                phase=phase,
                phase_end=phase_end,
            )