from MasterCode.src import movement_experiment


class FakeMotor:
    def __init__(self, position=0):
        self.position = position
        self.moves = []

    def set_position_callback(self, callback):
        self.position_callback = callback

    def get_position(self):
        return self.position

    def move_to(self, position, speed="1"):
        self.moves.append((position, speed))
        self.position = position


def test_run_logs_callback_positions_as_elapsed_time(monkeypatch):
    motor = FakeMotor()
    experiment = movement_experiment.MovementExperiment(motor)

    monkeypatch.setattr(movement_experiment.time, "perf_counter", lambda: 11.5)

    experiment.run(phases=[], start_time=10.0)
    motor.position_callback(3356)

    assert experiment.motor_log == [
        {"t": 1.5, "position": 3356}
    ]


def test_centered_moves_high_mid_low_mid(monkeypatch):
    motor = FakeMotor(position=3000)
    experiment = movement_experiment.MovementExperiment(motor)

    monkeypatch.setattr(movement_experiment.time, "sleep", lambda _: None)

    experiment.centered(
        low=0,
        mid=3000,
        high=6000,
        phase_end=0,
    )

    assert motor.moves == [
        (6000, "1"),
        (3000, "1"),
        (0, "1"),
        (3000, "1"),
    ]
