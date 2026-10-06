import json

import numpy as np

from MasterCode.src import main_collect, movement_experiment


class FakeMotor:
    def __init__(self):
        self.position = 0
        self.position_callback = None
        self.moves = []
        self.closed = False

    def initiate(self):
        pass

    def reference(self):
        pass

    def move_to(self, position, speed="1"):
        self.moves.append((position, speed))
        self.position = position

    def get_position(self):
        if self.position_callback is not None:
            self.position_callback(self.position)
        return self.position

    def set_position_callback(self, callback):
        self.position_callback = callback

    def close(self):
        self.closed = True


class FakeRadar:
    def __init__(self):
        self.radar_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.time_data = np.array([[10, 11], [12, 13]])
        self.saved_data = None
        self.plotted_data = None

    def collect(self, stop_event):
        assert not stop_event.is_set()
        return self.radar_data, self.time_data

    def save(self, radar_data, time_data):
        self.saved_data = (radar_data, time_data)
        return "fake-radar-data.npz"

    def plot(self, radar_data):
        self.plotted_data = radar_data


def test_main_collect_hands_off_radar_and_motor_data(monkeypatch, tmp_path):
    motor = FakeMotor()
    radar = FakeRadar()

    monkeypatch.setattr(main_collect, "ask_save_data", lambda: True)
    monkeypatch.setattr(main_collect, "SAVE_DIR_LOG", str(tmp_path))
    monkeypatch.setattr(main_collect, "PHASE_DURATION", 0)
    monkeypatch.setattr(
        main_collect,
        "centered_phases",
        lambda: [movement_experiment.MovementPhase("stationary", position=3000)],
    )
    monkeypatch.setattr(main_collect.time, "sleep", lambda _: None)

    main_collect.main(motor=motor, radar=radar)

    assert radar.saved_data is not None
    np.testing.assert_array_equal(radar.saved_data[0], radar.radar_data)
    np.testing.assert_array_equal(radar.saved_data[1], radar.time_data)
    np.testing.assert_array_equal(radar.plotted_data, radar.radar_data)
    assert motor.moves == [(3000, "1")]
    assert motor.closed

    log_files = list(tmp_path.glob("time_log_*.json"))
    assert len(log_files) == 1
    experiment_data = json.loads(log_files[0].read_text())
    assert experiment_data["motor_log"]
    assert experiment_data["motor_log"][0]["position"] == 3000
    assert "t" in experiment_data["motor_log"][0]
