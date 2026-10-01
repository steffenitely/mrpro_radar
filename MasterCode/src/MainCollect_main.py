import json
import os
import threading
import time
from datetime import datetime

try:
    from .Movement_Experiment import MovementExperiment, MovementPhase
    from .motor_class import MotorController
    from .radar_class import Radar
except ImportError:
    # Allow this file to be run directly from the repository root.
    from Movement_Experiment import MovementExperiment, MovementPhase
    from motor_class import MotorController
    from radar_class import Radar


SAVE_DIR_LOG = "Data/TimeLogs"
PHASE_DURATION = 40.0


def ask_save_data():
    while True:
        save_input = input(
            "Do you want to save experiment data (radar + log)? (y/n): "
        ).strip().lower()
        if save_input in {"y", "yes"}:
            return True
        if save_input in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def radar_worker(radar, stop_event, result, save_data):
    try:
        radar_cube, timestamps, time_cube = radar.collect(stop_event)
        result["data"] = (radar_cube, timestamps, time_cube)
        if save_data:
            result["file"] = radar.save(
                radar_cube,
                timestamps,
                time_cube,
            )
    except Exception as error:
        result["error"] = error


def centered_phases():
    phases = [MovementPhase("stationary", position=3000)]

    for low, high in (
        (0, 6000),
        (1000, 5000),
        (1500, 4500),
        (2000, 4000),
        (2500, 3500),
        (2750, 3250),
    ):
        phases.append(
            MovementPhase(
                "centered",
                low=low,
                mid=3000,
                high=high,
            )
        )

    for low, high in (
        (0, 6000),
        (1000, 5000),
        (2000, 4000),
        (2500, 3500),
        (2750, 3250),
        (0, 3000),
        (3000, 6000),
        (4000, 6000),
        (5000, 6000),
        (5500, 6000),
        (0, 3000),
        (0, 2000),
        (0, 1000),
        (0, 500),
        (0, 6000),
    ):
        phases.append(
            MovementPhase(
                "random",
                low=low,
                high=high,
                speed_choices=list("3456789ABCDEF"),
            )
        )

    return phases


def main():
    save_data = ask_save_data()
    os.makedirs(SAVE_DIR_LOG, exist_ok=True)

    stop_event = threading.Event()
    radar_result = {}
    motor = MotorController()
    radar = Radar()
    movement = MovementExperiment(
        motor,
        phase_duration=PHASE_DURATION,
    )

    start_time = time.perf_counter()
    radar_thread = threading.Thread(
        target=radar_worker,
        args=(radar, stop_event, radar_result, save_data),
    )

    try:
        motor.initiate()
        motor.reference()
        time.sleep(2)
        motor.move_to(3000, speed="1")

        radar_thread.start()
        time.sleep(1)
        print("Starting Movement")
        movement.run(centered_phases(), start_time=start_time)
    finally:
        stop_event.set()
        if radar_thread.is_alive():
            radar_thread.join()
        motor.close()

    end_time = time.perf_counter()

    if "error" in radar_result:
        raise RuntimeError("Radar acquisition failed") from radar_result["error"]

    radar_data = radar_result.get("data")
    if radar_data is not None and radar_data[0].size:
        radar.plot(radar_data[0])

    experiment = {
        "start_time": float(start_time),
        "end_time": float(end_time),
        "motor_log": movement.motor_log,
    }

    if save_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            SAVE_DIR_LOG,
            f"time_log_{timestamp}.json",
        )
        with open(log_file, "w") as file:
            json.dump(experiment, file, indent=2, default=str)
        print(f"Saved experiment log: {log_file}")
    else:
        print("Experiment data was not saved.")


if __name__ == "__main__":
    main()
