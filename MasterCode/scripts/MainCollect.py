from asyncio.log import logger
import threading
import time
import json
from datetime import datetime
import os
import random


#from motor import run_motor, run_motor_once
from radar import run_radar, plot_radar_data
from motor import MotorController

SAVE_DIR_LOG = "Data/TimeLogs"
os.makedirs(SAVE_DIR_LOG, exist_ok=True)

DURATION = 20
PHASE_DURATION = 40
#TOTAL_DURATION = 440
waiting_time = DURATION/2

radar_result = {}
motor_log = []
experiment = {
    "start_time": None,
    "end_time": None,
    "motor_log": motor_log
}

while True:
    save_input = input("Do you want to save experiment data (radar + log)? (y/n): ").strip().lower()
    if save_input in ["y", "yes"]:
        SAVE_DATA = True
        break
    elif save_input in ["n", "no"]:
        SAVE_DATA = False
        break
    else:
        print("Please enter 'y' or 'n'.")

def log_motor(position, start_time):
    motor_log.append({
        "t": now(start_time),
        "position": position,
    })

def radar_worker(stop_event, SAVE_DATA):
    data = run_radar(stop_event, plot=False, save=SAVE_DATA)
    radar_result["data"] = data   # store result safely

def now(start_time):
    return time.perf_counter() - start_time

def log_motor_raw(t, position):
    motor_log.append({
        "t": t,
        "position": position
    })

def calibrate_movement(motor, start_time):
     #------- move during duration
    log_motor("0", start_time)
    while now(start_time) < DURATION:

        motor.move_to_a()
        log_motor("A", start_time)
        time.sleep(0.5)
        log_motor("A", start_time)
        motor.move_to_zero()
        log_motor("0", start_time)
        time.sleep(1)
        log_motor("0", start_time)

def rndm_movement(motor, start_time):
    log_motor(0, start_time)
    while now(start_time) < DURATION:
        pos = random.randint(0, 6000)
        speed = random.choice(list("3456789ABCDEF"))
        
        current_pos=motor.get_current_position()
        log_motor(current_pos, start_time)

        motor.move_to(pos, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(pos, start_time)

        time.sleep(random.uniform(0.5, 2))
        log_motor(pos, start_time)

def calibration_movement(motor, start_time):
    log_motor(0, start_time)
    while now(start_time) < DURATION:
        posA = 6000
        pos0 = 0
        speed = "1"

        current_pos=motor.get_current_position()
        log_motor(current_pos, start_time)
        motor.move_to(posA, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(posA, start_time)
        time.sleep(1)
        log_motor(posA, start_time)

        motor.move_to(pos0, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(pos0, start_time)
        time.sleep(1)
        log_motor(pos0, start_time)

def scheduled_movement(motor, start_time):
    half_time = DURATION / 2
    switched = False  # to trigger the 1-second pause only once

    

    # -------- FIRST HALF: calibration (run once only) --------
    posA = 5000
    pos0 = 1000
    speed = "1"

    # run calibration immediately once
    #log_motor(0, start_time)
    current_pos=motor.get_current_position()
    log_motor(current_pos, start_time)
    motor.move_to(posA, speed, start_time=start_time, logger=log_motor_raw)
    log_motor(posA, start_time)
    time.sleep(1)

    log_motor(posA, start_time)
    motor.move_to(pos0, speed, start_time=start_time, logger=log_motor_raw)
    log_motor(pos0, start_time)
    time.sleep(1)

    while now(start_time) < DURATION:

        current_time = now(start_time)

        # -------- WAIT UNTIL HALFTIME --------
        if current_time < half_time:
            time.sleep(0.05)

        # -------- SWITCH PHASE (only once) --------
        elif not switched:
            print("Switching to random movement...")
            time.sleep(1)
            switched = True
            log_motor(pos0, start_time)

        # -------- SECOND HALF: random --------
        else:
            pos = random.randint(2500, 3500)
            speed = random.choice(list("3456789ABCDEF"))
            
            current_pos=motor.get_current_position()
            log_motor(current_pos, start_time)
            
            motor.move_to(pos, speed, start_time=start_time, logger=log_motor_raw)
            log_motor(pos, start_time)
            time.sleep(random.uniform(0.5, 2))
            log_motor(pos, start_time)


def scheduled_phase_movement(motor, start_time):
    # PHASE_DURATION = 50
    # TOTAL_DURATION = 400

    def run_fixed_cycle(target_pos, phase_end):
        """Move to target_pos and back to 0  until phase ends."""
        speed = "1"
        
        # move to target
        current_pos = motor.get_position()
        log_motor(current_pos, start_time)
        motor.move_to(target_pos, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(target_pos, start_time)
        time.sleep(1)
        log_motor(target_pos, start_time)
        # move back to 0
        motor.move_to(0, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(0, start_time)
        time.sleep(1)
        print("reached 0")
        # wait remaining time
        while now(start_time) < phase_end:
            current_pos = motor.get_position()
            log_motor(current_pos, start_time)
            time.sleep(0.05)
            #("waiting...")

    def run_2fixed_cycle(low, high, phase_end):
        """Move to target_pos and back to 0  until phase ends."""
        speed = "1"
        
        # move to target
        current_pos = motor.get_position()
        log_motor(current_pos, start_time)
        motor.move_to(low, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(low, start_time)
        time.sleep(0.5)
        log_motor(low, start_time)
        # move to high
        motor.move_to(high, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(high, start_time)
        time.sleep(1)
        log_motor(high, start_time)
        print("reached high")
        motor.move_to(low, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(low, start_time)
        time.sleep(0.2)
        print("reached low")
        # wait remaining time
        while now(start_time) < phase_end:
            current_pos = motor.get_position()
            log_motor(current_pos, start_time)
            time.sleep(0.05)
            #("waiting...")       

    def run_centered_cycle(mid, low, high, phase_end):
        """Move symmetrically around mid position."""
        speed = "1"


        current_pos = motor.get_position()
        log_motor(current_pos, start_time)

        if current_pos != mid:
            print(f"Current position {current_pos} is not at mid {mid}. Moving to mid first.")
            motor.move_to(mid, speed, start_time=start_time, logger=log_motor_raw)
            motor.wait_until_ready(start_time=start_time, logger=logger)
            log_motor(mid, start_time)
            time.sleep(0.5)

        # move up
        motor.move_to(high, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(high, start_time)
        time.sleep(0.5)
        log_motor(high, start_time)

        # back to mid
        motor.move_to(mid, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(mid, start_time)
        time.sleep(0.5)
        log_motor(mid, start_time)

        # move down
        motor.move_to(low, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(low, start_time)
        time.sleep(0.5)
        log_motor(low, start_time)

        # back to mid again
        motor.move_to(mid, speed, start_time=start_time, logger=log_motor_raw)
        log_motor(mid, start_time)
        time.sleep(0.1)
        print("Completed centered cycle")

        # idle logging until phase ends
        while now(start_time) < phase_end:
            current_pos = motor.get_position()
            log_motor(current_pos, start_time)
            time.sleep(0.05)

    def run_random_cycle(low, high, speed_choices, phase_end):
        """Random movement within range until phase ends."""
        while now(start_time) < phase_end:
            pos = random.randint(low, high)
            speed = random.choice(speed_choices)

            #L
            current_pos = motor.get_position()
            log_motor(current_pos, start_time)
            motor.move_to(pos, speed, start_time=start_time, logger=log_motor_raw)
            log_motor(pos, start_time)
            time.sleep(random.uniform(0.5, 2))
            log_motor(pos, start_time)

    def run_stationary(mid, phase_end):
        """Stay still at mid position."""
        speed = "1"
        current_pos = motor.get_position()
        if current_pos != mid:
            print(f"Current position {current_pos} is not at mid {mid}. Moving to mid first.")
            motor.move_to(mid, speed, start_time=start_time, logger=log_motor_raw)
            motor.wait_until_ready(start_time=start_time, logger=logger)
            log_motor(mid, start_time)

        print("Stationary phase at mid position")

        while now(start_time) < phase_end:
            current_pos = motor.get_position()
            log_motor(current_pos, start_time)
            time.sleep(0.05)

    # -------- PHASE DEFINITIONS --------
    # phases = [
    #     ("fixed", 6000),
    #     ("fixed", 4000),
    #     ("fixed", 3000),
    #     ("fixed", 2000),
    #     ("fixed", 1000),
    #     ("fixed", 500),
    #     ("random", (0, 6000)),
    #     ("random", (1000, 5000)),
    #     ("random", (2000, 4000)),
    #     ("random", (2500, 3500)),
    #     ("random", (2750, 3250)),
        
    # ]
    phases_mid = [
        ("2fixed", (0, 6000)),
        ("2fixed", (1000, 5000)),
        ("2fixed", (1500, 4500)),
        ("2fixed", (2000, 4000)),
        ("2fixed", (2500, 3500)),
        ("2fixed", (2750, 3250)),
        ("random", (0, 6000)),
        ("random", (1000, 5000)),
        ("random", (2000, 4000)),
        ("random", (2500, 3500)),
        ("random", (2750, 3250)),  
    ]
    phases_mid_test = [
        ("2fixed", (0, 6000)),
        ("2fixed", (1000, 5000)),
        ("2fixed", (1500, 4500)),]

    phases_low = [
        ("fixed", 6000),
        ("fixed", 4000),
        ("fixed", 3000),
        ("fixed", 2000),
        ("fixed", 1000),
        ("fixed", 500),
        ("random", (0, 6000)),
        ("random", (0, 4000)),
        ("random", (0, 2000)),
        ("random", (0, 1000)),
        ("random", (0, 500)),
    ]

    phases_high = [
        ("2fixed", (0, 6000)),
        ("2fixed", (2000, 6000)),
        ("2fixed", (3000, 6000)),
        ("2fixed", (4000, 6000)),
        ("2fixed", (5000, 6000)),
        ("2fixed", (5500, 6000)),
        ("random", (0, 6000)),
        ("random", (2000, 6000)),
        ("random", (4000, 6000)),
        ("random", (5000, 6000)),
        ("random", (5500, 6000)),
    ]

    phases_centered = [
        ("stationary", 3000),  # <-- your PCA mean segment

        #("centered", (0, 3000, 6000)),
        # ("centered", (1000, 3000, 5000)),
        # ("centered", (1500, 3000, 4500)),
        # ("centered", (2000, 3000, 4000)),
        # ("centered", (2500, 3000, 3500)),
        # ("centered", (2750, 3000, 3250)),

        ("random", (0, 6000)),
        # ("random", (1000, 5000)),
        # ("random", (2000, 4000)),
        # ("random", (2500, 3500)),
        # ("random", (2750, 3250)),
        # ("random", (0, 3000)),
        # ("random", (3000, 6000)),
        
        # #some more segments
        # ("random", (4000, 6000)),
        # ("random", (5000, 6000)),
        # ("random", (5500, 6000)),

        # ("random", (0, 3000)),
        # ("random", (0, 2000)),
        # ("random", (0, 1000)),
        # ("random", (0, 500)),

    ]
    
    
    TOTAL_DURATION = len(phases_centered) * PHASE_DURATION

    phase_start_time = now(start_time)
    print(f"phase_start_time: {phase_start_time}")
    

    for phase in phases_centered:
        phase_end = phase_start_time + PHASE_DURATION

        print(f"Starting phase: {phase}")

        if phase[0] == "fixed":
            target = phase[1]
            print(f"start with {phase[0]},{target}")
            run_fixed_cycle(target, phase_end)

        elif phase[0] == "random":
            low, high = phase[1]
            run_random_cycle(low, high, list("3456789ABCDEF"), phase_end)
        
        elif phase[0] == "2fixed":
            low, high = phase[1]
            print(f"start with {phase[0]},{low}-{high}")
            run_2fixed_cycle(low, high, phase_end)

        elif phase[0] == "centered":
            low, mid, high = phase[1]
            print(f"start centered: {low}-{mid}-{high}")
            run_centered_cycle(mid, low, high, phase_end)

        elif phase[0] == "stationary":
            mid = phase[1]
            print(f"start stationary at {mid}")
            run_stationary(mid, phase_end)

        # prepare next phase
        phase_start_time = phase_end

        if now(start_time) >= TOTAL_DURATION:
            break


def main():

    stop_event = threading.Event()

    motor = MotorController()
    motor.initiate()
    print("Motor initiated")
    motor.reference()
    print("Motor referenced")
    time.sleep(2) # some wait needed so motor can adjust and to leave the room so radar doesnt have any other movement it can pick up
    motor.move_to(3000, 1, None, None)
    motor.wait_until_ready()
    time.sleep(0.5)

    #radar_thread = threading.Thread(target=radar_worker)

    radar_thread = threading.Thread(
    target=radar_worker,
    args=(stop_event, SAVE_DATA)
    )

    start_time = time.perf_counter()

    radar_thread.start()
    time.sleep(1)  # ensure radar started

    print("Starting Movement")


    
    #calibration_movement(motor, start_time)
    #scheduled_movement(motor, start_time)
    scheduled_phase_movement(motor, start_time)
    #rndm_movement(motor, start_time)



    stop_event.set()
    radar_thread.join()
    end_time = time.perf_counter()


    print("Threads finished")

    motor.close()

    
    #print(motor_log)
    
    if "data" in radar_result:
        plot_radar_data(radar_result["data"])


    experiment["start_time"] = start_time
    experiment["end_time"] = end_time
    experiment["start_time"] = float(experiment["start_time"])
    experiment["end_time"] = float(experiment["end_time"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
    SAVE_DIR_LOG,
    f"time_log_{timestamp}.json"
    )
    if SAVE_DATA:
        with open(log_file, "w") as f:
            json.dump(experiment, f, indent=2, default=str)
        print(f"Saved experiment log: {log_file}")
    else:
        print("Experiment data was not saved.")

if __name__ == "__main__":
    main()


