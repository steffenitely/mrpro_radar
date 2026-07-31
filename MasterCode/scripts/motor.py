import serial
import time
import math

class MotorController:
    def __init__(self, port="COM6"):
        self.ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.SEVENBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )

    def send(self, cmd):
        print("SEND:", cmd)
        self.ser.write(cmd.encode())
        time.sleep(0.05)
        resp = self.ser.read_all()
        print("RECV:", resp)
        return resp
    
    def initiate(self):
        print("Motor: initializing...")
        self.send("#01\r")
        self.send("8401001C00000002\r")
        time.sleep(0.5)
        self.send("\r")
        time.sleep(0.5)

    def reference(self):
        self.send("0401002800000003\r")
        self.send("840400280000000F\r")
        self.check_if_referenced()

    def check_if_referenced(self, timeout=30):
        start = time.time()
        while True:
            self.ser.write(b"\r")
            resp = self.ser.read_all().decode(errors="ignore")

            if resp[3:].startswith("24006"):
                return

            if time.time() - start > timeout:
                raise TimeoutError("Reference timeout")

            time.sleep(0.1)

    def check_if_ready_old(self, timeout=30):
        start = time.time()
        while True:
            self.ser.write(b"\r")
            resp = self.ser.read_all().decode(errors="ignore")
            print(resp)

            if resp[4:].startswith("6006"):
                return

            if time.time() - start > timeout:
                raise TimeoutError("Motor timeout")

            time.sleep(0.05)

    def check_if_ready(self, start_time=None, logger=None, timeout=30):
        local_start = time.perf_counter()

        while True:
            self.ser.write(b"\r")
            resp = self.ser.read_all().decode(errors="ignore")
            #print(resp)

            # unified timestamp
            if start_time is not None:
                t = time.perf_counter() - start_time
            else:
                t = None

            # parse multiple responses
            for line in resp.split("\r"):
                if line.startswith("8023") and len(line) >= 16:
                    try:
                        pos = int(line[-4:], 16)

                        if logger and t is not None:
                            logger(t, pos)

                    except ValueError:
                        pass

            if resp[4:].startswith("6006"):
                return

            if time.perf_counter() - local_start > timeout:
                raise TimeoutError("Motor timeout")

            time.sleep(0.05)

    def get_current_position(self):
        self.ser.write("\r".encode())

        time.sleep(0.1)

        resp = self.ser.read_all().decode(errors="ignore")

        if not resp:
            raise TimeoutError("No response from motor")

        # Remove duplicate responses and empty strings
        messages = [m for m in resp.split("\r") if m]

        # Take the last complete message
        msg = messages[-1]

        # Take only the last 4 hex digits
        position_hex = msg[-4:]

        position = int(position_hex, 16)

        return position

    def get_speed_from_position(self, pos):
        """
        Position-based speed profile (0 → 6000)
        """

        if pos < 1000:
            return 1
        elif pos < 2000:
            return 2
        elif pos < 4000:
            return 4
        elif pos < 5000:
            return 3
        else:
            return 1

    def move_position_window(self, target=6000, start_time=None, logger=None):

    
        pos_hex = f"{target:04X}"

        # set target once
        self.send("8405002300000001\r")
        self.send(f"040100230000{pos_hex}\r")
        self.send("8405002300000001\r")

        last_speed = None

        while True:
            # request status
            self.ser.write(b"\r")
            resp = self.ser.read_all().decode(errors="ignore")

            for line in resp.split("\r"):
                if line.startswith("8023") and len(line) >= 16:
                    try:
                        pos = int(line[-4:], 16)

                        # check if done
                        if abs(pos - target) < 10:
                            self.check_if_ready(start_time=start_time, logger=logger)
                            return

                        speed = self.get_speed_from_position(pos)

                        # only send if changed (VERY important)
                        if speed != last_speed:
                            self.send(f"840500230000000{speed:X}\r")
                            time.sleep(0.02)
                            last_speed = speed

                        # logging
                        if start_time is not None and logger:
                            t = time.perf_counter() - start_time
                            logger(t, pos)

                    except ValueError:
                        pass

            time.sleep(0.02)

    def move_smooth_4step(self, target=6000, start_time=None, logger=None):
        """
        Simple 4-step speed ramp:
        1 → 2 → 3 → 4 → 3 → 2 → 1
        """

        import time

        # set target ONCE
        pos_hex = f"{target:04X}"
        self.send("8405002300000001\r")
        self.send(f"040100230000{pos_hex}\r")
        self.send("8405002300000001\r")

        # simple speed pattern (fake sine)
        speed_pattern = [1, 2, 3, 4, 3, 2, 1]

        for speed in speed_pattern:
            speed_hex = f"{speed:X}"

            # send speed command
            self.send(f"840500230000000{speed_hex}\r")

            # optional logging
            if start_time is not None:
                t = time.perf_counter() - start_time

                self.ser.write(b"\r")
                resp = self.ser.read_all().decode(errors="ignore")

                for line in resp.split("\r"):
                    if line.startswith("8023") and len(line) >= 16:
                        try:
                            pos = int(line[-4:], 16)
                            if logger:
                                logger(t, pos)
                        except ValueError:
                            pass

            time.sleep(0.05)  # adjust smoothness

        # ensure motion completes
        self.check_if_ready(start_time=start_time, logger=logger)
        
    def move_to_a(self, start_time=None, logger=None):
        self.send("8405002300000001\r")
        self.send("0401002300001770\r")
        self.send("8405002300000001\r")
        self.check_if_ready(start_time=start_time, logger=logger)

    def move_to_zero(self, start_time=None, logger=None):
        self.send("840500230000000F\r")
        self.send("0401002300000000\r")
        self.send("840500230000000F\r")
        self.check_if_ready(start_time=start_time, logger=logger)

    def move_to(self, position, speed="F", start_time=None, logger=None):
        """
        position: int (0–1770)
        speed: hex string "0"–"F"
        """

        pos_hex = f"{position:04X}"  # convert to hex (e.g. 100 -> 0064)

        # Example: you may need to adapt protocol here
        self.send(f"840500230000000{speed}\r")
        self.send(f"040100230000{pos_hex}\r")
        self.send(f"840500230000000{speed}\r")

        self.check_if_ready(start_time=start_time, logger=logger)

    def close(self):
        self.ser.close()