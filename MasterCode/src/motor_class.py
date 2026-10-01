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
        self.position_callback = None

    def send(self, cmd):
        self.ser.write(cmd.encode())
        time.sleep(0.05)
        resp = self.ser.read_all()
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

            time.sleep(0.05)

    def wait_until_ready(self, timeout=30):
        local_start = time.perf_counter()

        while True:
            self.ser.write(b"\r")
            resp = self.ser.read_all().decode(errors="ignore")

                        # parse multiple responses
            for line in resp.split("\r"):
                if line.startswith("8023") and len(line) >= 16:
                    try:
                        pos = int(line[-4:], 16)

                        if self.position_callback is not None:
                            self.position_callback(pos)

                    except ValueError:
                        pass


            if resp[4:].startswith("6006"):
                return

            if time.perf_counter() - local_start > timeout:
                raise TimeoutError("Motor timeout")

            time.sleep(0.05)


    def set_position_callback(self, callback):
        self.position_callback = callback


    def get_position(self):
        self.ser.write("\r".encode())

        time.sleep(0.05)

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


    def move_to(self, position, speed="F"):

        pos_hex = f"{position:04X}"  # convert to hex (e.g. 100 -> 0064)

        # Example: you may need to adapt protocol here
        self.send(f"840500230000000{speed}\r")
        self.send(f"040100230000{pos_hex}\r")
        self.send(f"840500230000000{speed}\r")

        self.wait_until_ready()

    def close(self):
        self.ser.close()