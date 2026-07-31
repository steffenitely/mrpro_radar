import serial
import time

def run_motor(duration=15, port="COM6"):

    ser = serial.Serial(
        port=port,
        baudrate=9600,
        bytesize=serial.SEVENBITS,
        parity=serial.PARITY_EVEN,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    def send(cmd):
        print("SEND:", cmd)
        ser.write(cmd.encode())
        time.sleep(0.05)
        resp = ser.read_all()
        print("RECV:", resp)
        return resp

    def check_if_ready(timeout=30):
        start = time.time()
        while True:
            ser.write("\r".encode())
            resp = ser.read_all().decode(errors="ignore")

            if resp[4:].startswith("6006"):
                return

            if time.time() - start > timeout:
                raise TimeoutError("Motor timeout")

            time.sleep(0.1)

    def get_current_position():
        ser.write("\r".encode())

        time.sleep(0.05)

        resp = ser.read_all().decode(errors="ignore")

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
    
    def check_if_referenced(timeout=30):
        start = time.time()
        while True:
            ser.write("\r".encode())
            resp = ser.read_all().decode(errors="ignore")

            if resp[3:].startswith("24006"):
                return

            if time.time() - start > timeout:
                raise TimeoutError("Reference timeout")

            time.sleep(0.1)

    print("Motor: initializing...")

    send("#01\r")
    send("8401001C00000002\r")

    time.sleep(0.5)
    send("\r")
    time.sleep(0.5)


    send("0401002800000003\r") #reference
    send("840400280000000F\r") #ref speed
    check_if_referenced()

    start_time = time.time()

    while time.time() - start_time < duration:
        time.sleep(0.1)
        print("Testing Current Position")
        position=get_current_position()
        print(position)
        time.sleep(0.1)
        print("Motor: move to A")
        send("840500230000000F\r")
        send("0401002300001000\r")
        send("840500230000000F\r")
        check_if_ready()

        time.sleep(0.1)
        print("Testing Current Position")
        position=get_current_position()
        print(position)
        time.sleep(0.1)
        
        print("Motor: move to 0")
        send("840500230000000F\r")
        send("0401002300000050\r")
        send("840500230000000F\r")
        check_if_ready()

    print("Motor finished")
    ser.close()


def run_motor_once(duration=15, port="COM6"):

    ser = serial.Serial(
        port=port,
        baudrate=9600,
        bytesize=serial.SEVENBITS,
        parity=serial.PARITY_EVEN,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    def send(cmd):
        print("SEND:", cmd)
        ser.write(cmd.encode())
        time.sleep(0.1)
        resp = ser.read_all()
        print("RECV:", resp)
        return resp

    def check_if_ready(timeout=30):
        start = time.time()
        while True:
            ser.write("\r".encode())
            resp = ser.read_all().decode(errors="ignore")

            if resp[4:].startswith("6006"):
                return

            if time.time() - start > timeout:
                raise TimeoutError("Motor timeout")

            time.sleep(0.1)

    def check_if_referenced(timeout=30):
        start = time.time()
        while True:
            ser.write("\r".encode())
            resp = ser.read_all().decode(errors="ignore")

            if resp[3:].startswith("24006"):
                return

            if time.time() - start > timeout:
                raise TimeoutError("Reference timeout")

            time.sleep(0.1)

    print("Motor: initializing...")

    send("#01\r")
    send("8401001C00000002\r")

    time.sleep(0.5)
    send("\r")
    time.sleep(0.5)


    send("0401002800000003\r") #reference
    send("840400280000000F\r") #ref speed
    check_if_referenced()

    start_time = time.time()

    time.sleep(5)
    print("Motor: move to A")
    send("840500230000000F\r")
    send("0401002300000064\r")
    send("840500230000000F\r")
    check_if_ready()


    print("Motor finished")
    ser.close()   


#Main

run_motor(duration=5, port="COM6")





            