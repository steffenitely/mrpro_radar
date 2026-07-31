import serial
import time

#----------------
def send(cmd):
    print("SEND:", cmd)
    ser.write(cmd.encode())
    time.sleep(0.1)
    resp = ser.read_all()
    print("RECV:", resp)
    return resp

def read():
    resp = ser.read_all()
    print("RECV:", resp)
    return resp

ser = serial.Serial(
    port="COM6",        # or /dev/ttyUSB0
    baudrate=9600,      # MUST match controller
    bytesize=serial.SEVENBITS,
    parity=serial.PARITY_EVEN,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

def check_if_ready(timeout=30):
    start = time.time()
    cmd = "\r"
    while True:
        ser.write(cmd.encode())
        resp = ser.read_all().decode(errors="ignore")
        print(resp)

        if  resp[4:].startswith("6006"):
            print("reached target position")
            return resp

        if time.time() - start > timeout:
            raise TimeoutError("Device did not respond in time")

        time.sleep(0.1)
        

def check_if_referenced(timeout=30):
    start = time.time()
    cmd = "\r"
    while True:
        ser.write(cmd.encode())
        resp = ser.read_all().decode(errors="ignore")
        print(resp)

        if resp[3:].startswith("24006"):
            print("reached reference position")
            return resp

        if time.time() - start > timeout:
            raise TimeoutError("Device did not respond in time")

        time.sleep(0.1)
        


DURATION = 15

#------------------------------------
# adress
send("#01\r")

#endstufe an
print("enable:")
send("8401001C00000002\r")




time.sleep(1)
send("\r")
time.sleep(1)

#send("840400280000000F\r") #ref speed
send("0401002800000003\r") #ref in negative Drehrichtung
send("840400280000000F\r") #ref speed
check_if_referenced()


#send("\r")
# PTP speed
#send("8405002300000003\r")
start_time = time.time()

while time.time() - start_time < DURATION:

    print("move to A...")
    
    time.sleep(0.1)
    # print("ask for speed:")
    # send("8005002300000000\r") #ask for speed
    # time.sleep(0.2)
    print("set speed:")
    send("840500230000000F\r") #speed
    send("0401002300001770\r") #pos: 1770 = 6000 #0000 = 0 #03E8 = 1000
    send("840500230000000F\r") #speed

    check_if_ready()
    time.sleep(0.1)
    #send("\r")
    print("move to 0...")
    # print("ask for speed:")
    # send("8005002300000000\r") #ask for speed
    # time.sleep(0.2)
    print("set speed:")
    send("840500230000000F\r") #speed
    send("0401002300000000\r") #pos: 1770 = 6000 #0000 = 0
    send("840500230000000F\r") #speed

    check_if_ready()
    time.sleep(0.1)

#time.sleep(20)

#send("8405002300000001\r") #speed
# time.sleep(0.5)
# send("840500230000000F\r") #speed
# send("0401002300001770\r") #pos: 1770 = 6000 #0000 = 0 #03E8 = 1000
# #send("8405002300000001\r") #speed

#send("8401001C00000001\r")
#send("8401001C00000001\r")
print("finished motion loop")



