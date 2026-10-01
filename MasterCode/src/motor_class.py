import serial
import time


class MotorController:
    """Controller for the motor serial protocol."""

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
        self._rx_buffer = b""


    # ---------------------------------------------------------
    # LIFECYCLE
    # ---------------------------------------------------------
    
    def initiate(self):
            print("Motor: initializing...")
            self._send_command("#01\r")
            self._send_command("8401001C00000002\r")
            time.sleep(0.5)
            self._send_command("\r")
            time.sleep(0.5)
    
    def reference(self):
        self._send_command("0401002800000003\r")
        self._send_command("840400280000000F\r")
        self._wait_until_referenced()

    def close(self):
        self.ser.close()


    # ---------------------------------------------------------
    # PUBLIC MOTOR API
    # ---------------------------------------------------------

    def move_to(self, position, speed="F"):

        print(f"Motor: moving to position {position}")
        pos_hex = f"{position:04X}"

        self._send_command(f"840500230000000{speed}\r")
        self._send_command(f"040100230000{pos_hex}\r") 
        self._send_command(f"840500230000000{speed}\r")
        time.sleep(0.05)

        self._wait_until_ready(position)

    def get_position(self, timeout=1.0):
        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:
            messages = self._send_command("\r")

            for message in messages:
                position = self._parse_position(message)
                if position is not None:
                    return position

            time.sleep(0.05)

        raise TimeoutError("No valid position received from motor")

    def set_position_callback(self, callback):
        self.position_callback = callback


    # ---------------------------------------------------------
    # WAITING AND STATUS
    # ---------------------------------------------------------

    def _wait_until_referenced(self, timeout=30):
        start = time.perf_counter()

        while True:
            self.ser.write(b"\r")

            messages = self._read_messages()

            for message in messages:

                if len(message) >= 8 and message[3:].startswith("24006"):
                    print("Motor: referenced")
                    return

            if time.perf_counter() - start > timeout:
                raise TimeoutError("Reference timeout")

            time.sleep(0.05)

    def _wait_until_ready(self, target_position, timeout=40):
        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:
            messages = self._send_command("\r")

            for message in messages:
                position = self._parse_position(message)

                if (
                    message.startswith("80236006")
                    and position == target_position
                ):
                    print("Motor: reached target position")
                    time.sleep(0.05)
                    return

            time.sleep(0.05)

        self.debug_buffers()
        raise TimeoutError(
            f"Motor did not reach target position {target_position}"
        )


    # ---------------------------------------------------------
    # SERIAL TRANSPORT
    # ---------------------------------------------------------

    def _send_command(self, cmd):
        self.ser.write(cmd.encode())
        self.ser.flush()
        time.sleep(0.05)

        messages = self._read_messages()
        self._process_position_messages(messages)

        return messages

    
    # ---------------------------------------------------------
    # MESSAGE PROCESSING
    # ---------------------------------------------------------

    def _read_messages(self):
        if self.ser.in_waiting > 0:
            raw = self.ser.read(self.ser.in_waiting)
            self._rx_buffer += raw

        messages = []

        while b"\r" in self._rx_buffer:
            raw_message, self._rx_buffer = self._rx_buffer.split(b"\r", 1)

            message = raw_message.decode("ascii", errors="ignore")

            if message:
                messages.append(message)

        return messages

    def _process_position_messages(self, messages):
        processed = []

        for line in messages:
            position = self._parse_position(line)
            if position is not None:
                self._handle_position(position)
            processed.append((line, position))

        return processed
    
    def _parse_position(self, message):
        if not message.startswith("8023") or len(message) < 16:
            return None

        try:
            return int(message[-4:], 16)
        except ValueError:
            return None

    def _handle_position(self, position):
        if self.position_callback is not None:
            self.position_callback(position)

    def debug_buffers(self):
        waiting = self.ser.in_waiting
        if waiting:
            self._rx_buffer += self.ser.read(waiting)

        parts = self._rx_buffer.split(b"\r")

        print("\n--- BUFFER DEBUG ---")
        print(f"Serial bytes moved to software buffer: {waiting}")
        print(f"Complete pending messages: {len(parts) - 1}")

        for raw_message in parts[:-1]:
            if raw_message:
                print(f"  {raw_message.decode('ascii', errors='backslashreplace')!r}")

        if parts[-1]:
            print(f"Partial message: {parts[-1]!r}")

        print("--------------------\n")
