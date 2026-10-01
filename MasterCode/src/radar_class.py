import os
import socket
import struct
import time
from datetime import datetime

import numpy as np


class Radar:
    def __init__(
        self,
        ip="192.168.101.131",
        port_ctrl=5000,
        port_data=5001,
        num_samples_per_frame=520,
        rx_offset=10,
        save_dir_data="Data/RadarTest",
        save_dir_time="Data/RadarTime",
    ):
        # Radar configuration
        self.ip = ip
        self.port_ctrl = port_ctrl
        self.port_data = port_data

        self.num_samples_per_frame = num_samples_per_frame
        self.rx_offset = rx_offset

        # Save paths
        self.save_dir_data = save_dir_data
        self.save_dir_time = save_dir_time

        os.makedirs(self.save_dir_data, exist_ok=True)
        os.makedirs(self.save_dir_time, exist_ok=True)

        # Sockets
        self.ctrl_socket = None
        self.data_socket = None

        # Commands
        self.m_parameters = struct.pack(
            "!5IB",
            0x061100e9,
            0x00000000,
            0x03000000,
            0x00000000,
            0xff010000,
            0xfd,
        )

        self.m_mlbson = struct.pack("!I", 0x020000fe)
        self.m_run = struct.pack("!I", 0x0d0000f3)
        self.m_stop = struct.pack("!I", 0x0e0000f2)
        self.m_mlbsoff = struct.pack("!I", 0x030000fd)
        self.m_restart = struct.pack("!I", 0x010000ff)

    # --------------------------------------------------
    # CONNECTION
    # --------------------------------------------------

    def connect(self):
        """Connect to the radar control and data sockets."""

        print("Connecting to radar...")

        self.ctrl_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.data_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.ctrl_socket.connect(
            (self.ip, self.port_ctrl)
        )

        self.data_socket.connect(
            (self.ip, self.port_data)
        )

        print("Connected.")

    # --------------------------------------------------
    # RADAR CONTROL
    # --------------------------------------------------

    def start(self):
        """Start radar acquisition."""

        print("Starting radar...")

        self.ctrl_socket.send(self.m_parameters)
        self.ctrl_socket.send(self.m_mlbson)
        self.ctrl_socket.send(self.m_run)

    def stop(self):
        """Stop radar acquisition."""

        print("Stopping radar...")

        try:
            self.ctrl_socket.send(self.m_stop)
            self.ctrl_socket.send(self.m_mlbsoff)
            self.ctrl_socket.send(self.m_restart)
        except Exception:
            pass

    # --------------------------------------------------
    # DATA RECEPTION
    # --------------------------------------------------

    @staticmethod
    def _recv_exact(sock, nbytes):
        """Receive exactly nbytes from a socket."""

        data = b""

        while len(data) < nbytes:
            packet = sock.recv(nbytes - len(data))

            if not packet:
                raise ConnectionError(
                    "Socket closed unexpectedly"
                )

            data += packet

        return data

    def _read_frame(self):
        """Read one complete radar frame."""

        frame = []

        for _ in range(self.num_samples_per_frame):

            raw = self._recv_exact(
                self.data_socket,
                4,
            )

            value = struct.unpack("<f", raw)[0]
            frame.append(value)

        return frame

    # --------------------------------------------------
    # ACQUISITION
    # --------------------------------------------------

    def collect(self, stop_event):
        """
        Acquire radar data until stop_event is set.

        Returns
        -------
        radar_cube : np.ndarray
            Radar data with shape (2, time, samples).
        timestamps : list
            Timestamp for every received frame.
        time_cube : np.ndarray
            Timestamps separated by RX channel.
        """

        self.connect()
        self.start()

        radar_frames = []
        timestamps = []

        start_time = time.perf_counter()

        try:

            while not stop_event.is_set():

                frame = self._read_frame()

                radar_frames.append(frame)
                timestamps.append(datetime.now())

        except KeyboardInterrupt:
            print("Stopped manually.")

        finally:
            self.stop()

            if self.ctrl_socket is not None:
                self.ctrl_socket.close()

            if self.data_socket is not None:
                self.data_socket.close()

        end_time = time.perf_counter()

        print(
            f"Radar ran for "
            f"{end_time - start_time:.3f}s"
        )

        # Convert raw data
        radar_frames = np.asarray(radar_frames)

        print("Raw shape:", radar_frames.shape)

        # Separate RX channels
        radar_cube, time_cube = self._separate_rx(
            radar_frames,
            timestamps,
        )

        return radar_cube, timestamps, time_cube

    # --------------------------------------------------
    # PROCESSING
    # --------------------------------------------------

    def _separate_rx(self, radar_frames, timestamps):
        """Separate alternating frames into RX1 and RX2."""

        rx1 = []
        rx2 = []

        for i in range(1, len(radar_frames), 2):

            rx1.append(
                radar_frames[
                    i - 1,
                    self.rx_offset:
                ]
            )

            rx2.append(
                radar_frames[
                    i,
                    self.rx_offset:
                ]
            )

        rx1 = np.asarray(rx1)
        rx2 = np.asarray(rx2)

        radar_cube = np.asarray([
            rx1,
            rx2,
        ])

        # Separate timestamps
        rx1_t = timestamps[0::2]
        rx2_t = timestamps[1::2]

        min_len = min(
            len(rx1_t),
            len(rx2_t),
        )

        rx1_t = rx1_t[:min_len]
        rx2_t = rx2_t[:min_len]

        time_cube = np.stack([
            rx1_t,
            rx2_t,
        ])

        return radar_cube, time_cube

    # --------------------------------------------------
    # PLOTTING
    # --------------------------------------------------

    def plot(self, data):
        """Plot the two radar receive channels."""

        import matplotlib.pyplot as plt

        print(data.shape)

        rx1 = data[0]
        rx2 = data[1]

        vmin = min(
            rx1.min(),
            rx2.min(),
        )

        vmax = max(
            rx1.max(),
            rx2.max(),
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            constrained_layout=True,
        )

        im0 = axes[0].imshow(
            rx1,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        axes[0].set_title(
            "Channel 0 (Correlation)"
        )

        im1 = axes[1].imshow(
            rx2,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        axes[1].set_title(
            "Channel 1 (Correlation)"
        )

        cbar = fig.colorbar(
            im1,
            ax=axes,
            location="right",
            shrink=0.9,
            pad=0.02,
        )

        cbar.set_label("Correlation strength")

        plt.show()

    # --------------------------------------------------
    # SAVING
    # --------------------------------------------------

    def save(self, radar_cube, timestamps, time_cube):
        """Save acquired radar data."""

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        data_file = os.path.join(
            self.save_dir_data,
            f"radar_{timestamp}.npz",
        )

        np.savez(
            data_file,
            radar_cube=radar_cube,
            timestamps=timestamps,
            time_cube=time_cube,
        )

        print(f"Saved: {data_file}")

        return data_file