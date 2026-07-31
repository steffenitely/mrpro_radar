import os
import socket
import struct
import numpy as np
from datetime import datetime
import time

# -----------------------------
# CONFIG
# -----------------------------
TCP_IP = "192.168.101.131"
PORT_CTRL = 5000
PORT_DATA = 5001

NUM_SAMPLES_PER_FRAME = 520
RX_OFFSET = 10   # remove metadata/header samples

SAVE_DIR_DATA = "Data/RadarTest"
SAVE_DIR_TIME = "Data/RadarTime"
os.makedirs(SAVE_DIR_DATA, exist_ok=True)
os.makedirs(SAVE_DIR_TIME, exist_ok=True)
# -----------------------------
# COMMANDS (vendor format)
# -----------------------------
m_parameters = struct.pack('!5IB',
                           0x061100e9,
                           0x00000000,
                           0x03000000,
                           0x00000000,
                           0xff010000,
                           0xfd)

m_mlbson   = struct.pack('!I', 0x020000fe)
m_run      = struct.pack('!I', 0x0d0000f3)
m_stop     = struct.pack('!I', 0x0e0000f2)
m_mlbsoff  = struct.pack('!I', 0x030000fd)
m_restart  = struct.pack('!I', 0x010000ff)


# -----------------------------
# SAFE RECV (ensures full 4 bytes)
# -----------------------------
def recv_exact(sock, nbytes):
    data = b""
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            raise ConnectionError("Socket closed unexpectedly")
        data += packet
    return data


# -----------------------------
# MAIN ACQUISITION
# -----------------------------
def collect_radar(stop_event):

    s_ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_data = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Connecting to radar...")
    s_ctrl.connect((TCP_IP, PORT_CTRL))
    s_data.connect((TCP_IP, PORT_DATA))

    # -----------------------------
    # START RADAR
    # -----------------------------
    print("Starting radar...")
    s_ctrl.send(m_parameters)
    s_ctrl.send(m_mlbson)
    s_ctrl.send(m_run)

    radar_frames = []
    timestamps = []

    start_time = time.perf_counter()

    try:
        while not stop_event.is_set():

            frame = []

            # -----------------------------
            # READ ONE FRAME
            # -----------------------------
            for _ in range(NUM_SAMPLES_PER_FRAME):
                raw = recv_exact(s_data, 4)
                val = struct.unpack('<f', raw)[0]
                frame.append(val)

            radar_frames.append(frame)
            timestamps.append(datetime.now())

    except KeyboardInterrupt:
        print("Stopped manually.")

    
    finally:
        # -----------------------------
        # STOP RADAR (guaranteed cleanup)
        # -----------------------------
        print("Stopping radar...")
        try:
            s_ctrl.send(m_stop)
            s_ctrl.send(m_mlbsoff)
            s_ctrl.send(m_restart)
        except:
            pass

        s_ctrl.close()
        s_data.close()

    end_time = time.perf_counter()
    print(f"Radar ran for {end_time - start_time:.3f}s")


    # -----------------------------
    # PROCESS DATA
    # -----------------------------
    radar_frames = np.array(radar_frames)

    print("Raw shape:", radar_frames.shape)

    # -----------------------------
    # RX SEPARATION (ASSUMPTION: alternating frames)
    # -----------------------------
    rx1 = []
    rx2 = []

    for i in range(1, len(radar_frames), 2):

        # RX1 = even frames
        rx1.append(radar_frames[i-1, RX_OFFSET:])

        # RX2 = odd frames
        rx2.append(radar_frames[i, RX_OFFSET:])

    rx1 = np.array(rx1)
    rx2 = np.array(rx2)
    #print(f"shape of rx1_t: {rx1.shape}")
    #print(f"shape of rx2_t: {rx2.shape}")

    radar_cube = np.array([rx1, rx2])

    #-------- RX Seperation for timestamps 
    
    #timeframes = np.array(timeframes)  # ensure numpy array

    rx1_t = timestamps[0::2]  # even indices: 0,2,4,...
    rx2_t = timestamps[1::2]
    #print(f"size of rx1_t: {len(rx1_t)}")
    #print(f"size of rx2_t: {len(rx2_t)}")

    min_len = min(len(rx1_t), len(rx2_t))

    rx1_t = rx1_t[:min_len]
    rx2_t = rx2_t[:min_len]
        
    time_cube = np.stack([rx1_t, rx2_t])


    return radar_cube, timestamps, time_cube


def plot_radar_data(data):
    import matplotlib.pyplot as plt
    from scipy.constants import c
    import numpy as np

    print(data.shape)

    rx1, rx2 = data[0], data[1]

    vmin = min(rx1.min(), rx2.min())
    vmax = max(rx1.max(), rx2.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(rx1, cmap="viridis", vmin=vmin, vmax=vmax,
                         interpolation="nearest", aspect="auto")
    axes[0].set_title("Channel 0 (Correlation)")

    im1 = axes[1].imshow(rx2, cmap="viridis", vmin=vmin, vmax=vmax,
                         interpolation="nearest", aspect="auto")
    axes[1].set_title("Channel 1 (Correlation)")

    cbar = fig.colorbar(im1, ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("Correlation strength")

    plt.show()

    # # Distance estimation
    # row_idx = rx1.shape[0] // 2
    # rx1_peak = np.argmax(rx1[row_idx])
    # rx2_peak = np.argmax(rx2[row_idx])

    # time_difference = abs(rx1_peak - rx2_peak) * 112e-12
    # distance = time_difference * c

    # print(f"The distance is {distance} meters")

    # fig2, ax = plt.subplots(figsize=(10, 4))

    # ax.plot(rx1[row_idx], label="Channel 0")
    # ax.plot(rx2[row_idx], label="Channel 1", linestyle="--")

    # ax.set_title(f"Correlation vs Samples (Row {row_idx})")
    # ax.set_xlabel("Sample index (0–509)")
    # ax.set_ylabel("Correlation strength")

    # ax.legend()
    # ax.grid(True)

    # plt.show()



def run_radar(stop_event, plot=False, save=False):
    print(f"Starting radar event controlled")

    radar_cube, timestamps, time_cube = collect_radar(stop_event)

    print("Radar finished")

    if plot:
        plot_radar_data(radar_cube)

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = os.path.join(SAVE_DIR_DATA, f"radar_{timestamp}.npz")

        np.savez(
            data_file,
            radar_cube=radar_cube,
            timestamps=timestamps,
            time_cube = time_cube
        )

        print(f"Saved: {data_file}")

    return radar_cube














#------------------------- OLD CODE JUNKYARD------------------
# -----------------------------
# MAIN ACQUISITION
# -----------------------------
# def collect_radar_old(duration_seconds=10):

#     s_ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s_data = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     print("Connecting to radar...")
#     s_ctrl.connect((TCP_IP, PORT_CTRL))
#     s_data.connect((TCP_IP, PORT_DATA))

#     # -----------------------------
#     # START RADAR
#     # -----------------------------
#     print("Starting radar...")
#     s_ctrl.send(m_parameters)
#     s_ctrl.send(m_mlbson)
#     s_ctrl.send(m_run)

#     radar_frames = []
#     timestamps = []

#     start_time = time.time()

#     try:
#         while time.time() - start_time < duration_seconds:

#             frame = []

#             # -----------------------------
#             # READ ONE FRAME
#             # -----------------------------
#             for _ in range(NUM_SAMPLES_PER_FRAME):
#                 raw = recv_exact(s_data, 4)
#                 val = struct.unpack('<f', raw)[0]
#                 frame.append(val)

#             radar_frames.append(frame)
#             timestamps.append(datetime.now())

#     except KeyboardInterrupt:
#         print("Stopped manually.")

#     # -----------------------------
#     # STOP RADAR
#     # -----------------------------
#     print("Stopping radar...")
#     s_ctrl.send(m_stop)
#     s_ctrl.send(m_mlbsoff)
#     s_ctrl.send(m_restart)

#     s_ctrl.close()
#     s_data.close()

#     # -----------------------------
#     # PROCESS DATA
#     # -----------------------------
#     radar_frames = np.array(radar_frames)

#     print("Raw shape:", radar_frames.shape)

#     # -----------------------------
#     # RX SEPARATION (ASSUMPTION: alternating frames)
#     # -----------------------------
#     rx1 = []
#     rx2 = []

#     for i in range(1, len(radar_frames), 2):

#         # RX1 = even frames
#         rx1.append(radar_frames[i-1, RX_OFFSET:])

#         # RX2 = odd frames
#         rx2.append(radar_frames[i, RX_OFFSET:])

#     rx1 = np.array(rx1)
#     rx2 = np.array(rx2)

#     radar_cube = np.array([rx1, rx2])



#     return radar_cube, timestamps


# def run_radar_old(duration, plot=False, save=False):
#     print(f"Starting radar for {duration}s")

#     radar_cube, timestamps = collect_radar(duration_seconds=duration)

#     print("Radar finished")

#     if plot:
#         plot_radar_data(radar_cube)

#     if save:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         data_file = os.path.join(SAVE_DIR_DATA, f"radar_data_{timestamp}.npy")
#         data_file_t = os.path.join(SAVE_DIR_TIME, f"timestamps_{timestamp}.npy")
#         np.save(data_file, radar_cube)
#         np.save(data_file_t, timestamps)
#         print(f"Saved: {data_file}")

#     return radar_cube
















