from djitellopy import Tello
from threading import Thread
import csv
import time
import cv2
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

DATASET_NAME = "dataset000"

def readData():

    with open(f"./tellodatasets/{DATASET_NAME}/state-data.csv", "r") as f:
        state_dataset = None
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if i > 0 and len(row) > 1:
                state_vect = np.array(row, dtype=np.float64)[:, np.newaxis]
                if state_dataset is None:
                    state_dataset = state_vect
                else:
                    state_dataset = np.hstack((state_dataset, state_vect))
        return state_dataset


def plot3D_Data(state_dataset):
    t_vect = state_dataset[0]
    t_init = t_vect[0]
    t_vect -= t_init
    dt_vect = np.zeros_like(t_vect)
    dt_vect[:-1] = t_vect[1:] - t_vect[:-1]

    speed_vect = R.from_euler('x', pi).as_matrix() @ state_dataset[4:7]
    speed_init = speed_vect[:, 0]
    # speed_vect -= speed_init[:, np.newaxis]
    accel_vect = state_dataset[7:10]
    accel_init = np.array([0., 0., -9.81]) # accel_vect[:, 0]
    accel_vect -= accel_init[:, np.newaxis]

    pos_vect = np.cumsum((speed_vect + (accel_vect)*dt_vect)*dt_vect, axis=1)
    ax_bound = np.max(np.abs(pos_vect))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim(ax_bound, -1.*ax_bound)
    ax.set_ylim(ax_bound, -1.*ax_bound)
    ax.set_zlim(ax_bound, -1.*ax_bound)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.plot3D(*pos_vect)
    return

def plot2D_Data(state_dataset):

    t_vect = state_dataset[0]
    t_init = t_vect[0]
    t_vect -= t_init
    dt_vect = np.zeros_like(t_vect)
    dt_vect[:-1] = t_vect[1:] - t_vect[:-1]

    speed_vect = R.from_euler('x', pi).as_matrix() @ state_dataset[4:7]
    # speed_init = speed_vect[:, 0]
    # speed_vect -= speed_init[:, np.newaxis]
    accel_vect = state_dataset[7:10]
    accel_init = accel_vect[:, 0] # np.array([0., 0., -9.81]) # 
    accel_vect -= accel_init[:, np.newaxis]

    fig, ax = plt.subplots(3, figsize=(20, 10))
    fig.suptitle("Linear Data over time")

    # ax[0].set_title("Principle Angles vs time")
    # ax[0].set_xlabel("t (s)")
    # ax[0].set_ylabel("Angle (rad)")
    # ax[0].plot(t_vect, state_dataset[1:4].T, label=["Roll", "Pitch", "Yaw"])
    # ax[0].legend()


    ax[0].set_title("Linear Acceleration vs time")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("a (m/s^2)")
    ax[0].plot(t_vect, accel_vect.T, label=["a_x", "a_y", "a_z"])
    ax[0].legend()

    ax[1].set_title("Linear Velocity vs time")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("v (m/s)")
    # ax[1].plot(t_vect, speed_vect, label=["v_x", "v_y", "v_z"])
    vel_vect = np.cumsum(accel_vect*dt_vect, axis=1)
    ax[1].plot(t_vect, vel_vect[-1].T, label=["v_x (derived)", "v_y (derived)", "v_z (derived)"])
    ax[1].plot(t_vect, speed_vect[-1].T, label=["v_x (measured)", "v_y (measured)", "v_z (measured)"])
    ax[1].legend()

    ax[2].set_title("Linear Position vs time")
    ax[2].set_xlabel("t (s)")
    ax[2].set_ylabel("x (m)")
    pos_vect = np.cumsum((vel_vect + 0.5*accel_vect*dt_vect)*dt_vect, axis=1)
    ax[2].plot(t_vect, pos_vect.T, label=["x", "y", "z"])
    ax[2].legend()


    return


if __name__ == "__main__":
    state_dataset = readData()
    plot2D_Data(state_dataset=state_dataset)
    plot3D_Data(state_dataset=state_dataset)
    plt.show()