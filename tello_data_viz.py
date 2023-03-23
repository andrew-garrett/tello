import csv
import numpy as np
from math import pi, sqrt

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt


DATASET_NAME = "dataset001"


def readStateData():
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


def create_quad():
    """
    Quad is x-shaped with a center point with coordinate origin at center
    """
    quad_pose = {}
    quad_pose["center"] = (0, 0, 0)
    quad_pose["arm_pose"] = [0.125/sqrt(2), 0.125/sqrt(2)]
    quad_pose["prop_r"] = 0.05

    quad_body = None
    for j in range(-1, 2, 2):
        for k in range(-1, 2, 2):
            quad_arm = np.zeros((3, 11))
            quad_arm[:, 1] = np.array([j*quad_pose["arm_pose"][0], k*quad_pose["arm_pose"][0], 0])
            quad_arm[:, 2] = np.array([j*(quad_pose["arm_pose"][0] + quad_pose["prop_r"]), k*quad_pose["arm_pose"][0], 0])
            quad_arm[:, 3] = np.array([j*quad_pose["arm_pose"][0], k*quad_pose["arm_pose"][0], 0])
            quad_arm[:, 4] = np.array([j*(quad_pose["arm_pose"][0] - quad_pose["prop_r"]), k*quad_pose["arm_pose"][0], 0])
            quad_arm[:, 5] = np.array([j*quad_pose["arm_pose"][0], k*quad_pose["arm_pose"][0], 0])
            quad_arm[:, 6] = np.array([j*quad_pose["arm_pose"][0], k*(quad_pose["arm_pose"][0] + quad_pose["prop_r"]), 0])
            quad_arm[:, 7] = np.array([j*quad_pose["arm_pose"][0], k*quad_pose["arm_pose"][0], 0])
            quad_arm[:, 8] = np.array([j*quad_pose["arm_pose"][0], k*(quad_pose["arm_pose"][0] - quad_pose["prop_r"]), 0])
            quad_arm[:, 9] = np.array([j*quad_pose["arm_pose"][0], k*quad_pose["arm_pose"][0], 0])
            if quad_body is None:
                quad_body = quad_arm
            else:
                quad_body = np.concatenate((quad_body, quad_arm), axis=1)
    
    return quad_body


def viz_3D_animate(state_dataset):

    t_vect = state_dataset[0]
    t_init = t_vect[0]
    t_vect -= t_init
    dt_vect = np.zeros_like(t_vect)
    dt_vect[:-1] = t_vect[1:] - t_vect[:-1]

    orientation_vect = state_dataset[1:4]
    speed_vect = R.from_euler('x', pi).as_matrix() @ state_dataset[4:7]
    speed_init = np.mean(speed_vect[:, 0:10], axis=1) # speed_vect[:, 0]
    speed_vect -= speed_init[:, np.newaxis]
    accel_vect = state_dataset[7:10]
    accel_init = np.mean(accel_vect[:, 0:10], axis=1) # np.array([0., 0., -9.81]) # 
    accel_vect -= accel_init[:, np.newaxis]
    hpfilt = 0.5
    accel_vect = np.where(abs(accel_vect) < hpfilt, 0., accel_vect)
    accel_vect = -1.0*savgol_filter(accel_vect, 50, 3)
    vel_vect = np.cumsum(accel_vect*dt_vect, axis=1)
    # pos_vect = np.cumsum((vel_vect + 0.5*accel_vect*dt_vect)*dt_vect, axis=1)
    # pos_vect = np.cumsum((speed_vect*10. - 0.5*accel_vect*dt_vect)*dt_vect, axis=1)
    pos_vect = np.cumsum((speed_vect*10.)*dt_vect, axis=1) #  + 0.5*accel_vect*dt_vect

    quad_body = create_quad()

    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax_bound = np.max(np.abs(pos_vect))
    # ax.set_xlim(ax_bound, -1.*ax_bound)
    # ax.set_ylim(ax_bound, -1.*ax_bound)
    ax.set_zlim(ax_bound, -1.*ax_bound)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    def animate_quad(i):
        pose = (R.from_euler('xyz', orientation_vect[:, i]).as_matrix() @ quad_body) + pos_vect[:, i, np.newaxis]
        ax.clear()
        ax.set_xlim(ax_bound, -1.*ax_bound)
        ax.set_ylim(ax_bound, -1.*ax_bound)
        ax.set_zlim(ax_bound, -1.*ax_bound)
        ax.plot3D(*pose)

    def plot_path(i):
        path_so_far = pos_vect[:, :i]
        ax.plot3D(*path_so_far)
    
    for i in range(0, orientation_vect.shape[1], 10):
        animate_quad(i)
        plot_path(i)
        plt.draw()
        plt.pause(0.01)

    plt.show()


def viz_3D(pos_vect):

    ax_bound = np.max(np.abs(pos_vect))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim(ax_bound, -1.*ax_bound)
    ax.set_ylim(ax_bound, -1.*ax_bound)
    ax.set_zlim(0, -1.*ax_bound)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.plot3D(*pos_vect)

    return fig, ax

    
def viz(state_dataset):

    t_vect = state_dataset[0]
    t_init = t_vect[0]
    t_vect -= t_init
    dt_vect = np.zeros_like(t_vect)
    dt_vect[:-1] = t_vect[1:] - t_vect[:-1]

    speed_vect = R.from_euler('x', pi).as_matrix() @ state_dataset[4:7]
    # speed_init = speed_vect[:, 0]
    # speed_vect -= speed_init[:, np.newaxis]
    accel_vect = state_dataset[7:10]
    accel_init = np.mean(accel_vect[:, 0:10], axis=1) # np.array([0., 0., -9.81]) # 
    accel_vect -= accel_init[:, np.newaxis]
    hpfilt = 0.3
    accel_vect = np.where(abs(accel_vect) < hpfilt, 0., accel_vect)
    accel_vect = savgol_filter(accel_vect, 50, 3)

    fig, ax = plt.subplots(3, figsize=(20, 10))
    fig.suptitle("Linear Data over time")

    # ax[0].set_title("Principle Angles vs time")
    # ax[0].set_xlabel("t (s)")
    # ax[0].set_ylabel("Angle (rad)")
    # ax[0].plot(t_vect, state_dataset[1:4].T, label=["Roll", "Pitch", "Yaw"])
    # ax[0].legend()

    ax[0].plot(t_vect, accel_vect.T, label=["a_x", "a_y", "a_z"])
    ax[0].set_title("Linear Acceleration vs time")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("a (m/s^2)")
    ax[0].grid()
    ax[0].legend()

    vel_vect = np.cumsum(accel_vect*dt_vect, axis=1)
    ax[1].plot(t_vect, vel_vect.T, label=["v_x (derived)", "v_y (derived)", "v_z (derived)"])
    ax[1].plot(t_vect, speed_vect.T, label=["v_x (reported)", "v_y (reported)", "v_z (reported)"])
    ax[1].set_title("Linear Velocity vs time")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("v (m/s)")
    ax[1].grid()
    ax[1].legend()

    pos_vect = np.cumsum((vel_vect + 0.5*accel_vect*dt_vect)*dt_vect, axis=1)
    ax[2].plot(t_vect, pos_vect.T, label=["x", "y", "z"])
    ax[2].set_title("Linear Position vs time")
    ax[2].set_xlabel("t (s)")
    ax[2].set_ylabel("x (m)")
    ax[2].grid()
    ax[2].legend()

    # viz_3D(pos_vect=pos_vect)
    plt.show()

    return fig, ax


if __name__ == "__main__":
    state_dataset = readStateData()
    viz(state_dataset=state_dataset)
    viz_3D_animate(state_dataset=state_dataset)