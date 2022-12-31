from djitellopy import Tello
from threading import Thread
import os, sys
import csv
import time
import cv2
import numpy as np
from math import pi

keepRecording = True
DATASET_NAME = "dataset001"

def videoRecording():
    with open(f"./tellodatasets/{DATASET_NAME}/frame-timestamps.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["timestamp"])

        height, width, _ = frame_reader.frame.shape
        video = cv2.VideoWriter(f"./tellodatasets/{DATASET_NAME}/video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))

        while keepRecording:
            csv_writer.writerow([time.time()])
            video.write(frame_reader.frame)
            time.sleep(1 / 30)

        video.release()

def dataCollection():

    with open(f"./tellodatasets/{DATASET_NAME}/state-data.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["timestamp", "roll", "pitch", "yaw", "vgx", "vgy", "vgz", "ax", "ay", "az", "h", "t_flight"])

        while keepRecording:
            try:
                csv_writer.writerow([
                    time.time(), # timestamp, s
                    tello.get_roll() * (pi / 180.), # roll, rad
                    tello.get_pitch() * (pi / 180.), # pitch, rad
                    tello.get_yaw() * (pi / 180.), # yaw, rad
                    tello.get_speed_x() / 100., # vgx, m/s
                    tello.get_speed_y() / 100., # vgy, m/s
                    tello.get_speed_z() / 100., # vgz, m/s
                    tello.get_acceleration_x() / 100., # ax, m/s^2
                    tello.get_acceleration_y() / 100., # ay, m/s^2
                    tello.get_acceleration_z() / 100., # az, m/s^2
                    tello.get_height() / 100., # h, m
                    float(tello.get_flight_time()), # t_flight, s
                ])
                time.sleep(0.01)
            except:
                time.sleep(0.01)



if __name__ == "__main__":
    try:
        os.mkdir(f"./tellodatasets/{DATASET_NAME}")
    except:
        pass
    tello = Tello()
    # tello.LOGGER.setLevel("DEBUG")
    tello.connect()
    t_init = time.time()
    tello.streamon()
    # tello.set_video_direction(0)
    frame_reader = tello.get_frame_read()
    
    videoRecordingThread = Thread(target=videoRecording)
    videoRecordingThread.start()

    dataCollectionThread = Thread(target=dataCollection)
    dataCollectionThread.start()
    
    try:
        tello.takeoff()
        time.sleep(1)

        # tello.go_xyz_speed(100, 0, 0, 10)
        rc_control = [0, 0, 0, 0]
        tello.send_rc_control(*rc_control)
        while time.time() - t_init < 30:
            # cv2.imshow("drone", frame_reader.frame)
            # waitkey = cv2.waitKey(1)
            # if waitkey == ord("q"):
            #     break
            # tello.send_rc_control(0, 20, 0, 100)
            if int(time.time() - t_init) % 5 == 0:
                rc_control[1] *= -1
                tello.send_rc_control(*rc_control)
            time.sleep(1)

        # tello.move_left(100)
        # tello.rotate_counter_clockwise(90)
        # tello.move_forward(100)

        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    finally:
        cv2.destroyAllWindows()
        keepRecording = False
        videoRecordingThread.join()
        dataCollectionThread.join()
        tello.end()