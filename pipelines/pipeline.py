import os, sys
import math
import time
from threading import Thread

from djitellopy import Tello


class TelloPipeline:
    def __init__(self) -> None:
        self.drone = Tello()
        self.drone.connect()
        self.T_INIT = time.time()

        
        self.drone.takeoff()
        time.sleep(2)