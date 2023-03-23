import csv
import cv2
import numpy as np
import time
from math import pi, sqrt

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

import tello_data_viz

DATASET_NAME = "dataset001"

quad_body = tello_data_viz.create_quad()
plt.ion()
FIG = plt.figure()
AX = plt.axes(projection="3d")

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


def processFrame(src, feature_extractor):

    gray_im = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # feature_extractor.detectAndCompute(gray_im, None)
    # find the keypoints with ORB
    kp = feature_extractor.detect(gray_im,None)
    # compute the descriptors with ORB
    kp, des = feature_extractor.compute(gray_im, kp)
    # draw only keypoints location,not size and orientation
    kp_im = cv2.drawKeypoints(src, kp, None, color=(0,255,0), flags=0)
    return kp_im

def drawFrames(prev_im, curr_im, feature_extractor, feature_matcher):
    prev_im, curr_im = cv2.resize(prev_im, (480, 360)), cv2.resize(curr_im, (480, 360))
    # prev_im = cv2.rotate(prev_im, cv2.ROTATE_90_CLOCKWISE)
    gray_prev_im = cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY)
    prev_kp = feature_extractor.detect(gray_prev_im,None)
    prev_kp, prev_des = feature_extractor.compute(gray_prev_im, prev_kp)

    gray_curr_im = cv2.cvtColor(curr_im, cv2.COLOR_BGR2GRAY)
    curr_kp = feature_extractor.detect(gray_curr_im,None)
    curr_kp, curr_des = feature_extractor.compute(gray_curr_im, curr_kp)

    matches = feature_matcher.knnMatch(prev_des,curr_des,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, match in enumerate(matches):
        if len(match) > 1:
            if match[0].distance < 0.65*match[1].distance:
                matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)

    match_im = cv2.drawMatchesKnn(prev_im, prev_kp, curr_im, curr_kp, matches, None, **draw_params)
    
    
    # # extracted via colmap
    # dims = (960, 720)
    # fx, fy, cx, cy = (1152.000000, 1152.000000, 480.000000, 360.000000)
    # K = np.array([[fx, 0., cx],
    #               [0., fy, cy],
    #               [0., 0., 1.]])
    # prev_kp_np = np.array([list(kp.pt) for kp in prev_kp])
    # curr_kp_np = np.array([list(kp.pt) for kp in curr_kp])
    
    # E, mask = cv2.findEssentialMat(
    #     prev_kp_np[:min(len(prev_kp_np), len(curr_kp_np))], 
    #     curr_kp_np[:min(len(prev_kp_np), len(curr_kp_np))],
    #     cameraMatrix=K,
    #     method=cv2.RANSAC
    # )
    # points, R, t, mask = cv2.recoverPose(
    #     E, 
    #     prev_kp_np[:min(len(prev_kp_np), len(curr_kp_np))], 
    #     curr_kp_np[:min(len(prev_kp_np), len(curr_kp_np))], 
    #     cameraMatrix=K
    # )
    # print(R, t)

    # M_r = np.hstack((R, t))
    # M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    # P_l = np.dot(K_l,  M_l)
    # P_r = np.dot(K_r,  M_r)
    # point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))
    # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    # point_3d = point_4d[:3, :].T

    return match_im


def preprocess(src, payload):
    res = src # cv2.resize(src, (480, 360)) # 
    payload["curr_disp"] = res
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    payload["curr_image"] = res
    return payload


def extract_and_match(feature_extractor, feature_matcher, payload):
    curr_kp = feature_extractor.detect(payload["curr_image"], None)
    payload["curr_kp"], payload["curr_des"] = feature_extractor.compute(payload["curr_image"], curr_kp)
    if payload["prev_des"] is not None:
        matches = feature_matcher.knnMatch(payload["prev_des"], payload["curr_des"], k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        matches_masked = []
        # ratio test as per Lowe's paper
        for i, match in enumerate(matches):
            if len(match) > 1:
                if match[0].distance < 0.7*match[1].distance:
                    matchesMask[i] = [1, 0]
                    matches_masked.append([match[0]])
        # filter and reorder matches for manual essential matrix estimation
        return payload, (matches, matches_masked, matchesMask)
    return payload, (None, None, None)


def get_pose_and_pcl(payload, matches):
    if matches is not None:
        # sort_inds = np.array([[match[0].trainIdx, match[0].queryIdx] for match in matches], dtype=np.int).T

        curr_kp_np = np.array([list(payload["curr_kp"][match[0].trainIdx].pt) for match in matches])
        prev_kp_np = np.array([list(payload["prev_kp"][match[0].queryIdx].pt) for match in matches])

        dims = (960, 720)
        
        E, mask = cv2.findEssentialMat(
            prev_kp_np[:min(len(prev_kp_np), len(curr_kp_np))], 
            curr_kp_np[:min(len(prev_kp_np), len(curr_kp_np))],
            cameraMatrix=payload["cameraMatrix"],
            method=cv2.RANSAC
        )
        points, R, t, mask = cv2.recoverPose(
            E, 
            prev_kp_np[:min(len(prev_kp_np), len(curr_kp_np))], 
            curr_kp_np[:min(len(prev_kp_np), len(curr_kp_np))], 
            cameraMatrix=payload["cameraMatrix"]
        )

        M_curr = np.hstack((R, t))
        M_prev = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        P_prev = np.dot(K,  M_prev)
        P_curr = np.dot(K,  M_curr)
        point_4d_hom = cv2.triangulatePoints(P_prev, P_curr, np.expand_dims(curr_kp_np, axis=1), np.expand_dims(prev_kp_np, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        pcl = point_4d[:3, :].T

        return R, t, pcl
    return None, None, None


def display_data(payload, matches, matchesMask, curr_R, curr_T, curr_pcl):
    # Draw Feature Matching
    draw_params = dict(
        matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        matchesMask = matchesMask,
        flags = cv2.DrawMatchesFlags_DEFAULT)
    if payload["prev_disp"] is not None and payload["curr_disp"] is not None:
        match_im = cv2.drawMatchesKnn(payload["prev_disp"], payload["prev_kp"], payload["curr_disp"], payload["curr_kp"], matches, None, **draw_params)
        cv2.imshow("orb-im", match_im)

        # Draw Live Plotting
        AX.clear()
        AX.set_xlim(5, -5)
        AX.set_ylim(5, -5)
        AX.set_zlim(5, -5)
        AX.scatter(*curr_pcl.T)
        plt.draw()
    


def step(payload, curr_R, curr_T):
    payload["prev_disp"] = payload["curr_disp"]
    payload["prev_image"] = payload["curr_image"]
    payload["prev_kp"] = payload["curr_kp"]
    payload["prev_des"] = payload["curr_des"]
    payload["curr_image"] = None
    payload["curr_kp"] = None
    payload["curr_des"] = None

    # Update the net Rotation and Translation
    # payload["R"]
    return payload

def vSlam(video, feature_extractor, feature_matcher, cameraMatrix):
    """
    Main Loop for performing orb-slam

    Arguments:
        - featureExtractor: orb feature extractor
        - featureMatcher: flann feature matcher
        - cameraMatrix: intrinsic matrix
    """

    payload = {
        "cameraMatrix": cameraMatrix,
        "curr_disp": None,
        "prev_disp": None,
        "curr_image": None,
        "prev_image": None,
        "curr_kp": None,
        "prev_kp": None,
        "curr_des": None,
        "prev_des": None,
        "prev_R": np.eye(3),
        "prev_T": np.zeros((3, 1))
    }
    fps_counter = 0
    t_i = time.time()

    
    while video.isOpened():
        retval, frame = video.read()

        if retval:
            """
            1. Take the current frame and compute its features and save these (processed)
            2. If features from the previous frame exist, match them and threshold based on the ratio test
            3. For filtered matches, compute the essential matrix and recover R,t
            4. Compose Transform from R,t and to extract point cloud
            """
            fps_counter += 1
            payload = preprocess(frame, payload)
            payload, (matches, matches_masked, matchesMask) = extract_and_match(feature_extractor, feature_matcher, payload)
            curr_R, curr_T, curr_pcl = get_pose_and_pcl(payload, matches_masked)

            display_data(payload, matches, matchesMask, curr_R, curr_T, curr_pcl)

            payload = step(payload, curr_R, curr_T)

            if time.time() - t_i >= 3:
                print(f"fps: {fps_counter / (time.time() - t_i)}")
                t_i = time.time()
                fps_counter = 0

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()




if __name__ == "__main__":
    
    video = cv2.VideoCapture(f"./tellodatasets/{DATASET_NAME}/video.avi")

    # FEATURE EXTRACTOR #
    feature_extractor = cv2.ORB_create() # feature extractor for matching between consecutive frames
    # FEATURE MATCHER #
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    feature_matcher = cv2.FlannBasedMatcher(index_params,search_params)
    # INTRINSICS #
    fx, fy, cx, cy = (1152.000000, 1152.000000, 480.000000, 360.000000)
    K = np.array([[fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]])
    # VISUALIZATION #
    cv2.namedWindow("orb-im", cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty("orb-im", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # ORB SLAM #
    vSlam(video, feature_extractor, feature_matcher, K)

    curr_frame = None
    prev_frame = None
    while video.isOpened():
        retval, frame = video.read()

        if retval:
            curr_frame = frame # set the current frame
            if prev_frame is not None and curr_frame is not None:
                match_im = drawFrames(prev_frame, curr_frame, feature_extractor, feature_matcher) # process consecutive frames

                cv2.imshow("orb-im", match_im)
            curr_frame = None
            prev_frame = frame

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()
