import cv2
import numpy as np
import time
import queue
import threading
import math
import json
import socket
import torch
import torch.nn as nn
import torch.optim as optim
#from collections import deque  # deque는 그대로 사용
from collections import deque
import datetime

import os
import sys

# stderr을 /dev/null로 리디렉션하여 모든 경고 차단
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, sys.stderr.fileno())

import apriltag  # 이후에는 AprilTag 라이브러리를 정상적으로 사용할 수 있음

########################################
# 관절 제한 (도 단위)
########################################

joint_angle_limits = [
    (0, 0),         # 채널 10: 고정 0도
    (90, 180),      # 채널 11
    (100, 170),     # 채널 12
    (75, 195),      # 채널 13
    (75, 195),      # 채널 14
    (90, 90),       # 채널 15: 고정 90도
    (30, 30)        # 채널 6: 고정 30도
]

########################################
# 태그별 오프셋 (엔드 이펙터 보정용)
########################################

tag_offset = {
    11: [0, 0.08, -0.065],
    12: [0, 0.08, 0.095],
    13: [0, 0, 0],
    14: [-0.02, 0.015, 0],
    15: [0.03, 0.02, 0]
}

end_xyz = {}
tag_xyz = {}  # 베이스링크 기준 태그 좌표 저장

########################################
# 3. 카메라 캘리브레이션 데이터 로드
########################################

def load_calibration_data(path, default_matrix, default_dist):
    try:
        with np.load(path) as X:
            camera_matrix = X["camera_matrix"]
            dist_coeffs = X["distortion_coefficients"]
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"{path} 캘리브레이션 데이터를 로드하지 못했습니다. 기본값 사용: {e}")
        return default_matrix, default_dist

default_matrix = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0, 0, 1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)
l_camera_matrix, l_dist_coeffs = load_calibration_data("../camera_calibration/cam8_calibration_data.npz", default_matrix, default_dist)
r_camera_matrix, r_dist_coeffs = load_calibration_data("../camera_calibration/cam9_calibration_data.npz", default_matrix, default_dist)
default_tag_size = 0.02  # m

########################################
# 4. 큐 및 스레드 동기화 변수 설정
########################################

left_frame_queue = deque(maxlen=1)
right_frame_queue = deque(maxlen=1)
left_result_queue = deque(maxlen=1)
right_result_queue = deque(maxlen=1)
stop_event = threading.Event()

# 초기 관절 각도 (예시)
current_joint_angles = [0, 135, 135, 120, 120, 90, 90]

########################################
# 5. 카메라 캡처 함수
########################################

def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"{cam_label} - 현재 적용된 코덱: {chr(fourcc & 0xFF)}{chr((fourcc >> 8) & 0xFF)}{chr((fourcc >> 16) & 0xFF)}{chr((fourcc >> 24) & 0xFF)}")
    print(f"{cam_label} - 카메라 해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)} FPS")
    if not cap.isOpened():
        print(f"{cam_label} 카메라 열기 실패")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.append(frame)
            except IndexError:
                pass
        else:
            print(f"{cam_label} 프레임 읽기 실패")
    cap.release()

########################################
# 6. 태그 검출 및 좌표 추정 (AprilTag)
########################################

def process_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    """
    apriltag.Detector로 태그 검출 및 solvePnP를 통해 좌표 추정.
    계산된 좌표는 베이스링크 기준으로 변환하여 전역 tag_xyz에 저장.
    (여기서는 501 태그 기준 30cm 오프셋 적용 예시)
    """
    global tag_xyz, end_xyz
    detector = apriltag.Detector()
    camera_matrix, dist_coeffs = calibration_data
    R_reflected = None
    new_tvec = None
    while not stop_event.is_set():
        try:
            if frame_queue:
                frame = frame_queue.popleft()
            else:
                continue
        except IndexError:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        for tag in tags:
            # 태그 크기 설정
            if tag.tag_id >= 400:
                tag_size = 0.05
            else:
                tag_size = 0.025
            object_points = np.array([
                [-tag_size/2, -tag_size/2, 0],
                [ tag_size/2, -tag_size/2, 0],
                [ tag_size/2,  tag_size/2, 0],
                [-tag_size/2,  tag_size/2, 0]
            ], dtype=np.float32)
            image_points = tag.corners.astype(np.float32)
            if not is_valid_corners(image_points):
                print(f"[WARNING] {cam_label} - Tag {tag.tag_id} 코너 데이터가 유효하지 않음: {image_points}")
                continue
            try:
                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    print(f"[WARNING] {cam_label} - solvePnP 실패 Tag {tag.tag_id}")
                else:
                    if tag.tag_id == 501:
                        tvec_501 = np.asarray(tvec, dtype=np.float32).reshape(3, 1)
                        R, _ = cv2.Rodrigues(rvec)
                        offset_tag = np.array([0, -0.3, 0], dtype=np.float32).reshape(3, 1)
                        offset_camera = R.dot(offset_tag)
                        new_tvec = tvec_501 + offset_camera
                        M = np.diag([1, -1, -1])
                        R_reflected = R.dot(M)
                        new_rvec, _ = cv2.Rodrigues(R_reflected)
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, new_rvec, new_tvec, tag_size / 2)
                    elif R_reflected is not None and new_tvec is not None:
                        tvec = np.asarray(tvec, dtype=np.float32).reshape(3,1)
                        R_other, _ = cv2.Rodrigues(rvec)
                        offset = np.array(tag_offset[tag.tag_id]).reshape(3,1)
                        transformed_offset = R_other @ offset
                        off_tvec = tvec + transformed_offset
                        
                        R_reflected_inv = R_reflected.T
                        relative_tvec = R_reflected_inv.dot(tvec - new_tvec)
                        re_off_tvec = R_reflected_inv.dot(off_tvec - new_tvec)

                        # 다른 태그의 상대 회전 계산 (출력을 위해 사용)
                        relative_R = R_reflected_inv.dot(R_other)
                        relative_rvec, _ = cv2.Rodrigues(relative_R)
                        end_xyz[tag.tag_id] = [re_off_tvec[0,0], re_off_tvec[1,0], re_off_tvec[2,0]]
                        
                        # 아래 형식대로 태그 좌표 출력 (단위: cm)
                        if tag.tag_id ==11:
                            text = f"{tag.tag_id} tvec: x:{relative_tvec[0,0]*100:.2f} y:{relative_tvec[1,0]*100:.2f} z:{relative_tvec[2,0]*100:.2f}"
                            print(f"{relative_tvec[2,0]*100:.2f}")
                        if cam_label == "Left":
                            tag_xyz[tag.tag_id] = [relative_tvec[0,0], relative_tvec[1,0], relative_tvec[2,0]]
                        
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size/2)
            except cv2.error as e:
                print(f"[ERROR] {cam_label} - Tag {tag.tag_id} solvePnP 에러: {e}")
        try:
            result_queue.append((frame, tag_xyz.copy()))
        except IndexError:
            pass

def is_valid_corners(corners, min_distance=10):
    if corners.shape != (4, 2):
        return False
    for i in range(4):
        for j in range(i+1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < min_distance:
                return False
    return True

########################################
# 7. 태그 좌표 전송 함수 (제어 대신)
########################################

def send_tag_coords_to_server():
    """
    전역 변수 tag_xyz에 저장된 태그 좌표 정보를 소켓을 통해 지속적으로 전송합니다.
    전송 메시지 예시: {"timestamp": "...", "tag_coords": {태그ID: [x, y, z], ...}}
    """
    HOST = '192.168.50.4'  # 전송 대상 서버 IP (필요에 따라 수정)
    PORT = 7000           # 전송 대상 포트 (필요에 따라 수정)
    while not stop_event.is_set():
        if tag_xyz:
            message = {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                "tag_coords": tag_xyz
            }
            data = json.dumps(message)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect((HOST, PORT))
                    s.sendall(data.encode('utf-8'))
                print("태그 좌표 전송:", data)
            except Exception as e:
                #print("태그 좌표 전송 실패:", e)
                pass
        time.sleep(0.15)  # 1초 간격으로 전송

########################################
# 9. 화면 출력 및 결과 확인
########################################

def display_results():
    cv2.namedWindow("AprilTag Pose (Left & Right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Pose (Left & Right)", 1080, 720)
    result_queue = queue.Queue()  # 관절값 결과 수신용 큐 (여기서는 태그 좌표 확인용)
    while not stop_event.is_set():
        try:
            left_frame, left_tag_xyz = left_result_queue.popleft()
            right_frame, right_tag_xyz = right_result_queue.popleft()
        except IndexError:
            continue

        # 화면 출력 시 단순히 영상을 회전/크기 변경하여 표시합니다.
        cv2.resize(left_frame, (1280,720), interpolation=cv2.INTER_NEAREST)
        cv2.resize(right_frame, (1280,720), interpolation=cv2.INTER_NEAREST)
        left_disp = cv2.rotate(left_frame, cv2.ROTATE_90_CLOCKWISE)
        right_disp = cv2.rotate(right_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined = cv2.hconcat([left_disp, right_disp])
        cv2.putText(combined, datetime.datetime.now().strftime('%S.%f'), (1400, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("AprilTag Pose (Left & Right)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

########################################
# 10. 스레드 생성 및 실행
########################################

threads = []
# 카메라 캡처 스레드
t_left_capture = threading.Thread(target=capture_camera, args=(0, left_frame_queue, "Left"), daemon=True)
t_right_capture = threading.Thread(target=capture_camera, args=(2, right_frame_queue, "Right"), daemon=True)
threads.extend([t_left_capture, t_right_capture])
# 태그 검출 및 좌표 추정 스레드
t_left_process = threading.Thread(target=process_frames, args=(left_frame_queue, left_result_queue, (l_camera_matrix, l_dist_coeffs), "Left"), daemon=True)
t_right_process = threading.Thread(target=process_frames, args=(right_frame_queue, right_result_queue, (r_camera_matrix, r_dist_coeffs), "Right"), daemon=True)
threads.extend([t_left_process, t_right_process])
# 태그 좌표 전송 스레드 추가
t_tag_sender = threading.Thread(target=send_tag_coords_to_server, daemon=True)
threads.append(t_tag_sender)

for t in threads:
    t.start()

print("실시간 AprilTag 검출 및 태그 좌표 전송 중...\n'q'를 눌러 종료하세요.")
try:
    display_results()
except KeyboardInterrupt:
    stop_event.set()
for t in threads:
    t.join()
