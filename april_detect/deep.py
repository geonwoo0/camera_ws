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
import apriltag
from collections import deque
import datetime
import os
import sys

# stderr을 /dev/null로 리디렉션하여 모든 경고 차단
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, sys.stderr.fileno())

########################################
# 1. 관절 제한 (도 단위)
########################################

joint_angle_limits = [
    (0, 0),         # 고정 0도
    (100, 170),     # Joint 1
    (135, 135),     # Joint 2 (고정)
    (70, 225),      # Joint 3
    (70, 225),      # Joint 4
    (90, 90),       # Joint 5 (고정)
    (30, 30)        # Joint 6 (고정)
]

########################################
# 2. 태그 오프셋 및 위치 저장
########################################

tag_offset = {
    11: [0, 0.08, -0.062],
    13: [0, 0, 0],
    14: [-0.026, 0.018, 0],
    15: [0.035, 0.023, 0]
}

end_xyz = {}
tag_xyz = {}

########################################
# 3. DirectNet 모델 (신경망 기반 관절 최적화)
########################################

class DirectNet(nn.Module):
    def __init__(self):
        super(DirectNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.fc(x)

########################################
# 4. Jacobian Transpose 기반 역기구학 (IK)
########################################
import numpy as np

def dh_transform(theta, d, a, alpha):
    """
    Denavit-Hartenberg (DH) 변환 행렬 계산
    :param theta: 관절 회전 (rad)
    :param d: 링크 오프셋
    :param a: 링크 길이
    :param alpha: 링크 회전
    :return: 4x4 변환 행렬
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def forward_kinematics(joint_angles):
    """
    주어진 관절 각도로 엔드 이펙터의 위치를 계산
    :param joint_angles: [θ1, θ2, θ3, θ4, θ5, θ6] (rad)
    :return: 3D 엔드 이펙터 위치 [x, y, z]
    """

    # 링크 길이 (URDF 기반)
    L1, L2, L3, L4, L5, L6 = 0.089,0.06, 0.096, 0.06, 0.094, 0.095
    
    # DH 파라미터 테이블 (θ, d, a, α)
    DH_params = [
        (joint_angles[0], L1, 0, 0),         # Base → Link1 (Z축 회전)
        (joint_angles[1], 0, L2, -np.pi/2),  # Link1 → Link2 (X축 회전)
        (joint_angles[2], 0, L3, np.pi/2),   # Link2 → Link3 (Z축 회전)
        (joint_angles[3], 0, L4, -np.pi/2),  # Link3 → Link4 (X축 회전)
        (joint_angles[4], 0, L5, 0),         # Link4 → Link5 (X축 회전)
        (joint_angles[5], L6, 0, np.pi/2)    # Link5 → Link6 (Z축 회전)
    ]

    # 베이스 좌표계
    T = np.eye(4)

    # 각 링크에 대해 변환 행렬 계산
    for theta, d, a, alpha in DH_params:
        T = T @ dh_transform(theta, d, a, alpha)

    # 최종 엔드 이펙터 위치
    end_effector_pos = T[:3, 3]
    
    return end_effector_pos

def compute_jacobian(joint_angles, delta=1e-6):
    """
    주어진 관절 각도에서 Jacobian 행렬을 수치미분 방식으로 계산
    :param joint_angles: 현재 관절 각도 (rad)
    :param delta: 미소 변화량 (기본값 1e-6)
    :return: 3x6 Jacobian 행렬
    """
    joint_angles = np.array(joint_angles)
    num_joints = len(joint_angles)
    jacobian = np.zeros((3, num_joints))

    # 현재 엔드 이펙터 위치
    base_position = forward_kinematics(joint_angles)

    for i in range(num_joints):
        perturbed_angles = joint_angles.copy()
        perturbed_angles[i] += delta  # i번째 관절을 delta만큼 변화

        perturbed_position = forward_kinematics(perturbed_angles)  # 변화된 위치

        # 미분 (f(x+Δx) - f(x)) / Δx
        jacobian[:, i] = (perturbed_position - base_position) / delta

    return jacobian

def jacobian_transpose_ik(current_angles, target_pos, learning_rate=0.1, iterations=100):
    """
    Jacobian Transpose를 사용한 역기구학
    :param current_angles: 현재 관절 각도 (rad)
    :param target_pos: 목표 엔드 이펙터 좌표 [x, y, z]
    :param learning_rate: 학습률 (기본값 0.1)
    :param iterations: 반복 횟수
    :return: 최적화된 관절 각도 (rad)
    """
    angles = np.array(current_angles, dtype=np.float32)

    for _ in range(iterations):
        ee_pos = forward_kinematics(angles)
        error = np.array(target_pos) - ee_pos

        if np.linalg.norm(error) < 1e-3:  # 오차가 작으면 중단
            break

        J = compute_jacobian(angles)
        delta_theta = learning_rate * J.T @ error  # Jacobian Transpose 방법 적용
        angles += delta_theta

    return angles


########################################
# 5. 카메라 캘리브레이션 데이터 로드
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

default_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)

l_camera_matrix, l_dist_coeffs = load_calibration_data("../camera_calibration/cam8_calibration_data.npz", default_matrix, default_dist)
r_camera_matrix, r_dist_coeffs = load_calibration_data("../camera_calibration/cam9_calibration_data.npz", default_matrix, default_dist)

########################################
# 6. 카메라 프레임 처리 및 AprilTag 검출
########################################

def process_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    detector = apriltag.Detector()
    camera_matrix, dist_coeffs = calibration_data
    while not stop_event.is_set():
        if frame_queue:
            frame = frame_queue.popleft()
        else:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        for tag in tags:
            object_points = np.array([
                [-0.025, -0.025, 0], [0.025, -0.025, 0],
                [0.025, 0.025, 0], [-0.025, 0.025, 0]
            ], dtype=np.float32)
            image_points = tag.corners.astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if success:
                tag_xyz[tag.tag_id] = tvec.flatten()
        result_queue.append((frame, tag_xyz.copy()))

########################################
# 7. Raspberry Pi로 관절각도 전송
########################################

def send_joint_angles_to_rpi(joint_angles):
    HOST, PORT = '192.168.50.4', 5000
    data = json.dumps([float(x) for x in joint_angles])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((HOST, PORT))
            s.sendall(data.encode('utf-8'))
    except Exception as e:
        print("전송 실패:", e)

########################################
# 8. 실시간 제어 루프
########################################

def control_loop():
    global current_joint_angles
    target_id = 13  # 목표 태그 ID
    while not stop_event.is_set():
        if target_id in tag_xyz:
            target_position = tag_xyz[target_id]
            new_angles = jacobian_transpose_ik(current_joint_angles, target_position)
            current_joint_angles = new_angles
            send_joint_angles_to_rpi(current_joint_angles)
        time.sleep(0.1)

########################################
# 9. UI 및 디스플레이
########################################

def display_results():
    cv2.namedWindow("AprilTag Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Pose", 1080, 720)
    while not stop_event.is_set():
        try:
            left_frame, left_tag_xyz = left_result_queue.popleft()
            right_frame, right_tag_xyz = right_result_queue.popleft()
        except IndexError:
            continue
        if left_tag_xyz:
            control_loop()
        combined = cv2.hconcat([left_frame, right_frame])
        cv2.putText(combined, datetime.datetime.now().strftime('%S.%f'), (1400, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("AprilTag Pose", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

########################################
# 10. 스레드 실행
########################################

stop_event = threading.Event()
left_frame_queue, right_frame_queue = deque(maxlen=1), deque(maxlen=1)
left_result_queue, right_result_queue = deque(maxlen=1), deque(maxlen=1)

threads = [
    threading.Thread(target=process_frames, args=(left_frame_queue, left_result_queue, (l_camera_matrix, l_dist_coeffs), "Left"), daemon=True),
    threading.Thread(target=process_frames, args=(right_frame_queue, right_result_queue, (r_camera_matrix, r_dist_coeffs), "Right"), daemon=True)
]

for t in threads:
    t.start()

print("AprilTag 감지 및 로봇 제어 중... 'q'를 눌러 종료")
try:
    display_results()
except KeyboardInterrupt:
    stop_event.set()

for t in threads:
    t.join()
