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
#import apriltag
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
    (90, 180),     # 채널 11
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

# 태그 좌표 (베이스링크 기준) 저장용 전역 딕셔너리
tag_xyz = {}

########################################
# 1. DirectNet: 관절값 (7개, 도 단위) → 센서 측정값 (3차원) 매핑 네트워크
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
# 2. DirectNet 기반 최적화 함수 (제어용)
########################################

class JointAngleOptimizer(threading.Thread):
    def __init__(self, model, initial_angles, target_position, iterations=200, lr=0.01, result_queue=None):
        super(JointAngleOptimizer, self).__init__()
        self.model = model
        self.initial_angles = initial_angles
        self.target_position = target_position
        self.iterations = iterations
        self.lr = lr
        self.result_queue = result_queue  # 결과 전달용 큐

    def run(self):
        optimized_angles = optimize_joint_angles_direct(self.model, self.initial_angles, self.target_position, 
                                                        iterations=self.iterations, lr=self.lr)
        if self.result_queue is not None:
            self.result_queue.put(optimized_angles)

def optimize_joint_angles_direct(model, initial_angles, target_position, iterations=200, lr=0.01):
    angles = torch.tensor(initial_angles, dtype=torch.float32, requires_grad=True)
    target = torch.tensor(target_position, dtype=torch.float32)
    optimizer = optim.Adam([angles], lr=lr)
    for i in range(iterations):
        optimizer.zero_grad()
        pred = model(angles)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        optimizer.step()
    return angles.detach().numpy()

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
                        # rvec, tvec가 이미 계산되었다고 가정
                        # tvec을 numpy 배열로 변환하고, shape를 (3,1)로 보장
                        tvec_501 = np.asarray(tvec, dtype=np.float32).reshape(3, 1)
                        # rvec를 회전 행렬로 변환
                        R, _ = cv2.Rodrigues(rvec)
                        # 태그 좌표계에서의 y축 오프셋 (태그 기준)
                        offset_tag = np.array([0, -0.3, 0], dtype=np.float32).reshape(3, 1)
                        # 오프셋을 카메라 좌표계로 변환
                        offset_camera = R.dot(offset_tag)
                        # 변환된 오프셋을 tvec에 더함
                        new_tvec = tvec_501 + offset_camera
                        # 새로운 tvec의 shape를 확인 (예: (3,1))
                        M = np.diag([1, -1, -1])
                        R_reflected = R.dot(M)
                        new_rvec, _ = cv2.Rodrigues(R_reflected)
                        # 새로운 tvec을 사용하여 축 그리기
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

                        # 다른 태그의 상대 회전 계산
                        relative_R = R_reflected_inv.dot(R_other)
                        relative_rvec, _ = cv2.Rodrigues(relative_R)
                        end_xyz[tag.tag_id] = [re_off_tvec[0,0],re_off_tvec[1,0],re_off_tvec[2,0]]
                        
                        text = f"{tag.tag_id} tvec: x:{relative_tvec[0,0]*100:.2f} y:{relative_tvec[1,0]*100:.2f} z:{relative_tvec[2,0]*100:.2f}"
                        tag_xyz[tag.tag_id] = [relative_tvec[0,0],relative_tvec[1,0],relative_tvec[2,0]]
                        #print(text,flush=True)                        
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
# 7. 엔드 이펙터 목표 좌표로 관절값 최적화 및 전송
########################################

def update_joint_angles_with_learning_using_xyz(tag_xyz, target_tag_id=11):
    """
    tag_xyz: 베이스링크 기준, 카메라에서 계산된 각 태그의 xyz 좌표 (딕셔너리)
    target_tag_id: 제어 목표 태그 ID (예: 11, 14, 15)
    """
    global end_xyz
    if target_tag_id not in end_xyz:
        print(f"태그 {target_tag_id} 좌표 정보 없음.")
        return current_joint_angles

    # 카메라에서 구한 태그 좌표 (베이스링크 기준)
    current_position = np.array(end_xyz[target_tag_id])
    # 해당 태그 오프셋 적용
    offset = np.array(tag_offset.get(target_tag_id, [0, 0, 0]))
    desired_candidate_position = current_position + offset

    print("카메라 측정 좌표:", current_position)
    print("엔드 이펙터 좌표:", desired_candidate_position)

    direct_model = DirectNet()
    new_angles = optimize_joint_angles_direct(direct_model, current_joint_angles, desired_candidate_position, iterations=200, lr=0.15)
    new_angles_clamped = clamp_angles(new_angles, joint_angle_limits)
    end_xyz={}
    return new_angles_clamped

def clamp_angles(angles, limits):
    clamped = []
    for angle, (lower, upper) in zip(angles, limits):
        clamped.append(max(min(angle, upper), lower))
    return clamped

def send_joint_angles_to_rpi(joint_angles):
    joint_angles = [float(x) for x in joint_angles]
    HOST = '192.168.50.4'
    PORT = 5000
    data = json.dumps(joint_angles)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((HOST, PORT))
            s.sendall(data.encode('utf-8'))
    except Exception as e:
        print("Raspberry Pi 전송 실패:", e)

########################################
# 8. 완료 신호 수신용 네트워크 리스너 (포트 6000)
########################################

complete_signal_event = threading.Event()

def complete_signal_listener():
    HOST = ''
    PORT = 6000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(6)
        print(f"[Listener] 포트 {PORT}에서 완료 신호 대기중...")
        while True:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    try:
                        message = json.loads(data.decode('utf-8'))
                        if message.get("status") == "complete":
                            print("[Listener] 완료 신호 수신됨.")
                            complete_signal_event.set()
                    except Exception as e:
                        print("[Listener] 메시지 파싱 에러:", e)

########################################
# 9. 제어 루프: tag_xyz 좌표를 기반으로 엔드 이펙터 제어
########################################

def control_loop_using_xyz(tag_xyz, result_queue, target_tag_id=13):
    """
    tag_xyz 딕셔너리에서 target_tag_id의 좌표를 가져와 오프셋을 적용한 후,
    해당 좌표로 이동하기 위한 관절 최적화 및 전송 수행.
    """
    global current_joint_angles, end_xyz
    if not complete_signal_event.is_set():
       return

    candidate_tags = {}
    for tag_id in [11,12,14,15]:
        if tag_id in end_xyz:
            candidate_tags[tag_id] = end_xyz[tag_id]
    
    if not candidate_tags:
        #print("[Control] 후보 태그 (0,3,7,1,4,8) 미검출")
        return
    
    selected_tag_id = None
    for tag_id in [11,12,14,15]:
        if tag_id in candidate_tags:
            selected_tag_id = tag_id
            break
    if selected_tag_id is None:
        print("[Control] 후보 태그 선택 실패")
        return
    
    current_position = np.array(end_xyz[selected_tag_id])

    target_position = np.array(tag_xyz[target_tag_id])
    if verify_end_effector_position(current_position, target_position):  
        print("[제어] 목표 위치 도달, 이동 중지")
        complete_signal_event.set()
        return  
    #print(f"[Control] 태그 {target_tag_id} 목표 좌표: {target_position}")

    new_angles = update_joint_angles_with_learning_using_xyz(tag_xyz, target_tag_id)
    current_joint_angles = new_angles
    send_joint_angles_to_rpi(current_joint_angles)
    complete_signal_event.clear()

def verify_end_effector_position(current_position, target_position, x_y_threshold=0.05, z_threshold=0.05):
    """
    엔드 이펙터의 현재 위치와 목표 위치를 비교하여, 
    x, y축은 ±0.05m, z축은 0.1m 이내면 도달했다고 판단.
    
    :param current_position: 현재 엔드 이펙터 좌표 (x, y, z)
    :param target_position: 목표 엔드 이펙터 좌표 (x, y, z)
    :param x_y_threshold: x, y 축 허용 오차 (기본값 0.05m)
    :param z_threshold: z 축 허용 오차 (기본값 0.1m)
    :return: 목표 도달 여부 (True / False)
    """
    current_position = np.array(current_position)
    target_position = np.array(target_position)
    
    # 오차 계산
    error = np.abs(current_position-target_position)
    
    # x, y는 ±0.05m, z는 0.1m 이내인지 확인
    if error[0] <= x_y_threshold and error[1] <= x_y_threshold and error[2] <= z_threshold:
        print(f"[검증 완료] 목표 도달: 현재 위치 {current_position}, 목표 {target_position}, 오차 {error}")
        return True
    else:
        print(f"[검증 실패] 목표 미도달: 현재 위치 {current_position}, 목표 {target_position}, 오차 {error}",flush=True)
        return False

########################################
# 10. 화면 출력 및 제어 루프 실행
########################################

def display_results():
    cv2.namedWindow("AprilTag Pose (Left & Right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Pose (Left & Right)", 1080, 720)
    result_queue = queue.Queue()  # 관절값 결과 수신용 큐
    while not stop_event.is_set():
        try:
            left_frame, left_tag_xyz = left_result_queue.popleft()
            right_frame, right_tag_xyz = right_result_queue.popleft()
        except IndexError:
            continue

        # 여기서는 왼쪽 카메라 결과를 기준으로 제어
        if left_tag_xyz:
            control_loop_using_xyz(left_tag_xyz, result_queue, target_tag_id=13)
        elif right_tag_xyz:
            control_loop_using_xyz(right_tag_xyz, result_queue, target_tag_id=13)
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
# 11. 스레드 생성 및 실행
########################################

threads = []
t_left_capture = threading.Thread(target=capture_camera, args=(0, left_frame_queue, "Left"), daemon=True)
t_right_capture = threading.Thread(target=capture_camera, args=(2, right_frame_queue, "Right"), daemon=True)
threads.extend([t_left_capture, t_right_capture])
t_left_process = threading.Thread(target=process_frames, args=(left_frame_queue, left_result_queue, (l_camera_matrix, l_dist_coeffs), "Left"), daemon=True)
t_right_process = threading.Thread(target=process_frames, args=(right_frame_queue, right_result_queue, (r_camera_matrix, r_dist_coeffs), "Right"), daemon=True)
threads.extend([t_left_process, t_right_process])
t_complete_listener = threading.Thread(target=complete_signal_listener, daemon=True)
threads.append(t_complete_listener)

for t in threads:
    t.start()

print("실시간 AprilTag 검출, DirectNet 제어 및 관절값 전송 중...\n'q'를 눌러 종료하세요.")
try:
    display_results()
except KeyboardInterrupt:
    stop_event.set()
for t in threads:
    t.join()
