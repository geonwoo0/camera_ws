import cv2
import numpy as np
import math
import json
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from pupil_apriltags import Detector
from collections import deque
import datetime
import queue
import threading

########################################
# 관절 제한 (도 단위)
########################################
joint_angle_limits = [
    (0, 0),       # 채널 10: 고정 0도
    (90, 135),    # 채널 11
    (100, 135),   # 채널 12
    (80, 135),    # 채널 13
    (80, 135),    # 채널 14
    (90, 90),     # 채널 15: 고정 90도
    (30, 30)      # 채널 6: 고정 30도
]

########################################
# 태그별 목표 거리 설정
########################################
target_distances = {
    0: 0.15,
    3: 0.15,
    7: 0.15,
    1: 0.05,
    4: 0.07,
    8: 0.07
}

########################################
# 전역 변수
########################################
current_joint_angles = [0, 135, 135, 135, 135, 90, 90]  # 시작 관절값
stop_event = threading.Event()
complete_signal_event = threading.Event()

left_frame_queue = deque(maxlen=1)
right_frame_queue = deque(maxlen=1)
left_result_queue = deque(maxlen=1)
right_result_queue = deque(maxlen=1)

########################################
# 모델 정의 (한 번만 생성 & 재사용)
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

# 전역 모델(가중치 로드 가능)
global_model = DirectNet()
global_model.eval()  # 추론 모드

########################################
# 관절값 최적화 스레드
########################################
class JointAngleOptimizer(threading.Thread):
    def __init__(self, model, initial_angles, target_position, iterations=200, lr=0.01, result_queue=None):
        super(JointAngleOptimizer, self).__init__()
        self.model = model
        self.initial_angles = initial_angles
        self.target_position = target_position
        self.iterations = iterations
        self.lr = lr
        self.result_queue = result_queue
        self._stop_event = threading.Event()

    def run(self):
        # 비동기 최적화
        optimized = optimize_joint_angles_direct(
            self.model, 
            self.initial_angles, 
            self.target_position, 
            iterations=self.iterations, 
            lr=self.lr
        )
        if self.result_queue:
            self.result_queue.put(optimized)

def optimize_joint_angles_direct(model, initial_angles, target_position, iterations=200, lr=0.01):
    # torch.no_grad()는 여기서는 최적화에 그래디언트가 필요하므로 제외(추론만 할 때 사용)
    angles = torch.tensor(initial_angles, dtype=torch.float32, requires_grad=True)
    target = torch.tensor(target_position, dtype=torch.float32)

    optimizer = optim.Adam([angles], lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad()
        pred = model(angles)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        optimizer.step()
    return angles.detach().numpy()

########################################
# AprilTag 검출 함수
########################################
def process_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    detector = Detector(families='tag36h11', nthreads=2, quad_decimate=1.5)
    camera_matrix, dist_coeffs = calibration_data

    while not stop_event.is_set():
        if frame_queue:
            frame = frame_queue.pop()  # 최신 프레임만
        else:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        pose_dict = {}
        for tag in tags:
            if tag.tag_id < 0:
                continue

            # 태그 크기 설정
            tag_size = 0.05 if tag.tag_id >= 400 else 0.018
            object_points = np.array([
                [-tag_size/2, -tag_size/2, 0],
                [ tag_size/2, -tag_size/2, 0],
                [ tag_size/2,  tag_size/2, 0],
                [-tag_size/2,  tag_size/2, 0]
            ], dtype=np.float32)
            image_points = tag.corners.astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, 
                camera_matrix, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                pose_dict[tag.tag_id] = (rvec, tvec)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size/2)

        result_queue.append((frame, pose_dict))

########################################
# 카메라 캡처 함수
########################################
def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print(f"[Error] {cam_label} 카메라 열기 실패")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] {cam_label} 프레임 읽기 실패")
            continue
        frame_queue.append(frame)
    cap.release()

########################################
# 제어 루프
########################################
def control_loop(left_pose, right_pose, result_queue):
    global current_joint_angles

    base_tag_id = 501
    if base_tag_id in left_pose:
        base_pose = left_pose[base_tag_id]
    elif base_tag_id in right_pose:
        base_pose = right_pose[base_tag_id]
    else:
        # 기준 태그 없음
        return

    rvec_base, tvec_base = base_pose
    Rb, _ = cv2.Rodrigues(rvec_base)
    T_base = np.eye(4)
    T_base[:3,:3] = Rb
    T_base[:3, 3] = tvec_base.ravel()

    # 엔드 태그 후보
    for tag_id in [3,7,1,0,4,8]:
        if tag_id in left_pose:
            end_pose = left_pose[tag_id]
            break
        elif tag_id in right_pose:
            end_pose = right_pose[tag_id]
            break
    else:
        return

    rvec_end, tvec_end = end_pose
    Re, _ = cv2.Rodrigues(rvec_end)
    T_end = np.eye(4)
    T_end[:3,:3] = Re
    T_end[:3, 3] = tvec_end.ravel()

    # 현재 거리
    curr_dist = np.linalg.norm(T_base[:3,3] - T_end[:3,3])
    dist_threshold = target_distances.get(tag_id, 0.1)

    if curr_dist <= dist_threshold:
        print(f"[Info] 태그 {tag_id} 목표 범위 이내")
    else:
        # 완료 신호로 관절계산
        if complete_signal_event.wait(timeout=0.05):
            # 비동기로 관절값 계산
            pos_queue = queue.Queue(maxsize=1)
            optimizer_thread = JointAngleOptimizer(
                global_model, 
                current_joint_angles, 
                T_end[:3,3], 
                iterations=100, # 반복수 조절
                lr=0.05,        # 학습률 조절
                result_queue=pos_queue
            )
            optimizer_thread.start()
            optimizer_thread.join()

            new_angles = pos_queue.get()
            # 각도 제한
            new_angles_clamped = clamp_angles(new_angles, joint_angle_limits)
            current_joint_angles = new_angles_clamped

            print("[Info] 새 관절 값:", current_joint_angles)
            send_joint_angles_to_rpi(current_joint_angles)
            complete_signal_event.clear()

def clamp_angles(angles, limits):
    clamped = []
    for angle, (lower, upper) in zip(angles, limits):
        clamped.append(max(min(angle, upper), lower))
    return clamped
########################################
# 화면 표시 + 컨트롤 루프
########################################
def display_results():
    result_queue = queue.Queue()

    while not stop_event.is_set():
        try:
            left_frame, left_pose = left_result_queue.popleft()
            right_frame, right_pose = right_result_queue.popleft()
        except IndexError:
            continue

        control_loop(left_pose, right_pose, result_queue)

        # 화면 표시
        left_disp = cv2.rotate(left_frame, cv2.ROTATE_90_CLOCKWISE)
        right_disp = cv2.rotate(right_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined = cv2.hconcat([left_disp, right_disp])
        cv2.imshow("AprilTag Pose (Left & Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

########################################
# 네트워크 리스너 (완료 신호 수신)
########################################
def complete_signal_listener():
    HOST = ''
    PORT = 6000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(6)
        print(f"[Info] 완료 신호 리스너 {PORT} 대기중...")
        while True:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    try:
                        msg = json.loads(data.decode('utf-8'))
                        if msg.get("status") == "complete":
                            complete_signal_event.set()
                    except:
                        pass

########################################
# 전송 함수
########################################
def send_joint_angles_to_rpi(joint_angles):
    HOST = '192.168.0.153'
    PORT = 5000
    data = json.dumps([float(a) for a in joint_angles])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((HOST, PORT))
            s.sendall(data.encode('utf-8'))
            print("[Info] 관절 값 전송:", joint_angles)
    except Exception as e:
        print("[Error] 라즈베리파이 전송 실패:", e)

#캘리브레이션값 로드
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
# 메인 실행
########################################
def main():
    # 예) 전역 모델에 학습된 파라미터 로드 가능
    # global_model.load_state_dict(torch.load("model.pth"))

    # 스레드 생성
    t_left_cap = threading.Thread(target=capture_camera, args=(4, left_frame_queue, "Left"), daemon=True)
    t_right_cap = threading.Thread(target=capture_camera, args=(2, right_frame_queue, "Right"), daemon=True)
    t_left_proc = threading.Thread(target=process_frames, args=(left_frame_queue, left_result_queue, (l_camera_matrix, l_dist_coeffs), "Left"), daemon=True)
    t_right_proc = threading.Thread(target=process_frames, args=(right_frame_queue, right_result_queue, (r_camera_matrix, r_dist_coeffs), "Right"), daemon=True)
    t_signal = threading.Thread(target=complete_signal_listener, daemon=True)

    # 스레드 시작
    for t in [t_left_cap, t_right_cap, t_left_proc, t_right_proc, t_signal]:
        t.start()

    # 메인 루프
    try:
        display_results()
    except KeyboardInterrupt:
        stop_event.set()

    # 스레드 종료 대기
    for t in [t_left_cap, t_right_cap, t_left_proc, t_right_proc, t_signal]:
        t.join()

if __name__ == "__main__":
    main()
