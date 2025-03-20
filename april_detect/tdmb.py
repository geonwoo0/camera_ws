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
#from pupil_apriltags import Detector
import apriltag
from collections import deque
import datetime

########################################
# 관절 제한 (도 단위)
########################################

joint_angle_limits = [
    (0, 50),         # 채널 10: 고정 0도
    (100, 170),      # 채널 11
    (80, 225),      # 채널 12
    (55, 225),      # 채널 13
    (55, 225),      # 채널 14
    (90, 90),       # 채널 15: 고정 90도
    (30, 30)        # 채널 6: 고정 30도
]

#태그 별 목표 위치 거리
target_distances = {
    0: 0.15,
    3: 0.15,  # 15cm
    7: 0.15,
    1: 0.05,
    4: 0.07,  # 5cm
    8: 0.07
    # 추가 태그 필요시 추가 가능
}

tag_check_left = {0: [None, None], 3: [None, None], 7: [None, None], 1: [None, None], 4: [None, None], 8: [None, None]}
tag_check_right = {0: [None, None], 3: [None, None], 7: [None, None], 1: [None, None], 4: [None, None], 8: [None, None]}


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
# 2. Forward Kinematics 함수 (단순 예제)
########################################

def T_xyz(tx, ty, tz):
    """3D Translation Matrix"""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def Rz(theta):
    """Rotation about z-axis"""
    return np.array([
        [math.cos(theta), -math.sin(theta), 0, 0],
        [math.sin(theta),  math.cos(theta), 0, 0],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ], dtype=np.float32)

def Rx(theta):
    """Rotation about x-axis"""
    return np.array([
        [1, 0,                0,               0],
        [0, math.cos(theta), -math.sin(theta), 0],
        [0, math.sin(theta),  math.cos(theta), 0],
        [0, 0,                0,               1]
    ], dtype=np.float32)

def simple_fk(joint_angles_rad):
    """
    Forward Kinematics 함수.
    입력: 6개의 관절 각도 [θ0, θ1, θ2, θ3, θ4, θ5] (단위: rad)
    출력: base_link로부터 end_link까지의 4x4 변환 행렬 (위치와 회전)
    """
    # 링크 높이 (URDF에서 정의된 값)
    L0 = 0.089   # height0
    L1 = 0.06    # height1
    L2 = 0.096   # height2
    L3 = 0.06    # height3
    L4 = 0.094   # height4
    L5 = 0.095   # height5
    L6 = 0.1033  # height6
    
    # 관절 입력 (6개 관절)
    theta0, theta1, theta2, theta3, theta4, theta5 = joint_angles_rad

    # joint2와 joint5에 적용되는 x축 오프셋 (m)
    offset_x = 0.02  
    # joint6의 y축 오프셋 (m)
    offset_y_joint6 = -0.069

    # 각 관절의 변환 행렬 계산
    # joint0: translation along z by L0 then rotation about z (θ0)
    T0 = np.dot(T_xyz(0, 0, L0), Rz(theta0))
    # joint1: translation along z by L1 then rotation about x (–θ1)
    T1 = np.dot(T_xyz(0, 0, L1), Rx(-theta1))
    # joint2: translation (offset_x, 0, L2) then rotation about z (θ2)
    T2 = np.dot(T_xyz(offset_x, 0, L2), Rz(theta2))
    # joint3: translation along z by L3 then rotation about x (–θ3)
    T3 = np.dot(T_xyz(0, 0, L3), Rx(-theta3))
    # joint4: translation along z by L4 then rotation about x (–θ4)
    T4 = np.dot(T_xyz(0, 0, L4), Rx(-theta4))
    # joint5: translation (offset_x, 0, L5) then rotation about z (θ5)
    T5 = np.dot(T_xyz(offset_x, 0, L5), Rz(theta5))
    # joint6 (fixed): translation (0, offset_y_joint6, L6)
    T6 = T_xyz(0, offset_y_joint6, L6)

    # 최종 변환 행렬: T_total = T0 * T1 * T2 * T3 * T4 * T5 * T6
    T_total = T0.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    return T_total

########################################
# 3. 학습 데이터 생성 함수 (오프라인 학습용)
########################################

def generate_training_data(num_samples=5000):
    joint_limits_deg = joint_angle_limits.copy()
    X, Y = [], []
    for _ in range(num_samples):
        joint_angles_deg = []
        for lower, upper in joint_limits_deg:
            if lower == upper:
                joint_angles_deg.append(lower)
            else:
                joint_angles_deg.append(np.random.uniform(lower, upper))
        joint_angles_deg = np.array(joint_angles_deg)
        X.append(joint_angles_deg)
        joint_angles_rad = np.radians(joint_angles_deg[:6])
        fk_result = simple_fk(joint_angles_rad)
        bias = np.array([0.02, -0.01, 0.03])
        noise = np.random.normal(0, 0.005, fk_result.shape)
        sensor_value = fk_result + bias + noise
        Y.append(sensor_value)
    return np.array(X), np.array(Y)

########################################
# 4. 모델 학습 함수 (오프라인 학습용)
########################################

def train_direct_net(model, X_train, Y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            #print(f"DirectNet Epoch {epoch}, Loss: {loss.item():.6f}")
            pass
    return model

########################################
# 5. DirectNet 기반 최적화 함수 (제어용)
########################################

class JointAngleOptimizer(threading.Thread):
    def __init__(self, model, initial_angles, target_position, iterations=200, lr=0.01, result_queue=None):
        super(JointAngleOptimizer, self).__init__()
        self.model = model
        self.initial_angles = initial_angles
        self.target_position = target_position
        self.iterations = iterations
        self.lr = lr
        self.result_queue = result_queue  # 결과를 전달할 큐 추가

    def run(self):
        optimized_angles = optimize_joint_angles_direct(self.model, self.initial_angles, self.target_position, 
                                                        iterations=self.iterations, lr=self.lr)
        if self.result_queue is not None:
            self.result_queue.put(optimized_angles)  # 계산 완료 후 큐에 결과 저장

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
# 6. AprilTag 검출 관련 함수
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

left_frame_queue = deque(maxlen=1)  # 최신 데이터만 유지
right_frame_queue = deque(maxlen=1)  # 최신 데이터만 유지
left_result_queue = deque(maxlen=1)  # 최신 데이터만 유지
right_result_queue = deque(maxlen=1)  # 최신 데이터만 유지
stop_event = threading.Event()

current_joint_angles = [0, 135, 135, 120, 120, 90, 90]

########################################
# 7. 카메라 캡처 및 처리 함수 (AprilTag)
########################################

def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"현재 적용된 코덱: {chr(fourcc & 0xFF)}{chr((fourcc >> 8) & 0xFF)}{chr((fourcc >> 16) & 0xFF)}{chr((fourcc >> 24) & 0xFF)}")
    print("카메라 해상도:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),"/",cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print(f"{cam_label} 카메라 열기 실패")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.append(frame)  # deque에 최신 프레임 추가
            except IndexError:
                pass  # 큐가 가득 차면 가장 오래된 데이터가 자동으로 제거됨
        else:
            print(f"{cam_label} 프레임 읽기 실패")
    cap.release()

def process_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    """detector = Detector(
                        families='tag36h11',
                        nthreads=2,             # 사용 스레드 수 (기본값=1, 멀티코어 활용 가능)
                        quad_decimate=1.0,      # 이미지 해상도 감소 비율 (기본값=1.0, 낮을수록 정확성 증가)
                        quad_sigma=0.0,         # 이미지 블러(smoothing) 정도 (기본값=0.0)
                        refine_edges=1,         # 엣지 미세조정 여부 (기본값=1, 활성화 권장)
                        decode_sharpening=0.25, # 디코딩 향상을 위한 샤프닝 정도 (기본값=0.25)
                    )"""
                    
    options = apriltag.DetectorOptions(
    families="tag36h11",      # 사용할 태그 패밀리 (예: 'tag36h11', 'tag25h9' 등)
    nthreads=2,               # 사용 스레드 수
    quad_decimate=1.0,        # 입력 이미지 해상도 축소 비율 (1.0이면 원본 크기)
    refine_edges=True,        # 엣지 재조정 활성화 여부 (True/False)
)

    detector = apriltag.Detector(options)
    camera_matrix, dist_coeffs = calibration_data
    while not stop_event.is_set():
        try:
            # 큐가 비어 있지 않으면 frame을 popleft하여 가져옵니다.
            if frame_queue:
                frame = frame_queue.popleft()
            else:
                continue  # 큐가 비어 있으면 건너뜁니다

        except IndexError:
            continue  # 큐에서 꺼낼 때 오류가 나면 계속 대기
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        pose_dict = {}
        for tag in tags:
            if tag.tag_id >= 400:
                tag_size = 0.05
            elif tag.tag_id >= 0:
                tag_size = 0.018
            else:
                continue
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
                    print(f"[WARNING] {cam_label} - solvePnP failed for Tag {tag.tag_id}")
                else:
                    pose_dict[tag.tag_id] = (rvec, tvec)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size/2)
                    # 태그의 위치(tvec)를 업데이트하는 코드 추가
                    # tag_check는 미리 전역에서 아래와 같이 초기화해둡니다.
                    # tag_check = {0: [None, None], 3: [None, None], 7: [None, None],
                    #              1: [None, None], 4: [None, None], 8: [None, None]}
                    if cam_label == "Left":
                        tag_check= tag_check_left
                    else:
                        tag_check = tag_check_right
                    if tag.tag_id in tag_check:
                        # 현재 tvec를 튜플로 변환 (예: (x, y, z))
                        new_pos = tuple(tvec.flatten())
                        prev, curr = tag_check[tag.tag_id]
                        c_pose = [None,None,None]
                        n_pose = [None,None,None]
                        
                        if curr is not None:
                            # 전체 위치 변화(mm) 계산 (필요한 경우 사용)
                            diff = np.linalg.norm(np.array(new_pos) - np.array(curr)) * 1000

                            # 각 좌표를 mm로 변환
                            for i in range(3):
                                n_pose[i] = int(new_pos[i] * 1000)
                                c_pose[i] = int(curr[i] * 1000)

                            # 각 좌표별 변화가 5mm 이상인 경우 출력 (한 좌표라도 5mm 이상이면)
                            if any(n_pose[i] - c_pose[i] >= 5 for i in range(3)):
                                print(f"[{datetime.datetime.now().strftime('%S.%f')}] [INFO]{cam_label} Tag {tag.tag_id} 이전위치: x: {c_pose[0]} mm, y: {c_pose[1]} mm, z: {c_pose[2]} mm")
                                print(f"[{datetime.datetime.now().strftime('%S.%f')}] [INFO]{cam_label} Tag {tag.tag_id} 현재위치: x: {n_pose[0]} mm, y: {n_pose[1]} mm, z: {n_pose[2]} mm")

                           # if diff < 5:  # 0.005 m = 0.5 cm
                                #print(f"[INFO] Tag {tag.tag_id} 위치 변화가 0.5cm 이하입니다: {diff:.4f} m")
                            #    pass
                            #else:
                                #print("[",datetime.datetime.now().strftime('%S.%f'),"]",f"[INFO]{cam_label} Tag {tag.tag_id} 위치 변화: {diff:.2f} mm")
                            #    print("[",datetime.datetime.now().strftime('%S.%f'),"]",f"[INFO]{cam_label} Tag {tag.tag_id} 현재위치: x: {new_pos[0]*1000} y: {new_pos[1]*1000} z: {new_pos[2]*1000} mm")

                        # 이전 위치는 기존의 현재 위치로 업데이트, 현재 위치는 새 측정값으로 갱신
                        tag_check[tag.tag_id] = [curr, new_pos]
                    else:
                        tag_check[tag.tag_id] = [None, tuple(tvec.flatten())]
            except cv2.error as e:
                print(f"[ERROR] {cam_label} - Tag {tag.tag_id} solvePnP error: {e}")
        try:
            result_queue.append((frame, pose_dict))  # deque에 데이터를 추가
        except IndexError:
            pass  # 큐가 가득 차면 가장 오래된 데이터가 자동으로 제거됨

def is_valid_corners(corners, min_distance=10):
    if corners.shape != (4, 2):
        return False
    for i in range(4):
        for j in range(i+1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < min_distance:
                return False
    return True

########################################
# 8. 피드백 제어 및 관절 값 전송 (AprilTag 결과를 실제 센서값으로 사용)
########################################

target_distance = 0.10  # 목표 거리: 20cm

def update_joint_angles_with_learning(T_501, T_3):
    # T_501과 T_3을 이용해 새로운 관절 각도를 계산하는 로직
    measured_position = T_3[:3, 3]
    print("센서(카메라) 측정값:", measured_position)
    current_vector = T_501[:3, 3] - T_3[:3, 3]
    current_distance = np.linalg.norm(current_vector)
    if current_distance == 0:
        desired_candidate_position = T_3[:3, 3]
    else:
        desired_candidate_position = T_501[:3, 3] - (current_vector / current_distance) * target_distance
    print("Desired candidate position:", desired_candidate_position)
    print("현재 후보 태그 위치:", T_3[:3, 3])
    direct_model = DirectNet()
    
    new_angles = optimize_joint_angles_direct(direct_model, current_joint_angles, desired_candidate_position, iterations=200, lr=0.15)
    new_angles_clamped = clamp_angles(new_angles, joint_angle_limits)
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
            #print("[",datetime.datetime.now().strftime('%S.%f'),"]","Pi로 관절 값 전송", flush=True)
    except Exception as e:
        print("Raspberry Pi 전송 실패:", e)

def verify_end_effector_position(pose_dict, base_tag=501, end_tag=3, desired_distance=target_distance):
    if base_tag not in pose_dict or end_tag not in pose_dict:
        #print("[Verification] 필요한 태그 모두 검출되지 않음.")
        return None
    rvec_base, tvec_base = pose_dict[base_tag]
    R_base, _ = cv2.Rodrigues(rvec_base)
    T_base = np.eye(4)
    T_base[:3, :3] = R_base
    T_base[:3, 3] = tvec_base.ravel()
    rvec_end, tvec_end = pose_dict[end_tag]
    R_end, _ = cv2.Rodrigues(rvec_end)
    T_end = np.eye(4)
    T_end[:3, :3] = R_end
    T_end[:3, 3] = tvec_end.ravel()
    T_rel = np.linalg.inv(T_base) @ T_end
    measured_distance = np.linalg.norm(T_rel[:3, 3])
    error = abs(measured_distance - desired_distance)
    #print(f"[Verification] 측정 거리: {measured_distance:.3f} m, 목표 거리: {desired_distance:.3f} m, 오차: {error:.6f} m")
    return error

def update_tag_positions(tag_check, new_positions):
    """
    new_positions: 각 태그 id에 대한 현재 측정 위치를 담은 딕셔너리 {tag_id: (x, y, z)}
    tag_check: 기존 딕셔너리, 각 태그에 대해 [이전 위치, 현재 위치] 저장
    """
    for tag_id, current_pos in new_positions.items():
        if tag_id in tag_check:
            prev, curr = tag_check[tag_id]
            # 이전 위치를 현재 위치로 업데이트하고, 새 위치를 현재 위치로 저장
            tag_check[tag_id] = [curr, current_pos]


def control_loop(left_pose, right_pose, result_queue):
    global current_joint_angles
    base_tag_id = 501
    base_pose = None

    # 기준 태그(501번) 먼저 찾기
    if base_tag_id in left_pose or base_tag_id in right_pose:
        # 기준 태그가 있는 쪽은 우선 결정 (왼쪽 우선, 없으면 오른쪽)
        if base_tag_id in left_pose:
            base_pose = left_pose[base_tag_id]
            cam_used = "Left"
        else:
            base_pose = right_pose[base_tag_id]
            cam_used = "Right"
        
        # 후보 태그: 양쪽 카메라에서 모두 찾음
        candidate_tags = {}
        for tag_id in [3, 7, 1, 0, 4, 8]:
            if tag_id in left_pose:
                candidate_tags[tag_id] = left_pose[tag_id]
            if tag_id in right_pose:
                candidate_tags[tag_id] = right_pose[tag_id]
        
        if not candidate_tags:
            #print("[Control] 후보 태그 (0,3,7,1,4,8) 미검출")
            return
        
        # 우선순위에 따라 후보 태그 선택 (양쪽 정보를 결합한 candidate_tags 딕셔너리에서 선택)
        selected_tag_id = None
        for tag_id in [3, 7, 1, 0, 4, 8]:
            if tag_id in candidate_tags:
                selected_tag_id = tag_id
                break
        if selected_tag_id is None:
            print("[Control] 후보 태그 선택 실패")
            return

        rvec_end, tvec_end = candidate_tags[selected_tag_id]
        # 이후 처리...
    else:
        #print("[Control] 기준 태그 501 미검출")
        return

    rvec_base, tvec_base = base_pose
    R_base, _ = cv2.Rodrigues(rvec_base)
    T_base = np.eye(4)
    T_base[:3, :3] = R_base
    T_base[:3, 3] = tvec_base.ravel()

    R_end, _ = cv2.Rodrigues(rvec_end)
    T_end = np.eye(4)
    T_end[:3, :3] = R_end
    T_end[:3, 3] = tvec_end.ravel()

    current_distance = np.linalg.norm(T_base[:3, 3] - T_end[:3, 3])

    # 태그별 목표 거리
    target_distance = target_distances.get(tag_id, 0.1)  # 기본 10cm

    if current_distance <= target_distance:
        print(f"[성공] 태그 {tag_id} 목표 거리 이내 도달!")
    else:
        # 관절값 계산을 스레드에서 처리
        if complete_signal_event.wait(timeout=0.005):
            #print("[",datetime.datetime.now().strftime('%S.%f'),"]","관절값 계산 시작", flush=True)
            optimizer_thread = JointAngleOptimizer(DirectNet(), current_joint_angles, T_end[:3, 3], iterations=100, lr=0.1, result_queue=result_queue)
            optimizer_thread.start()  # 별도의 스레드에서 관절값 계산
            optimizer_thread.join()  # 계산 완료까지 기다림
            new_angles = result_queue.get()  # 큐에서 계산된 값 가져오기
            new_angles_clamped = clamp_angles(new_angles, joint_angle_limits)
            current_joint_angles = new_angles_clamped
            print("새 관절 값:", current_joint_angles)
            send_joint_angles_to_rpi(current_joint_angles)
            complete_signal_event.clear()


    verify_end_effector_position(
        left_pose if cam_used == "Left" else right_pose,
        base_tag=base_tag_id,
        end_tag=tag_id,
        desired_distance=target_distance
    )


def display_results():
    cv2.namedWindow("AprilTag Pose (Left & Right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Pose (Left & Right)", 1080, 720)
    result_queue = queue.Queue()  # 관절값 결과를 받기 위한 큐 생성
    while not stop_event.is_set():
        try:
            left_frame, left_pose = left_result_queue.popleft()
            right_frame, right_pose = right_result_queue.popleft()
            #print("[",datetime.datetime.now().strftime('%S.%f'),"]","태그 감지", flush=True)
        except IndexError:
            continue
        #print("[",datetime.datetime.now().strftime('%S.%f'),"]","관절, 태그 거리 계산시작", flush=True)
        control_loop(left_pose, right_pose, result_queue)
        #print("[",datetime.datetime.now().strftime('%S.%f'),"]","계산완료", flush=True)
        cv2.resize(left_frame,(1280,720),interpolation=cv2.INTER_NEAREST)
        cv2.resize(right_frame,(1280,720),interpolation=cv2.INTER_NEAREST)
        left_disp = cv2.rotate(left_frame, cv2.ROTATE_90_CLOCKWISE)
        right_disp = cv2.rotate(right_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined = cv2.hconcat([left_disp, right_disp])
        #print("[",datetime.datetime.now().strftime('%S.%f'),"]","이미지출력", flush=True)
        #print("-"*30)
        cv2.putText(combined, datetime.datetime.now().strftime('%S.%f'), (1400, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("AprilTag Pose (Left & Right)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

########################################
# 9. 완료 신호 수신용 네트워크 리스너 (포트 6000)
########################################

complete_signal_event = threading.Event()

def complete_signal_listener():
    HOST = ''  # 모든 인터페이스
    PORT = 6000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(6)
        print(f"[complete_signal_listener] 포트 {PORT}에서 완료 신호 대기중...")
        while True:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    try:
                        message = json.loads(data.decode('utf-8'))
                        if message.get("status") == "complete":
                            #print("[",datetime.datetime.now().strftime('%S.%f'),"]","[complete_signal_listener] 완료 신호 수신됨.")
                            complete_signal_event.set()
                    except Exception as e:
                        print("[complete_signal_listener] 메시지 파싱 오류:", e)

########################################
# 10. 스레드 생성 및 실행
########################################

CHANNELS = [10, 11, 12, 13, 14, 15, 6]
CHANNEL_ANGLES = {ch: limits for ch, limits in zip(CHANNELS, joint_angle_limits)}

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

print("실시간 AprilTag 위치 추정, 실제 카메라 기반 제어 (DirectNet) 및 관절 값 전송 중...\n'q'를 눌러 종료하세요.")
try:
    display_results()
except KeyboardInterrupt:
    stop_event.set()
for t in threads:
    t.join()
