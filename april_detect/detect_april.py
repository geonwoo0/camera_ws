import cv2
import numpy as np
import time
import threading
from collections import deque
#from pupil_apriltags import Detector
import apriltag
import datetime
import json
import socket

# --- 커스텀 예외 ---
class Empty(Exception):
    pass

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

# --- FastQueue: deque 기반의 커스텀 큐 ---
class FastQueue:
    def __init__(self, maxsize=0):
        self.deque = deque()
        self.lock = threading.Lock()
        self.maxsize = maxsize  # 0이면 제한 없음

    def put(self, item):
        with self.lock:
            if self.maxsize > 0 and len(self.deque) >= self.maxsize:
                # 최신 데이터를 유지하기 위해 기존 항목 모두 제거
                self.deque.clear()
            self.deque.append(item)

    def get(self, timeout=None):
        start_time = time.time()
        while True:
            with self.lock:
                if self.deque:
                    return self.deque.popleft()
            if timeout is not None and (time.time() - start_time) > timeout:
                raise Empty()
            time.sleep(0.0001)

    def empty(self):
        with self.lock:
            return len(self.deque) == 0

# --- 보정 데이터 로드 함수 ---
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
                           [0,   0,   1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)
l_camera_matrix, l_dist_coeffs = load_calibration_data("../camera_calibration/cam8_calibration_data.npz",
                                                        default_matrix, default_dist)
r_camera_matrix, r_dist_coeffs = load_calibration_data("../camera_calibration/cam9_calibration_data.npz",
                                                        default_matrix, default_dist)
default_tag_size = 0.02  # 기본 태그 크기 (미터)
timestamp = "[" + datetime.datetime.now().strftime('%S.%f') + "]"

# --- FastQueue 기반의 큐 및 종료 이벤트 ---
left_frame_queue = FastQueue(maxsize=1)
right_frame_queue = FastQueue(maxsize=1)
left_result_queue = FastQueue(maxsize=1)
right_result_queue = FastQueue(maxsize=1)
tag_check_left = {0: [None, None], 3: [None, None], 7: [None, None], 1: [None, None], 4: [None, None], 8: [None, None]}
tag_check_right = {0: [None, None], 3: [None, None], 7: [None, None], 1: [None, None], 4: [None, None], 8: [None, None]}

def put_single(q, item):
    # 큐에 항목이 있으면 모두 제거 후 새 항목 추가
    while not q.empty():
        try:
            q.get(timeout=0.0001)
        except Empty:
            break
    q.put(item)

stop_event = threading.Event()

# --- 카메라 캡처 함수 ---
def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(timestamp, "width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "height :", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(timestamp, "fps :", cap.get(cv2.CAP_PROP_FPS))
    
    if not cap.isOpened():
        print(f"{cam_label} 카메라 열기 실패")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            put_single(frame_queue, (frame, datetime.datetime.now()))
        else:
            print(f"{cam_label} 프레임 읽기 실패")
    cap.release()
    
# --- 코너 데이터 검증 함수 ---
def is_valid_corners(corners, min_distance=10):
    if corners.shape != (4, 2):
        return False
    for i in range(4):
        for j in range(i+1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < min_distance:
                return False
    return True

def update_tag_positions(tag_check, new_positions):
    for tag_id, current_pos in new_positions.items():
        if tag_id in tag_check:
            prev, curr = tag_check[tag_id]
            tag_check[tag_id] = [curr, current_pos]

# --- 프레임 처리 함수 ---
def process_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    """detector = Detector(
                        families='tag36h11',
                        nthreads=2,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0
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
            frame, frame_timestamp = frame_queue.get(timeout=0.0001)
        except Empty:
            continue
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
                [-tag_size / 2, -tag_size / 2, 0],
                [ tag_size / 2, -tag_size / 2, 0],
                [ tag_size / 2,  tag_size / 2, 0],
                [-tag_size / 2,  tag_size / 2, 0]
            ], dtype=np.float32)
            image_points = tag.corners.astype(np.float32)
            if not is_valid_corners(image_points):
                print(f"[WARNING] {cam_label} - Tag {tag.tag_id} 코너 데이터가 유효하지 않음: {image_points}")
                continue
            try:
                success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                                    camera_matrix, dist_coeffs,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    print(f"[WARNING] {cam_label} - solvePnP failed for Tag {tag.tag_id}")
                else:
                    pose_dict[tag.tag_id] = (rvec, tvec)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size / 2)
                    
                    if cam_label == "Left":
                        tag_check = tag_check_left
                    else:
                        tag_check = tag_check_right
                    text = f"{tag.tag_id} x: {tvec[0][0]:.2f}, y: {tvec[1][0]:.2f}, z: {tvec[2][0]:.2f}"
                    pos = (int(image_points[0][0]), int(image_points[0][1]))

                    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)    
                    

                    new_pos = tuple(tvec.flatten())
                    if tag.tag_id in tag_check:
                        prev, curr = tag_check[tag.tag_id]
                        c_pose = [None, None, None]
                        n_pose = [None, None, None]
                        if curr is not None:
                            for i in range(3):
                                c_pose[i] = int(curr[i] * 1000)
                                n_pose[i] = int(new_pos[i] * 1000)
                            if any(n_pose[i] - c_pose[i] >= 5 for i in range(3)):
                                send_pose_to_rpi([tag.tag_id, n_pose[0], n_pose[1], n_pose[2]])
                        tag_check[tag.tag_id] = [curr, new_pos]
                    else:
                        tag_check[tag.tag_id] = [None, new_pos]
            except cv2.error as e:
                print(f"[ERROR] {cam_label} - Tag {tag.tag_id} solvePnP error: {e}")
                put_single(result_queue, (frame, pose_dict, frame_timestamp))
            except cv2.error as e:
                print(f"[ERROR] {cam_label} - Tag {tag.tag_id} solvePnP error: {e}")
        try:
            put_single(result_queue, (frame, pose_dict, frame_timestamp))
        except Exception:
            pass

# --- 상대 거리 계산 함수 ---
def compute_relative_distances(pose_dict, base_tag=501):
    rel_distances = {}
    if base_tag not in pose_dict:
        return rel_distances
    rvec_base, tvec_base = pose_dict[base_tag]
    R_base, _ = cv2.Rodrigues(rvec_base)
    T_cam_base = np.eye(4)
    T_cam_base[:3, :3] = R_base
    T_cam_base[:3, 3] = tvec_base.ravel()
    T_base_cam = np.linalg.inv(T_cam_base)
    for tag_id, (rvec, tvec) in pose_dict.items():
        if tag_id == base_tag:
            continue
        R, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R
        T_cam_tag[:3, 3] = tvec.ravel()
        T_base_tag = T_base_cam @ T_cam_tag
        distance = np.linalg.norm(T_base_tag[:3, 3])
        rel_distances[tag_id] = distance
    return rel_distances

def send_pose_to_rpi(n_pos):
    HOST = '192.168.50.4'
    PORT = 5000    
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((HOST, PORT))
            n_pos = [int(x) for x in n_pos]
            message = {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                "pose": n_pos
            }    
            data = json.dumps(message)
            s.sendall(data.encode('utf-8'))
    except Exception as e:
        print("Raspberry Pi 전송 실패:", e)

current_frame_ts = ""
def complete_signal_listener():
    global current_frame_ts
    HOST = ''  # 모든 인터페이스
    PORT = 6000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(6)
        while True:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    try:
                        message = json.loads(data.decode('utf-8'))
                        received_timestamp_str = message["timestamp"]
                        received_timestamp = datetime.datetime.strptime(received_timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        time_diff = datetime.datetime.now() - received_timestamp
                        ms = time_diff.total_seconds() * 1000
                        if message.get("status") == "complete":
                            print(f"[지연시간 : {ms:.2f} ms]", "완료신호 입력")
                            print(f"현재 시간 : {datetime.datetime.now()} / 현재 프레임 시간 :", current_frame_ts)
                            frame_ms = (datetime.datetime.now() - current_frame_ts).total_seconds() * 1000
                            print(f"화면지연 : {frame_ms:.2f}ms")
                    except Exception as e:
                        print("[complete_signal_listener] 메시지 파싱 오류:", e)

# --- 메인 디스플레이 함수 ---
def display_results():
    global current_frame_ts
    cv2.namedWindow("AprilTag Pose (Left & Right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Pose (Left & Right)", 1080, 720)
    while not stop_event.is_set():
        try:
            left_frame, left_pose, left_ts = left_result_queue.get(timeout=0.0001)
            right_frame, right_pose, right_ts = right_result_queue.get(timeout=0.0001)
            current_frame_ts = max(left_ts, right_ts)
        except Empty:
            continue
        left_frame = cv2.rotate(left_frame, cv2.ROTATE_90_CLOCKWISE)
        right_frame = cv2.rotate(right_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combined = cv2.hconcat([left_frame, right_frame])
        cv2.imshow("AprilTag Pose (Left & Right)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 엔터키 입력 시
            current_timestamp = datetime.datetime.now()
            time_diff0 = (current_timestamp - current_frame_ts).total_seconds() * 1000
            print("카메라 프레임 타임스탬프와 현재 타임스탬프 차이: {:.3f} ms".format(time_diff0))
        elif key == 27:  # ESC 키: 종료
            break
    cv2.destroyAllWindows()

# --- 스레드 생성 ---
threads = []
t_complete_listener = threading.Thread(target=complete_signal_listener, daemon=True)
t_left_capture = threading.Thread(target=capture_camera, args=(0, left_frame_queue, "Left"), daemon=True)
t_right_capture = threading.Thread(target=capture_camera, args=(2, right_frame_queue, "Right"), daemon=True)
threads.extend([t_complete_listener, t_left_capture, t_right_capture])
t_left_process = threading.Thread(target=process_frames, args=(left_frame_queue, left_result_queue, (l_camera_matrix, l_dist_coeffs), "Left"), daemon=True)
t_right_process = threading.Thread(target=process_frames, args=(right_frame_queue, right_result_queue, (r_camera_matrix, r_dist_coeffs), "Right"), daemon=True)
threads.extend([t_left_process, t_right_process])

for t in threads:
    t.start()
print("실시간 AprilTag 위치 추정 및 501 기준 거리 계산 중... 'q'를 눌러 종료하세요.")

try:
    display_results()
except KeyboardInterrupt:
    stop_event.set()
for t in threads:
    t.join()
