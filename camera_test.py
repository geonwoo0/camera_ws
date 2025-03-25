import cv2
import numpy as np
import time
import threading
import queue
import datetime
import socket
import json
import os, sys

#devnull = os.open(os.devnull, os.O_WRONLY)
#os.dup2(devnull, sys.stderr.fileno())

import apriltag
from collections import deque

# 최근 N개 측정값을 저장 (예: N = 5)
N = 5
recent_measurements_left = deque(maxlen=N)
recent_measurements_right = deque(maxlen=N)
########################################
# 설정
########################################

def load_calibration_data(path, default_matrix, default_dist):
    try:
        with np.load(path) as X:
            camera_matrix = X["camera_matrix"]
            dist_coeffs = X["distortion_coefficients"]
            print("[Calibration]", camera_matrix)
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"{path} 캘리브레이션 데이터를 로드하지 못했습니다. 기본값 사용: {e}")
        return default_matrix, default_dist
    
# 기본 캘리브레이션 데이터 (실제 값으로 교체 필요)
default_matrix = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0, 0, 1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)

camera_matrix_left, dist_coeffs_left = load_calibration_data("../camera_calibration/cam8_calibration_data.npz", default_matrix, default_dist)
camera_matrix_right, dist_coeffs_right = load_calibration_data("../camera_calibration/cam9_calibration_data.npz", default_matrix, default_dist)

# 기준 태그와 대상 태그 리스트
BASE_TAG = 501
TARGET_TAGS = [11, 12, 13, 14, 15]

# 카메라 인덱스 (왼쪽, 오른쪽)
LEFT_CAM_INDEX = 0
RIGHT_CAM_INDEX = 2

########################################
# 태그 처리 함수
########################################

def update_measurements(measurement, measurements_deque):
    measurements_deque.append(measurement)
    # measurements_deque에 저장된 값들의 평균 계산 (numpy array로 반환)
    avg = np.mean(measurements_deque, axis=0)
    return avg

def process_frame(frame, detector, camera_matrix, dist_coeffs):
    """
    입력 프레임에서 AprilTag를 검출하여 각 태그의 (rvec, tvec)를 계산합니다.
    반환: (annotated_frame, pose_dict)
         pose_dict: { tag_id: (rvec, tvec), ... }
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    pose_dict = {}
    for tag in tags:
        # 태그 크기 결정: tag_id가 0 이상이면 0.025m, 그 외는 0.05m (필요에 따라 조정)
        tag_size = 0.025 if tag.tag_id <= 200 else 0.05
        object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float32)
        image_points = tag.corners.astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            pose_dict[tag.tag_id] = (rvec, tvec)
            cv2.putText(frame, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"[Process] solvePnP 실패 - 태그 {tag.tag_id}")
    return frame, pose_dict

########################################
# 기준 태그(501)를 기준으로 상대 좌표 계산 함수
########################################

def compute_relative_positions(pose_dict, base_tag=501, target_tags=[11,12,13,14,15]):
    """
    pose_dict: { tag_id: (rvec, tvec), ... }
    기준 태그(501번)의 좌표계를 기준으로 대상 태그들의 상대 좌표를 계산합니다.
    좌표 변환: t_relative = R_base^(-1) * (t_target - t_base)
    반환: { tag_id: [dx, dy, dz], ... }
    """
    relative_positions = {}
    if base_tag not in pose_dict:
        print(f"[Relative] 기준 태그 {base_tag} 미검출")
        return relative_positions

    # 기준 태그의 rvec, tvec을 가져와 회전행렬로 변환
    rvec_base, tvec_base = pose_dict[base_tag]
    R_base, _ = cv2.Rodrigues(rvec_base)
    tvec_base = np.asarray(tvec_base).reshape(3)

    for tag in target_tags:
        if tag in pose_dict:
            rvec_target, tvec_target = pose_dict[tag]
            tvec_target = np.asarray(tvec_target).reshape(3)
            # 501 태그 좌표계로 변환: R_base의 역행렬은 전치행렬
            rel_vec = R_base.T.dot(tvec_target - tvec_base)
            relative_positions[tag] = rel_vec
            print(f"[Relative] 태그 {base_tag} 기준 태그 {tag} 상대 좌표: x={rel_vec[0]:.4f}, y={rel_vec[1]:.4f}, z={rel_vec[2]:.4f} m")
        else:
            #print(f"[Relative] 태그 {tag} 미검출")
            pass
    return relative_positions

########################################
# 카메라 캡처 스레드
########################################

def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    cap.set(cv2.CAP_PROP_FPS, 15)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"현재 적용된 코덱: {chr(fourcc & 0xFF)}{chr((fourcc >> 8) & 0xFF)}{chr((fourcc >> 16) & 0xFF)}{chr((fourcc >> 24) & 0xFF)}")
    print("카메라 해상도:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),"/",cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print(f"[{cam_label}] 카메라 열기 실패")
        return
    print(f"[{cam_label}] 카메라 열림: 인덱스 {cam_index}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_label}] 프레임 읽기 실패")
            continue
        # 최신 프레임만 유지하기 위해 큐가 꽉 차 있으면 제거
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
        
    cap.release()

########################################
# 프레임 처리 스레드
########################################

def process_camera_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    options = apriltag.DetectorOptions(
                                        families="tag36h11",      # 사용할 태그 패밀리 (예: 'tag36h11', 'tag25h9' 등)
                                        nthreads=4,               # 사용 스레드 수
                                        quad_decimate=1.0,        # 입력 이미지 해상도 축소 비율 (1.0이면 원본 크기)
                                        refine_edges=True,        # 엣지 재조정 활성화 여부 (True/False)
    )
    detector = apriltag.Detector(options)
    camera_matrix, dist_coeffs = calibration_data
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        annotated_frame, pose_dict = process_frame(frame, detector, camera_matrix, dist_coeffs)
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put((annotated_frame, pose_dict))
        
########################################
# 소켓 전송 함수: 상대 좌표 전송
########################################

def send_relative_positions(relative_positions, host="192.168.50.4", port=5000):
    """
    상대 좌표 딕셔너리(relative_positions)를 JSON으로 인코딩하여 소켓을 통해 전송합니다.
    relative_positions: { tag_id: [dx, dy, dz], ... }
    """
    message = json.dumps(relative_positions)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.02)
            s.connect((host, port))
            s.sendall(message.encode('utf-8'))
            print(f"[Socket] 전송 성공: {message}")
    except Exception as e:
        #print(f"[Socket] 전송 실패: {e}")
        pass

########################################
# 메인 함수
########################################

def main():
    cv2.namedWindow("Left & Right AprilTag Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left & Right AprilTag Detection", 2048, 768)
    left_frame_queue = queue.Queue(maxsize=1)
    right_frame_queue = queue.Queue(maxsize=1)
    left_result_queue = queue.Queue(maxsize=1)
    right_result_queue = queue.Queue(maxsize=1)

    calibration_left = (camera_matrix_left, dist_coeffs_left)
    calibration_right = (camera_matrix_right, dist_coeffs_right)

    # 카메라 캡처 스레드 시작
    t_left = threading.Thread(target=capture_camera, args=(LEFT_CAM_INDEX, left_frame_queue, "Left"), daemon=True)
    t_right = threading.Thread(target=capture_camera, args=(RIGHT_CAM_INDEX, right_frame_queue, "Right"), daemon=True)
    t_left.start()
    t_right.start()

    # 프레임 처리 스레드 시작
    t_left_proc = threading.Thread(target=process_camera_frames, args=(left_frame_queue, left_result_queue, calibration_left, "Left"), daemon=True)
    t_right_proc = threading.Thread(target=process_camera_frames, args=(right_frame_queue, right_result_queue, calibration_right, "Right"), daemon=True)
    t_left_proc.start()
    t_right_proc.start()

    print("두 카메라를 이용한 AprilTag 검출 및 501 태그 기준 상대 좌표 계산 시작 (ESC 키로 종료)")
    while True:
        start_time = time.time()  # 프레임 처리 시작 시각
        if not left_result_queue.empty() and not right_result_queue.empty():
            left_frame, left_pose_dict = left_result_queue.get()
            right_frame, right_pose_dict = right_result_queue.get()

            # 왼쪽 카메라 501 태그 처리: 최근 측정값 이동 평균 적용
            if BASE_TAG in left_pose_dict:
                rvec, tvec = left_pose_dict[BASE_TAG]
                measurement_left = np.array(tvec, dtype=np.float32).reshape(3)
                avg_left = update_measurements(measurement_left, recent_measurements_left)
                left_pose_dict[BASE_TAG] = (rvec, avg_left)
            else:
                if recent_measurements_left:
                    avg_left = np.mean(recent_measurements_left, axis=0)
                else:
                    avg_left = np.zeros(3, dtype=np.float32)
                left_pose_dict[BASE_TAG] = (np.zeros((3, 1), dtype=np.float32), avg_left)

            # 오른쪽 카메라 501 태그 처리: 최근 측정값 이동 평균 적용
            if BASE_TAG in right_pose_dict:
                rvec, tvec = right_pose_dict[BASE_TAG]
                measurement_right = np.array(tvec, dtype=np.float32).reshape(3)
                avg_right = update_measurements(measurement_right, recent_measurements_right)
                right_pose_dict[BASE_TAG] = (rvec, avg_right)
            else:
                if recent_measurements_right:
                    avg_right = np.mean(recent_measurements_right, axis=0)
                else:
                    avg_right = np.zeros(3, dtype=np.float32)
                right_pose_dict[BASE_TAG] = (np.zeros((3, 1), dtype=np.float32), avg_right)

            # 기준 태그(501)를 기준으로 상대 좌표 계산
            print("left")
            rel_left = compute_relative_positions(left_pose_dict, base_tag=BASE_TAG, target_tags=TARGET_TAGS)
            print("right")
            rel_right = compute_relative_positions(right_pose_dict, base_tag=BASE_TAG, target_tags=TARGET_TAGS)

            combined_rel = {
                "left": {str(tag): vec.tolist() for tag, vec in rel_left.items()},
                "right": {str(tag): vec.tolist() for tag, vec in rel_right.items()}
            }

            send_relative_positions(combined_rel, host="192.168.50.4", port=5000)

            # 두 카메라 영상 좌우 결합하여 표시 (원본 해상도 그대로 결합)
            combined = cv2.hconcat([left_frame, right_frame])
            end_time = time.time()  # 프레임 처리 끝 시각
            elapsed_ms = (end_time - start_time) * 1000
            print(f"Frame update time: {elapsed_ms:.2f} ms")
            cv2.imshow("Left & Right AprilTag Detection", combined)
            
        
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            break



    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
