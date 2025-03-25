import cv2
import numpy as np
import time
import threading
import queue
import os, sys
import apriltag

########################################
# 캘리브레이션 관련 설정
########################################

def load_calibration_data(path, default_matrix, default_dist):
    try:
        with np.load(path) as X:
            camera_matrix = X["camera_matrix"]
            dist_coeffs = X["distortion_coefficients"]
            print(f"[Calibration] {path} ->", camera_matrix)
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"{path} 캘리브레이션 데이터를 로드하지 못했습니다. 기본값 사용: {e}")
        return default_matrix, default_dist

# 기본 캘리브레이션 데이터 (실제 값으로 교체 필요)
default_matrix = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0, 0, 1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)

# 4대 카메라에 대한 캘리브레이션 파일 경로 (필요시 수정)
calibration_files = [
    "../camera_calibration/cam0_calibration_data.npz",
    "../camera_calibration/cam1_calibration_data.npz",
    "../camera_calibration/cam2_calibration_data.npz",
    "../camera_calibration/cam3_calibration_data.npz"
]

# 각 카메라의 캘리브레이션 데이터를 리스트에 저장
calibration_data_list = []
for path in calibration_files:
    cam_matrix, dist_coeffs = load_calibration_data(path, default_matrix, default_dist)
    calibration_data_list.append((cam_matrix, dist_coeffs))

########################################
# 카메라 인덱스 및 기타 설정
########################################

# 카메라 인덱스 (예시: 0,1,2,3)
CAMERA_INDICES = [0, 2,4,  8]

########################################
# AprilTag 검출 함수
########################################

def process_frame(frame, detector, camera_matrix, dist_coeffs):
    """
    입력 프레임에서 AprilTag를 검출하고, 각 태그의 rvec, tvec을 계산합니다.
    검출된 태그에 태그 id를 표시합니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    pose_dict = {}
    for tag in tags:
        # 태그 크기: tag_id가 200 이하이면 0.025m, 그 외는 0.05m
        tag_size = 0.025 if tag.tag_id <= 200 else 0.05
        object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float32)
        image_points = tag.corners.astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                           camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            pose_dict[tag.tag_id] = (rvec, tvec)
            cv2.putText(frame, str(tag.tag_id),
                        (int(tag.center[0]), int(tag.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"[Process] solvePnP 실패 - 태그 {tag.tag_id}")
    return frame, pose_dict

########################################
# 카메라 캡처 스레드 함수
########################################

def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    # 4:3 비율 해상도 예시 (원하는 해상도로 수정 가능)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[{cam_label}] 카메라 열림 (인덱스: {cam_index})")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_label}] 프레임 읽기 실패")
            continue
        # 큐가 꽉 찼으면 기존 프레임 제거
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

########################################
# 프레임 처리 스레드 함수
########################################

def process_camera_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    detector = apriltag.Detector()
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
# 메인 함수
########################################

def main():
    num_cameras = len(CAMERA_INDICES)
    # 각 카메라별 프레임과 결과를 위한 큐 생성
    frame_queues = [queue.Queue(maxsize=1) for _ in range(num_cameras)]
    result_queues = [queue.Queue(maxsize=1) for _ in range(num_cameras)]
    
    # 각 카메라 캡처 스레드 시작
    capture_threads = []
    for i, cam_index in enumerate(CAMERA_INDICES):
        t = threading.Thread(target=capture_camera,
                             args=(cam_index, frame_queues[i], f"Camera {cam_index}"),
                             daemon=True)
        t.start()
        capture_threads.append(t)
    
    # 각 카메라 처리 스레드 시작
    process_threads = []
    for i in range(num_cameras):
        t = threading.Thread(target=process_camera_frames,
                             args=(frame_queues[i], result_queues[i],
                                   calibration_data_list[i], f"Camera {CAMERA_INDICES[i]}"),
                             daemon=True)
        t.start()
        process_threads.append(t)
    
    print("4대의 카메라로 AprilTag 검출 시작 (ESC 키로 종료)")
    
    # 결과 출력: 각 카메라별 창으로 표시
    while True:
        for i in range(num_cameras):
            if not result_queues[i].empty():
                annotated_frame, pose_dict = result_queues[i].get()
                window_name = f"Camera {CAMERA_INDICES[i]}"
                cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
