import cv2
import numpy as np
import time
import datetime
import cv2.aruco as aruco
from collections import deque
import threading

########################################
# 기본 설정 및 캘리브레이션 데이터 로드
########################################

N = 5  # 최근 N개 측정값 저장
recent_measurements_left = deque(maxlen=N)

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

default_matrix = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0,   0,   1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)
camera_matrix_left, dist_coeffs_left = load_calibration_data(
    "../camera_calibration/cam8_calibration_data.npz",
    default_matrix,
    default_dist
)

default_marker_length = 0.025  # 사용하지 않음

########################################
# ArUco 설정 및 함수 정의
########################################

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()  # 생성자 호출
detector_obj = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def estimate_pose_single_marker(corners, marker_length, camera_matrix, dist_coeffs):
    object_points = np.array([
         [-marker_length/2,  marker_length/2, 0],
         [ marker_length/2,  marker_length/2, 0],
         [ marker_length/2, -marker_length/2, 0],
         [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)
    image_points = corners.reshape(-1, 2).astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                       camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        print("solvePnP 실패")
    return rvec, tvec

def update_measurements(measurement, measurements_deque):
    measurements_deque.append(measurement)
    avg = np.mean(measurements_deque, axis=0)
    return avg

def detect_markers_low_res(frame_high, scale_factor=0.5):
    t0 = time.time()
    frame_low = cv2.resize(frame_high, None, fx=scale_factor, fy=scale_factor)
    t1 = time.time()
    gray_low = cv2.cvtColor(frame_low, cv2.COLOR_BGR2GRAY)
    t2 = time.time()
    corners, ids, rejected = detector_obj.detectMarkers(gray_low)
    t3 = time.time()
    print(f"[LowRes] 리사이즈: {(t1-t0)*1000:.2f} ms, 그레이스케일: {(t2-t1)*1000:.2f} ms, 검출: {(t3-t2)*1000:.2f} ms")
    return (corners, ids), scale_factor

def get_roi_from_low_res_markers(detection_result, scale_factor, margin=20):
    t0 = time.time()
    corners, ids = detection_result
    if corners is None or len(corners) == 0:
        return None
    all_corners = np.concatenate([c.reshape(-1, 2) for c in corners])
    x_min = np.min(all_corners[:, 0])
    y_min = np.min(all_corners[:, 1])
    x_max = np.max(all_corners[:, 0])
    y_max = np.max(all_corners[:, 1])
    x_min_hr = int(x_min / scale_factor) - margin
    y_min_hr = int(y_min / scale_factor) - margin
    x_max_hr = int(x_max / scale_factor) + margin
    y_max_hr = int(y_max / scale_factor) + margin
    roi = (max(0, x_min_hr), max(0, y_min_hr),
           x_max_hr - x_min_hr, y_max_hr - y_min_hr)
    t1 = time.time()
    print(f"[ROI] 계산 시간: {(t1-t0)*1000:.2f} ms")
    return roi

def process_frame_high_res_with_roi(frame_high, roi):
    t0 = time.time()
    if roi is not None:
        x, y, w, h = roi
        frame_roi = frame_high[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = cv2.cvtColor(frame_high, cv2.COLOR_BGR2GRAY)
    t1 = time.time()
    
    corners, ids, rejected = detector_obj.detectMarkers(gray_roi)
    t2 = time.time()
    
    pose_dict = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            ml = 0.5 if marker_id == 1 else 0.25
            rvec, tvec = estimate_pose_single_marker(corners[i], ml, camera_matrix_left, dist_coeffs_left)
            center = np.mean(corners[i].reshape(-1, 2), axis=0)
            if roi is not None:
                center += np.array([x, y])
            cv2.drawFrameAxes(frame_high, camera_matrix_left, dist_coeffs_left, rvec, tvec, 0.03)
            cv2.putText(frame_high, str(marker_id), (int(center[0]), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            pose_dict[marker_id] = (rvec, tvec)
    t3 = time.time()
    print(f"[HighRes] ROI 적용 및 그레이스케일: {(t1-t0)*1000:.2f} ms, 검출: {(t2-t1)*1000:.2f} ms, Pose 추정: {(t3-t2)*1000:.2f} ms")
    return frame_high, pose_dict

def process_frame_with_dual_resolution(frame_high, scale_factor=0.5, roi=None):
    t0 = time.time()
    if roi is None:
        detection_result, scale = detect_markers_low_res(frame_high, scale_factor)
        roi = get_roi_from_low_res_markers(detection_result, scale)
        if roi:
            print(f"[ROI] 업데이트된 영역: {roi}")
        else:
            print("[ROI] 마커 미검출 - 전체 영역 사용")
    t1 = time.time()
    annotated_frame, pose_dict = process_frame_high_res_with_roi(frame_high, roi)
    t2 = time.time()
    print(f"[Dual] ROI 업데이트 및 HighRes 처리: {(t2-t1)*1000:.2f} ms")
    return (annotated_frame, pose_dict), roi

def compute_relative_positions(pose_dict, base_tag=1, target_tags=[11, 12, 13, 14, 15]):
    t0 = time.time()
    relative_positions = {}
    if base_tag not in pose_dict:
        #print(f"[Relative] 기준 마커 {base_tag} 미검출")
        return relative_positions
    rvec_base, tvec_base = pose_dict[base_tag]
    R_base, _ = cv2.Rodrigues(rvec_base)
    tvec_base = np.asarray(tvec_base).reshape(3)
    for tag in target_tags:
        if tag in pose_dict:
            rvec_target, tvec_target = pose_dict[tag]
            tvec_target = np.asarray(tvec_target).reshape(3)
            rel_vec = R_base.T.dot(tvec_target - tvec_base)
            relative_positions[tag] = rel_vec
            print(f"[Relative] 마커 {base_tag} 기준 마커 {tag} 상대 좌표: x={rel_vec[0]:.4f}, y={rel_vec[1]:.4f}, z={rel_vec[2]:.4f} m")
    t1 = time.time()
    print(f"[Relative] 상대 좌표 계산 시간: {(t1-t0)*1000:.2f} ms")
    return relative_positions

########################################
# 각 카메라 처리 스레드 함수
########################################

def camera_thread(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"카메라 {cam_index} 열기 실패")
        return
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    cap.set(cv2.CAP_PROP_FPS, 15)
    print(f"카메라 {cam_index} 열림")
    
    frame_count = 0
    last_roi = None
    prev_frame_time = None
    
    while True:
        overall_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"카메라 {cam_index} 프레임 읽기 실패")
            continue
        
        capture_timestamp = time.time()
        frame_count += 1
        
        if frame_count % 15 == 0:
            (annotated_frame, pose_dict), last_roi = process_frame_with_dual_resolution(frame, scale_factor=0.5, roi=None)
        else:
            (annotated_frame, pose_dict), _ = process_frame_with_dual_resolution(frame, scale_factor=0.5, roi=last_roi)
        
        process_duration = (time.time() - capture_timestamp) * 1000
        #print(f"[{datetime.datetime.now()}] 카메라 {cam_index} 디텍션 처리 완료 (처리 시간: {process_duration:.2f} ms)")
        
        update_start = time.time()
        BASE_TAG = 1
        if BASE_TAG in pose_dict:
            rvec, tvec = pose_dict[BASE_TAG]
            measurement_left = np.array(tvec, dtype=np.float32).reshape(3)
            avg_left = update_measurements(measurement_left, recent_measurements_left)
            pose_dict[BASE_TAG] = (rvec, avg_left)
        else:
            if recent_measurements_left:
                avg_left = np.mean(recent_measurements_left, axis=0)
            else:
                avg_left = np.zeros(3, dtype=np.float32)
            pose_dict[BASE_TAG] = (np.zeros((3,1), dtype=np.float32), avg_left)
        update_duration = (time.time() - update_start) * 1000
        #print(f"[{datetime.datetime.now()}] 카메라 {cam_index} 1번 마커 업데이트: {update_duration:.2f} ms")
        
        rel_start = time.time()
        TARGET_TAGS = [11,12,13,14,15]
        rel_positions = compute_relative_positions(pose_dict, base_tag=BASE_TAG, target_tags=TARGET_TAGS)
        rel_duration = (time.time() - rel_start) * 1000
        #print(f"[{datetime.datetime.now()}] 카메라 {cam_index} 상대 좌표 계산: {rel_duration:.2f} ms")
        
        if prev_frame_time is not None:
            delta_time = (capture_timestamp - prev_frame_time) * 1000
            print(f"[{datetime.datetime.now()}] 카메라 {cam_index} 프레임 간 시간 차이: {delta_time:.2f} ms",flush=True)
        prev_frame_time = capture_timestamp
        
        overall_duration = (time.time() - overall_start) * 1000
        print(f"[{datetime.datetime.now()}] 카메라 {cam_index} 전체 한 프레임 처리: {overall_duration:.2f} ms\n", flush=True)
        
        #display_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
        #cv2.imshow(f"ArUco Detection {cam_index}", display_frame)
        #if cv2.waitKey(1) & 0xFF == 27:
        #    break
    
    cap.release()
    cv2.destroyAllWindows()

########################################
# 메인 함수: 각 카메라에 대해 스레드를 생성하여 실행
########################################

def main():
    cam_indices = [0, 2, 4, 6]
    threads = []
    for idx in cam_indices:
        t = threading.Thread(target=camera_thread, args=(idx,), daemon=True)
        t.start()
        threads.append(t)
    
    print("다중 카메라 스레드 실행 중 (ESC 키 또는 Ctrl+C로 종료)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("프로그램 종료")

if __name__ == "__main__":
    main()
