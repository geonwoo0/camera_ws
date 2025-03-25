import cv2
import numpy as np
import time
import threading
import queue
import datetime
import apriltag
from collections import deque

# 최근 N개 측정값 저장 (예: N = 5)
N = 5
recent_measurements_left = deque(maxlen=N)

########################################
# 캘리브레이션 데이터 로드 함수
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

# 기본 캘리브레이션 데이터
default_matrix = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0,   0,   1]], dtype=np.float32)
default_dist = np.zeros((5, 1), dtype=np.float32)

# 왼쪽 카메라 캘리브레이션 데이터 (경로 수정 필요)
camera_matrix_left, dist_coeffs_left = load_calibration_data(
    "../camera_calibration/cam8_calibration_data.npz",
    default_matrix,
    default_dist
)

########################################
# 태그 및 ROI 관련 함수
########################################

def update_measurements(measurement, measurements_deque):
    measurements_deque.append(measurement)
    avg = np.mean(measurements_deque, axis=0)
    return avg

def detect_tags_low_res(frame_high, detector, scale_factor=0.5):
    """
    높은 해상도의 프레임을 낮은 해상도로 다운샘플하여 태그를 검출.
    """
    frame_low = cv2.resize(frame_high, None, fx=scale_factor, fy=scale_factor)
    gray_low = cv2.cvtColor(frame_low, cv2.COLOR_BGR2GRAY)
    tags_low = detector.detect(gray_low)
    return tags_low, scale_factor

def get_roi_from_low_res_tags(tags_low, scale_factor, margin=20):
    """
    낮은 해상도에서 검출된 태그들의 코너 좌표를 기반으로 원본(고해상도) 이미지의 ROI 계산.
    """
    if not tags_low:
        return None

    all_corners = np.concatenate([tag.corners for tag in tags_low])
    x_min = np.min(all_corners[:, 0])
    y_min = np.min(all_corners[:, 1])
    x_max = np.max(all_corners[:, 0])
    y_max = np.max(all_corners[:, 1])

    # 고해상도 좌표로 변환 (scale_factor 역수 적용) 및 margin 추가
    x_min_hr = int(x_min / scale_factor) - margin
    y_min_hr = int(y_min / scale_factor) - margin
    x_max_hr = int(x_max / scale_factor) + margin
    y_max_hr = int(y_max / scale_factor) + margin

    roi = (max(0, x_min_hr), max(0, y_min_hr),
           x_max_hr - x_min_hr, y_max_hr - y_min_hr)
    return roi

def process_frame_high_res_with_roi(frame_high, detector, camera_matrix, dist_coeffs, roi):
    """
    고해상도 이미지에서 지정된 ROI 영역만 잘라내어 태그 검출 수행.
    ROI 내에서 검출된 태그 좌표를 원본 좌표로 변환.
    """
    if roi is not None:
        x, y, w, h = roi
        frame_roi = frame_high[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = cv2.cvtColor(frame_high, cv2.COLOR_BGR2GRAY)

    tags_hr = detector.detect(gray_roi)
    pose_dict = {}
    for tag in tags_hr:
        if roi is not None:
            #tag.corners += np.array([x, y])
            #tag.center += np.array([x, y])
            corners = tag.corners.copy()  # 복사본 생성
            corners += np.array([x, y])
            center = tag.center.copy()    # 복사본 생성
            center += np.array([x, y])
        # 태그 크기 결정 (예시: tag_id에 따라)
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
            cv2.putText(frame_high, str(tag.tag_id),
                        (int(tag.center[0]), int(tag.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"[Process] solvePnP 실패 - 태그 {tag.tag_id}")
    return frame_high, pose_dict

def process_frame_with_dual_resolution(frame_high, detector, camera_matrix, dist_coeffs, scale_factor=0.5):
    """
    낮은 해상도에서 태그 검출로 ROI 결정 후, 고해상도 ROI 영역에서 정밀 검출 수행.
    """
    # 1. 낮은 해상도에서 태그 후보 검출
    tags_low, scale = detect_tags_low_res(frame_high, detector, scale_factor)
    # 2. ROI 계산
    roi = get_roi_from_low_res_tags(tags_low, scale)
    if roi:
        print(f"[ROI] 결정된 영역: {roi}")
    else:
        print("[ROI] 태그 미검출 - 전체 영역 사용")
    # 3. 고해상도 ROI 영역에서 태그 정밀 검출
    annotated_frame, pose_dict = process_frame_high_res_with_roi(frame_high, detector, camera_matrix, dist_coeffs, roi)
    return annotated_frame, pose_dict

def compute_relative_positions(pose_dict, base_tag=501, target_tags=[11, 12, 13, 14, 15]):
    relative_positions = {}
    if base_tag not in pose_dict:
        print(f"[Relative] 기준 태그 {base_tag} 미검출")
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
            print(f"[Relative] 태그 {base_tag} 기준 태그 {tag} 상대 좌표: "
                  f"x={rel_vec[0]:.4f}, y={rel_vec[1]:.4f}, z={rel_vec[2]:.4f} m")
    return relative_positions

########################################
# 카메라 캡처 및 처리 스레드
########################################

def capture_camera(cam_index, frame_queue, cam_label="Camera"):
    cap = cv2.VideoCapture(cam_index)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # 해상도 설정 (예: 고해상도)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"현재 적용된 코덱: "
          f"{chr(fourcc_val & 0xFF)}"
          f"{chr((fourcc_val >> 8) & 0xFF)}"
          f"{chr((fourcc_val >> 16) & 0xFF)}"
          f"{chr((fourcc_val >> 24) & 0xFF)}")
    print(f"카메라 해상도: {width} x {height} / {fps}")

    if not cap.isOpened():
        print(f"[{cam_label}] 카메라 열기 실패")
        return
    print(f"[{cam_label}] 카메라 열림: 인덱스 {cam_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_label}] 프레임 읽기 실패")
            continue
        timestamp = time.time()
        frame_queue.put((timestamp, frame))
    cap.release()

def process_camera_frames(frame_queue, result_queue, calibration_data, cam_label="Camera"):
    # 디텍터 옵션 최적화 (낮은 해상도, ROI, 고해상도 적용)
    options = apriltag.DetectorOptions(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        refine_edges=False,
    )
    detector = apriltag.Detector(options)
    camera_matrix, dist_coeffs = calibration_data

    while True:
        try:
            capture_timestamp, frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        process_start = time.time()
        # 낮은 해상도에서 ROI를 결정하고 고해상도에서 정밀 검출 수행
        annotated_frame, pose_dict = process_frame_with_dual_resolution(frame, detector,
                                                                        camera_matrix, dist_coeffs,
                                                                        scale_factor=0.5)
        process_end = time.time()
        print(f"[{datetime.datetime.now()}] 디텍션 처리 완료 "
              f"(처리 시간: {(process_end - process_start)*1000:.2f} ms)")

        result_queue.put((capture_timestamp, annotated_frame, pose_dict))

########################################
# 메인 루프
########################################

def main():
    LEFT_CAM_INDEX = 2
    left_frame_queue = queue.Queue(maxsize=2)
    left_result_queue = queue.Queue(maxsize=2)
    calibration_left = (camera_matrix_left, dist_coeffs_left)

    # 캡처 스레드
    t_left = threading.Thread(
        target=capture_camera,
        args=(LEFT_CAM_INDEX, left_frame_queue, "Left"),
        daemon=True
    )
    t_left.start()

    # 처리 스레드
    t_left_proc = threading.Thread(
        target=process_camera_frames,
        args=(left_frame_queue, left_result_queue, calibration_left, "Left"),
        daemon=True
    )
    t_left_proc.start()

    prev_timestamp = None
    BASE_TAG = 501
    TARGET_TAGS = [11, 12, 13, 14, 15]
    print("단일 카메라를 이용한 Dual Resolution AprilTag 검출 및 "
          "501 태그 기준 상대 좌표 계산 시작 (Ctrl+C 종료)")

    while True:
        overall_start = time.time()
        if left_result_queue.empty():
            continue

        res_start = time.time()
        capture_timestamp, left_frame, left_pose_dict = left_result_queue.get()
        res_end = time.time()

        print(f"[{datetime.datetime.now()}] "
              f"결과 큐에서 프레임 가져오기 시간: {(res_end - res_start)*1000:.2f} ms")

        # 프레임 간 시간 차이
        if prev_timestamp is not None:
            delta_ms = (capture_timestamp - prev_timestamp) * 1000
            print(f"[{datetime.datetime.now()}] 이전 프레임 대비 시간: {delta_ms:.2f} ms")
        prev_timestamp = capture_timestamp

        # 501 태그 업데이트 (최근 측정값 평활화)
        update_start = time.time()
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
        update_end = time.time()
        print(f"[{datetime.datetime.now()}] 501 태그 업데이트: {(update_end - update_start)*1000:.2f} ms")

        # 상대 좌표 계산
        rel_start = time.time()
        rel_left = compute_relative_positions(left_pose_dict, base_tag=BASE_TAG, target_tags=TARGET_TAGS)
        rel_end = time.time()
        print(f"[{datetime.datetime.now()}] 상대 좌표 계산: {(rel_end - rel_start)*1000:.2f} ms")

        overall_end = time.time()
        print(f"[{datetime.datetime.now()}] 전체 한 프레임 처리: {(overall_end - overall_start)*1000:.2f} ms\n")

        # 필요 시 결과 프레임 표시 (디버깅 용)
        cv2.imshow("AprilTag Detection", left_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
