import cv2
import numpy as np
from pupil_apriltags import Detector

##### --- 보정 데이터 로드 / 캘리브레이션 파일명 변경 필요 --- #####
# 좌측 카메라 보정 데이터
try:
    with np.load("../camera_calibration/cam8_calibration_data.npz") as X:
        l_camera_matrix = X["camera_matrix"]
        l_dist_coeffs = X["distortion_coefficients"]
except Exception as e:
    print("좌측 캘리브레이션 데이터를 로드하지 못했습니다. 기본값을 사용합니다.", e)
    l_camera_matrix = np.array([[600, 0, 320],
                               [0, 600, 240],
                               [0,   0,   1]], dtype=np.float32)
    l_dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 우측 카메라 보정 데이터
try:
    with np.load("../camera_calibration/cam9_calibration_data.npz") as X:
        r_camera_matrix = X["camera_matrix"]
        r_dist_coeffs = X["distortion_coefficients"]
except Exception as e:
    print("우측 캘리브레이션 데이터를 로드하지 못했습니다. 기본값을 사용합니다.", e)
    r_camera_matrix = np.array([[600, 0, 320],
                               [0, 600, 240],
                               [0,   0,   1]], dtype=np.float32)
    r_dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# --- AprilTag Detector 초기화 ---
detector = Detector(families='tag36h11')

# AprilTag 기본 크기 (미터 단위, 태그별로 다르게 처리할 예정)
default_tag_size = 0.02  # 2cm

# --- 카메라 캡처 ---
# 좌측: cam index 4, 우측: cam index 0 (환경에 맞게 조정)
l_cap = cv2.VideoCapture(4)
r_cap = cv2.VideoCapture(2)

# 해상도 설정
l_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
l_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
r_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
r_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

#카메라 압축
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
l_cap.set(cv2.CAP_PROP_FOURCC,fourcc)
r_cap.set(cv2.CAP_PROP_FOURCC,fourcc)

#프레임 설정
l_cap.set(cv2.CAP_PROP_FPS,30)
r_cap.set(cv2.CAP_PROP_FPS,30)

cv2.namedWindow("AprilTag Pose (Left & Right)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AprilTag Pose (Left & Right)", 1080, 720)

if not l_cap.isOpened() or not r_cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 AprilTag 위치 추정 중... 'q'를 눌러 종료하세요.")

while True:
    l_ret, l_frame = l_cap.read()
    r_ret, r_frame = r_cap.read()
    if not l_ret or not r_ret:
        print("프레임 읽기 실패")
        break
    
    # 보정 적용
    l_frame = cv2.undistort(l_frame, l_camera_matrix, l_dist_coeffs, None)
    r_frame = cv2.undistort(r_frame, r_camera_matrix, r_dist_coeffs, None)
    
    # 그레이스케일 변환
    l_gray = cv2.cvtColor(l_frame, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
    
    # AprilTag 검출
    l_tags = detector.detect(l_gray)
    r_tags = detector.detect(r_gray)
    
    l_pose_dict = {}
    r_pose_dict = {}
    
    # 좌측 카메라에서 태그 처리
    for tag in l_tags:
        # 태그 ID에 따라 크기 조정 (예: 태그 500, 499는 더 크게)
        if tag.tag_id >= 400:
            tag_size = 0.05
        else:
            tag_size = 0.018
        
        # 3D 객체 점 (태그 중심 기준, z=0 평면)
        object_points = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2,  tag_size / 2, 0],
            [-tag_size / 2,  tag_size / 2, 0]
        ], dtype=np.float32)
        
        image_points = tag.corners.astype(np.float32)
        
        # 좌측 카메라에서 자세 추정
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, l_camera_matrix, l_dist_coeffs)
        if success:
            l_pose_dict[tag.tag_id] = (rvec, tvec)
            
            cv2.drawFrameAxes(l_frame, l_camera_matrix, l_dist_coeffs, rvec, tvec, tag_size / 2)
            """distance = np.linalg.norm(tvec)
            cv2.putText(l_frame, f"ID: {tag.tag_id}, d:{distance:.2f}", 
                        (int(np.mean(tag.corners[:,0])) - 20, int(np.mean(tag.corners[:,1])) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            print(f"좌측 카메라 - Tag ID: {tag.tag_id}, tvec: {tvec.ravel()}, distance: {distance}")
"""
       
    # 우측 카메라에서 태그 처리
    for tag in r_tags:
        if tag.tag_id >= 400:
            tag_size = 0.05
        else:
            tag_size = 0.018
        
        object_points = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2,  tag_size / 2, 0],
            [-tag_size / 2,  tag_size / 2, 0]
        ], dtype=np.float32)
        
        image_points = tag.corners.astype(np.float32)
        
        # 우측 카메라에서 자세 추정
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, r_camera_matrix, r_dist_coeffs)
        if success:
            r_pose_dict[tag.tag_id] = (rvec, tvec)
            
            cv2.drawFrameAxes(r_frame, r_camera_matrix, r_dist_coeffs, rvec, tvec, tag_size / 2)
            """distance = np.linalg.norm(tvec)
            cv2.putText(r_frame, f"ID: {tag.tag_id}, d:{distance:.2f}", 
                        (int(np.mean(tag.corners[:,0])) - 20, int(np.mean(tag.corners[:,1])) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            print(f"우측 카메라 - Tag ID: {tag.tag_id}, tvec: {tvec.ravel()}, distance: {distance}")
    """
    if 501 not in l_pose_dict or 501 not in r_pose_dict:
        pass
        print("기준 태그 501이 왼쪽 혹은 오른쪽 카메라에서 감지되지 않았습니다.")
    else:
        # --- 왼쪽 카메라의 태그 501 변환 행렬 구성 ---
        l_rvec_501, l_tvec_501 = l_pose_dict[501]
        l_R_501, _ = cv2.Rodrigues(l_rvec_501)
        l_T_cam_501 = np.eye(4)
        l_T_cam_501[:3, :3] = l_R_501
        l_T_cam_501[:3, 3] = l_tvec_501.ravel()
        # 태그501의 좌표계로 변환 (왼쪽 카메라)
        l_T_501_cam = np.linalg.inv(l_T_cam_501)

        # --- 오른쪽 카메라의 태그 501 변환 행렬 구성 ---
        r_rvec_501, r_tvec_501 = r_pose_dict[501]
        r_R_501, _ = cv2.Rodrigues(r_rvec_501)
        r_T_cam_501 = np.eye(4)
        r_T_cam_501[:3, :3] = r_R_501
        r_T_cam_501[:3, 3] = r_tvec_501.ravel()
        # 태그501의 좌표계로 변환 (오른쪽 카메라)
        r_T_501_cam = np.linalg.inv(r_T_cam_501)

        # --- 왼쪽 카메라에서 다른 태그들의 상대 좌표 및 거리 계산 ---
        print("왼쪽 카메라 기준:")
        for tag_id, (rvec, tvec) in l_pose_dict.items():
            if tag_id == 501:
                continue
            R, _ = cv2.Rodrigues(rvec)
            T_cam_tag = np.eye(4)
            T_cam_tag[:3, :3] = R
            T_cam_tag[:3, 3] = tvec.ravel()
            # 기준 태그501 좌표계로 변환
            T_501_tag = l_T_501_cam @ T_cam_tag
            relative_tvec = T_501_tag[:3, 3]
            relative_distance = np.linalg.norm(relative_tvec)
            print(f"태그 501 기준 - 태그 {tag_id}: 상대 좌표 {relative_tvec}, 거리 {relative_distance:.2f}")

        # --- 오른쪽 카메라에서 다른 태그들의 상대 좌표 및 거리 계산 ---
        print("오른쪽 카메라 기준:")
        for tag_id, (rvec, tvec) in r_pose_dict.items():
            if tag_id == 501:
                continue
            R, _ = cv2.Rodrigues(rvec)
            T_cam_tag = np.eye(4)
            T_cam_tag[:3, :3] = R
            T_cam_tag[:3, 3] = tvec.ravel()
            # 기준 태그501 좌표계로 변환
            T_501_tag = r_T_501_cam @ T_cam_tag
            relative_tvec = T_501_tag[:3, 3]
            relative_distance = np.linalg.norm(relative_tvec)
            print(f"태그 501 기준 - 태그 {tag_id}: 상대 좌표 {relative_tvec}, 거리 {relative_distance:.2f}")
        
    #이미지 출력 전 사이즈 조정
    l_frame = cv2.resize(l_frame, (1920,1080), interpolation=cv2.INTER_AREA)
    r_frame = cv2.resize(r_frame, (1920,1080), interpolation=cv2.INTER_AREA)
    
    # (옵션) 프레임 회전 (필요 시)
    l_frame = cv2.rotate(l_frame, cv2.ROTATE_90_CLOCKWISE)
    r_frame = cv2.rotate(r_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 좌측과 우측 이미지를 나란히 결합하여 표시
    combined = cv2.hconcat([l_frame, r_frame])
    cv2.imshow("AprilTag Pose (Left & Right)", combined)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

l_cap.release()
r_cap.release()
cv2.destroyAllWindows()