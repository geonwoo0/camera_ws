import cv2
import numpy as np
import glob

######카메라 번호 수정 필수#######
cam_number = 8
#################################

# 체스보드 패턴 크기 (내부 교차점 개수)
CHECKERBOARD = (7,10)

# 3D 점과 2D 점을 저장할 배열
objpoints = []  # 실제 3D 공간의 점들
imgpoints = []  # 이미지 평면의 2D 점들

# 체스보드의 3D 점 준비 (z=0 평면)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 캘리브레이션 이미지 경로 설정 (현재 폴더 내의 jpg 파일)
images = glob.glob(f'./cam_{cam_number}/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"이미지 {fname} 를 읽어오지 못했습니다.")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 그리기 및 결과 표시
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
    else:
        print(f"체스보드 코너를 찾지 못했습니다: {fname}")

cv2.destroyAllWindows()

# 유효한 이미지가 하나 이상 있는지 확인
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("체스보드 코너를 검출한 이미지가 하나도 없습니다. 캘리브레이션을 진행할 수 없습니다.")

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)

# 캘리브레이션 결과 저장
np.savez(f"./cam{cam_number}_calibration_data.npz", camera_matrix=mtx, distortion_coefficients=dist)


cv2.destroyAllWindows()
