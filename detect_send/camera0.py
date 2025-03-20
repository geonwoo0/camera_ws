import cv2
from pupil_apriltags import Detector
import datetime
import numpy as np
import queue

def capture(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = Detector(families='tag36h11',
                        nthreads=1,             # 사용 스레드 수 (기본값=1, 멀티코어 활용 가능)
                        quad_decimate=1.0,      # 이미지 해상도 감소 비율 (기본값=1.0, 낮을수록 정확성 증가)
                        quad_sigma=0.0,         # 이미지 블러(smoothing) 정도 (기본값=0.0)
                        refine_edges=1,         # 엣지 미세조정 여부 (기본값=1, 활성화 권장)
                        decode_sharpening=0.25, # 디코딩 향상을 위한 샤프닝 정도 (기본값=0.25)
                    )
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"[{index}] 카메라 프레임 불러오기 실패")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        pose_dict = {}
        
        for tag in tags:
            if tag.tag_id >=400:
                tag_size = 0.05
            elif tag.tag_id>=0:
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