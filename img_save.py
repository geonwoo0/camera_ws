import os
import cv2
import numpy as np

######카메라 번호 수정 필수#######
cam_number = 8
##############################

#카메라 인덱스 수정
cap = cv2.VideoCapture(2)

#해상도 설정(4K)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

#카메라 압축 - MJPG가 빠름
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC,fourcc)

#프레임 설정
cap.set(cv2.CAP_PROP_FPS,30)

img_count = 0

try:
    if not os.path.exists(f"./cam_{cam_number}"):
        os.makedirs(f"./cam_{cam_number}")
except OSError:
    print("Error: Failed to create the directory.")

while True :
    _,frame = cap.read()
    
    #이미지 축소 후 화면에 표시
    resize_frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_AREA)
    cv2.imshow("camera_calib",resize_frame)
    key = cv2.waitKey(5) & 0xFF
    
    if key == 13: #enter
        #저장은 원본사이즈(4k)
        cv2.imwrite(f"./cam_{cam_number}/cam{cam_number}_img{img_count}.jpg",frame)
        print(f"cam{cam_number} - {img_count}번 이미지 저장")
        img_count+=1
    elif key == 27 : #esc
        break
    
cap.release()
cv2.destroyAllWindows()