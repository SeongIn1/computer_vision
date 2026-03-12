import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # Numpy 라이브러리 임포트

drawing = False # 마우스 드래그 상태를 확인하는 전역 변수 (초기값 False)
ix, iy = -1, -1 # 마우스 드래그 시작(클릭) 좌표를 저장할 전역 변수 초기화
roi = None # 추출된 관심 영역(ROI) 이미지를 임시 저장할 전역 변수 초기화

img = cv.imread('girl_laughing.jpg') # 'girl_laughing.jpg' 원본 이미지를 불러옵니다.

if img is None: # 이미지 로드에 실패했을 경우
    print("이미지를 찾을 수 없습니다.") # 콘솔에 에러 메시지를 출력합니다.
    exit() # 프로그램을 즉시 종료합니다.

clone = img.copy() # 화면 갱신 및 초기화를 위해 원본 이미지의 복사본(clone)을 생성합니다.

def draw_roi(event, x, y, flags, param): # 마우스 이벤트를 처리할 콜백 함수 정의
    global ix, iy, drawing, img, clone, roi # 외부의 전역 변수들을 함수 내부에서 제어하기 위해 선언

    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 처음 클릭했을 때
        drawing = True # 드래그(영역 선택) 상태를 시작합니다.
        ix, iy = x, y # 클릭한 시작점의 좌표를 ix, iy에 저장합니다.

    elif event == cv.EVENT_MOUSEMOVE: # 마우스를 이동할 때
        if drawing: # 드래그 중인 상태라면
            img_copy = clone.copy() # 사각형 궤적을 보여주기 위해 깨끗한 복사본을 다시 복사해옵니다.
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2) # 시작점(ix,iy)부터 현재 위치(x,y)까지 초록색(0,255,0), 두께 2의 사각형을 그립니다.
            cv.imshow('ROI Selection', img_copy) # 실시간으로 사각형이 커지고 작아지는 모습을 화면에 출력합니다.

    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼을 뗐을 때 (선택 완료)
        drawing = False # 드래그 상태를 종료합니다.

        x1, x2 = min(ix, x), max(ix, x) # 드래그 방향에 상관없이 올바른 슬라이싱을 위해 x좌표를 작은값(x1)과 큰값(x2)으로 정렬합니다.
        y1, y2 = min(iy, y), max(iy, y) # 마찬가지로 y좌표를 작은값(y1)과 큰값(y2)으로 정렬합니다.

        if x2 > x1 and y2 > y1: # 단순 클릭이 아니라 유효한 크기의 사각형 영역을 선택했을 경우에만
            cv.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2) # 선택 완료를 알리기 위해 기준 이미지(clone)에 최종 초록색 사각형을 픽스하여 그립니다.
            cv.imshow('ROI Selection', clone) # 최종 사각형이 그려진 이미지를 출력합니다.

            roi = img[y1:y2, x1:x2] # Numpy 배열 슬라이싱을 이용해 사각형이 없는 원본(img)에서 선택된 영역만 잘라내어 roi 변수에 저장합니다.
            cv.imshow('Cropped ROI', roi) # 잘라낸 관심 영역(ROI)을 확인하기 위해 새로운 창에 띄웁니다.

cv.namedWindow('ROI Selection') # 'ROI Selection' 이라는 이름의 메인 창을 생성합니다.
cv.setMouseCallback('ROI Selection', draw_roi) # 해당 메인 창에 위에서 정의한 마우스 콜백 함수를 등록합니다.

while True: # 화면을 유지하고 키 입력을 받기 위한 무한 루프 시작
    if not drawing: # 마우스 드래그 중이 아닐 때만
        cv.imshow('ROI Selection', clone) # 화면 깜빡임을 방지하기 위해 현재 상태의 이미지를 보여줍니다.

    key = cv.waitKey(1) & 0xFF # 1ms 대기 후 키보드 입력값을 받습니다.

    if key == ord('r'): # 사용자가 'r' 키를 누르면 (Reset)
        clone = img.copy() # clone을 다시 깨끗한 원본 이미지로 덮어씌워 사각형을 지웁니다.
        roi = None # 저장된 roi 데이터도 초기화합니다.
        print("영역 선택 초기화") # 콘솔에 초기화 상태를 알립니다.
    elif key == ord('s'): # 사용자가 's' 키를 누르면 (Save)
        if roi is not None: # 유효하게 잘라낸 영역(roi)이 존재한다면
            cv.imwrite('roi_saved.png', roi) # 해당 영역을 'roi_saved.png' 라는 이름의 이미지 파일로 저장합니다.
            print("저장 완료") # 콘솔에 저장 성공을 알립니다.
    elif key == ord('q'): # 사용자가 'q' 키를 누르면 (Quit)
        break # 무한 루프를 종료하고 프로그램을 끝냅니다.

cv.destroyAllWindows() # 열려있는 모든 OpenCV 창을 닫습니다.