# OpenCV 실습 과제 (E01)

## 📝 과제에 대한 설명
본 저장소는 동아대학교 컴퓨터비전 강의의 OpenCV 기초 실습 과제 결과물입니다. 
총 3가지의 실습을 통해 OpenCV의 기초적인 이미지 입출력, 변환, 마우스 이벤트 처리 및 관심영역(ROI) 추출 방법을 학습했습니다.

1. **실습 01. 이미지 불러오기 및 그레이스케일 변환 (`ex01.py`)**
   - 이미지를 불러와 화면에 출력하고, 원본 이미지와 흑백(Grayscale)으로 변환된 이미지를 나란히 가로로 연결하여 표시합니다.
* [cite_start]**`cv.imread()`**: 지정한 경로의 이미지 파일을 읽어와 NumPy 배열(BGR 형식)로 로드합니다. [cite: 1429, 1433]
* [cite_start]**`cv.cvtColor()`**: 이미지의 색상 공간을 변환합니다. [cite: 1430] [cite_start]BGR 이미지를 흑백으로 변환할 때 `cv.COLOR_BGR2GRAY` 속성을 사용했습니다. [cite: 1434]
* [cite_start]**`np.hstack()`**: 두 개의 이미지 배열을 가로로 나란히 병합합니다. [cite: 1431] (이때 두 이미지의 채널 수가 같아야 하므로, 흑백 이미지를 임시로 3채널로 변환하여 병합했습니다.)
* [cite_start]**`cv.imshow()` & `cv.waitKey()`**: 이미지를 화면에 띄우고, 창이 즉시 닫히지 않도록 키보드 입력을 대기합니다. [cite: 1432]

전체코드

import cv2 as cv # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import numpy as np # 배열 연산을 위한 numpy 라이브러리를 np라는 이름으로 불러옵니다.

img = cv.imread('soccer.jpg') # 'soccer.jpg' 이미지를 BGR 형식으로 읽어와 img 변수에 저장합니다.

if img is None: # 이미지를 정상적으로 불러오지 못했다면 (파일이 없거나 경로가 틀린 경우)
    print("이미지를 찾을 수 없습니다.") # 에러 메시지를 콘솔에 출력합니다.
    exit() # 프로그램을 즉시 종료합니다.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 이미지를 흑백(Grayscale) 이미지로 변환합니다.

gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 흑백 이미지를 원본과 병합하기 위해 채널 수를 3개(BGR)로 맞춥니다. (색상은 흑백 유지)

combined = np.hstack((img, gray_3ch)) # numpy의 hstack 함수를 사용하여 원본 이미지와 3채널 흑백 이미지를 가로로 나란히 연결합니다.

cv.imshow('Original and Grayscale', combined) # 'Original and Grayscale'이라는 이름의 창을 생성하고 병합된 이미지를 화면에 띄웁니다.

cv.waitKey(0) # 사용자가 키보드의 아무 키나 누를 때까지 창을 닫지 않고 무한히 대기합니다.
cv.destroyAllWindows() # 키 입력이 감지되면 열려있는 모든 OpenCV 창을 닫고 메모리를 해제합니다.

2. **실습 02. 페인팅 붓 크기 조절 기능 추가 (`ex02.py`)**
   - 마우스 좌/우 클릭 및 드래그를 통해 이미지 위에 파란색/빨간색 선을 그리고, 키보드 `+`, `-` 키를 이용해 붓의 크기를 조절하는 기능을 구현했습니다.
* [cite_start]**`cv.setMouseCallback()`**: 특정 창에서 발생하는 마우스 이벤트(클릭, 드래그, 이동 등)를 감지하고, 지정한 콜백 함수를 실행하여 이벤트를 처리합니다. [cite: 1448, 1466]
* [cite_start]**`cv.circle()`**: 마우스가 이동한 좌표를 중심으로 지정된 크기와 색상의 원을 그려 페인팅 브러시 효과를 구현했습니다. [cite: 1448]
* [cite_start]**`cv.rectangle()`**: 마우스 드래그 시작점과 끝점을 연결하여 화면에 사각형을 그려 선택 영역을 시각화했습니다. [cite: 1471]

전체코드

import cv2 as cv # OpenCV 라이브러리를 불러옵니다.
import numpy as np # 배열 연산을 위한 numpy 라이브러리를 불러옵니다.

img = np.full((500, 500, 3), 255, dtype=np.uint8) # 500x500 픽셀 크기의 흰색(255) 배경을 가진 3채널(BGR) 이미지를 생성합니다.

brush_size = 5 # 붓의 초기 크기를 5로 설정합니다.
is_drawing = False # 현재 마우스로 그림을 그리고 있는 상태인지 나타내는 상태 변수를 False로 초기화합니다.
color = (0, 0, 0) # 붓의 색상을 저장할 변수를 검은색으로 초기화합니다.

def draw_circle(event, x, y, flags, param): # 마우스 이벤트를 처리할 콜백 함수를 정의합니다.
    global is_drawing, color, brush_size # 함수 외부의 전역 변수들을 함수 내부에서 수정할 수 있도록 선언합니다.

    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼이 눌렸을 때
        is_drawing = True # 그리기 상태를 True로 변경하여 그리기를 시작합니다.
        color = (255, 0, 0) # 붓의 색상을 파란색(BGR 기준 255, 0, 0)으로 설정합니다.
    elif event == cv.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼이 눌렸을 때
        is_drawing = True # 그리기 상태를 True로 변경하여 그리기를 시작합니다.
        color = (0, 0, 255) # 붓의 색상을 빨간색(BGR 기준 0, 0, 255)으로 설정합니다.
    elif event == cv.EVENT_MOUSEMOVE: # 마우스가 움직일 때
        if is_drawing: # 만약 그리기 상태가 True라면 (버튼을 누른 채로 움직인다면)
            cv.circle(img, (x, y), brush_size, color, -1) # 현재 마우스 좌표(x,y)에 설정된 크기와 색상으로 내부가 채워진(-1) 원을 그립니다.
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP: # 마우스 왼쪽이나 오른쪽 버튼을 뗐을 때
        is_drawing = False # 그리기 상태를 False로 변경하여 그리기를 멈춥니다.
        cv.circle(img, (x, y), brush_size, color, -1) # 버튼을 뗀 마지막 위치에 원을 하나 그리고 마무리합니다.

cv.namedWindow('Painting') # 'Painting'이라는 이름의 윈도우 창을 생성합니다.
cv.setMouseCallback('Painting', draw_circle) # 'Painting' 창에 마우스 이벤트가 발생하면 draw_circle 함수를 실행하도록 콜백을 연결합니다.

while True: # 사용자가 종료할 때까지 화면을 계속 갱신하는 무한 루프를 엽니다.
    cv.imshow('Painting', img) # 'Painting' 창에 현재까지 그려진 캔버스(img) 이미지를 보여줍니다.

    key = cv.waitKey(1) & 0xFF # 1밀리초 동안 대기하며 키보드 입력을 받고, 그 값의 아스키코드를 추출합니다.

    if key == ord('q'): # 만약 입력된 키가 소문자 'q'라면
        break # 무한 루프를 빠져나가 프로그램을 종료합니다.
    elif key == ord('+') or key == ord('='): # 만약 입력된 키가 '+' 이거나 '=' 라면
        if brush_size < 15: # 현재 붓 크기가 최대 제한인 15보다 작을 때만
            brush_size += 1 # 붓 크기를 1 증가시킵니다.
    elif key == ord('-'): # 만약 입력된 키가 '-' 라면
        if brush_size > 1: # 현재 붓 크기가 최소 제한인 1보다 클 때만
            brush_size -= 1 # 붓 크기를 1 감소시킵니다.

cv.destroyAllWindows() # 열려있는 모든 창을 닫고 프로그램을 안전하게 종료합니다.

3. **실습 03. 마우스로 영역 선택 및 ROI 추출 (`ex03.py`)**
   - 마우스 드래그를 통해 이미지에서 특정 영역(ROI)을 선택하고, 선택된 영역만 새로운 창에 띄우거나 `s` 키를 눌러 별도의 파일로 저장하는 기능을 구현했습니다.
* [cite_start]**Numpy 배열 슬라이싱 (Slicing)**: OpenCV에서 이미지는 Numpy 배열로 취급되므로, 마우스로 선택한 사각형의 좌표(y1:y2, x1:x2)를 이용해 배열을 슬라이싱하는 방식으로 원하는 ROI만 추출했습니다. [cite: 1472]
* [cite_start]**`cv.imwrite()`**: 추출된 ROI 이미지 배열을 지정한 파일 이름(`roi_saved.png`)으로 PC에 저장합니다. [cite: 1470, 1473]

전체코드

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

## 📸 실행 결과물

### 실습 01 결과
* **최종 결과물:** 원본 이미지와 그레이스케일 이미지가 나란히 병합되어 출력된 화면
  > <img width="1601" height="559" alt="과제1번 출력" src="https://github.com/user-attachments/assets/477d1617-6572-4137-aaae-6a72babf8cd3" />


### 실습 02 결과
* **최종 결과물:** `+`, `-` 키로 붓 크기를 조절하여 다양한 두께로 그린 최종 화면
  > <img width="495" height="527" alt="과제2번 출력" src="https://github.com/user-attachments/assets/eb635e73-7276-4952-b56f-cf884a968c7e" />


### 실습 03 결과
* **중간 결과물:** 마우스로 드래그하여 초록색 사각형으로 영역을 선택하는 중인 화면
  > <img width="1434" height="988" alt="과제3번 출력" src="https://github.com/user-attachments/assets/4692b9cb-e226-4bc7-9952-0bfbbedb25c4" />


* **최종 결과물:** 선택된 영역이 크롭(Crop)되어 새로운 창에 뜬 화면 및 저장된 `roi_saved.png` 파일
  > <img width="515" height="455" alt="roi_saved" src="https://github.com/user-attachments/assets/3d6195f1-d966-4619-ab8a-7c2d1a02ad96" />
