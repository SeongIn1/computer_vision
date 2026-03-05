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