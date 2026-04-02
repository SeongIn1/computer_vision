
# [컴퓨터비전] L05. Image Recognition 실습 과제

본 레포지토리는 TensorFlow와 Keras를 활용하여 MNIST 데이터셋을 이용한 간단한 이미지 분류기를 구현하고, CIFAR-10 데이터셋을 활용해 합성곱 신경망(CNN) 모델을 구축하여 이미지 분류 및 외부 이미지 예측을 수행하는 두 가지 실습 과제의 결과물을 담고 있습니다.

---

## 01. 간단한 이미지 분류기 구현 (MNIST)

### 📌 과제 설명

`tensorflow.keras.datasets`에서 제공하는 28x28 픽셀 크기의 흑백 손글씨 숫자(MNIST) 데이터셋을 로드하고, 모델의 학습 효율을 높이기 위해 픽셀 값을 0~1 범위로 정규화합니다. `Sequential` 모델을 기반으로 `Flatten` 레이어를 통해 이미지를 1차원으로 펼친 후, `Dense` 레이어를 활용하여 간단한 인공신경망을 구축하고 훈련시켜 성능을 평가합니다.

### 🖼️ 중간 및 최종 결과물

* **모델 학습 및 테스트 정확도 평가 결과:**
  <img width="601" height="256" alt="1번과제 결과" src="https://github.com/user-attachments/assets/f1dca31d-94ad-4089-9660-88593f4b096a" />


### 💻 소스 코드

```python
import tensorflow as tf # 텐서플로우 라이브러리를 가져오고, tf라는 별칭으로 사용합니다.
from tensorflow.keras.datasets import mnist # 텐서플로우의 케라스 데이터셋에서 mnist 손글씨 데이터를 가져옵니다.
from tensorflow.keras.models import Sequential # 케라스 모델 중 순차적으로 레이어를 쌓는 Sequential 모델을 가져옵니다.
from tensorflow.keras.layers import Dense, Flatten # 케라스 레이어 중 완전 연결 층(Dense)과 평탄화 층(Flatten)을 가져옵니다.

# 1. MNIST 데이터셋 로드 및 훈련/테스트 세트 분할
(x_train, y_train), (x_test, y_test) = mnist.load_data() # MNIST 데이터를 다운로드 및 로드하여 훈련용과 테스트용 튜플로 각각 나눕니다.

# 데이터 전처리: 0~255 픽셀 값을 0~1 범위로 정규화하여 학습 효율을 높입니다.
x_train = x_train / 255.0 # 훈련 데이터의 모든 픽셀 값을 255로 나누어 0과 1 사이의 실수 값으로 변환합니다.
x_test = x_test / 255.0 # 테스트 데이터의 모든 픽셀 값도 동일하게 255로 나누어 정규화합니다.

# 2. 간단한 신경망 모델 구축
model = Sequential([ # 레이어를 순서대로 쌓아 올릴 Sequential 모델 객체를 생성합니다.
    Flatten(input_shape=(28, 28)), # 입력 데이터인 28x28 크기의 2차원 배열 이미지를 784개의 1차원 배열로 평탄화합니다.
    Dense(128, activation='relu'), # 128개의 노드를 가지고, 렐루(ReLU) 활성화 함수를 사용하는 은닉층을 추가합니다.
    Dense(10, activation='softmax') # 10개의 노드(0~9 클래스)를 가지고, 소프트맥스(softmax) 활성화 함수를 사용하는 출력층을 추가합니다.
]) # 모델 구성 리스트와 괄호를 닫아 Sequential 모델 구성을 마칩니다.

# 모델 컴파일 (최적화 알고리즘, 손실 함수, 평가 지표 설정)
model.compile(optimizer='adam', # 모델 학습 과정에서 가중치를 업데이트할 최적화 알고리즘으로 'adam'을 설정합니다.
              loss='sparse_categorical_crossentropy', # 정수형 레이블에 대한 다중 분류 손실 함수인 'sparse_categorical_crossentropy'를 설정합니다.
              metrics=['accuracy']) # 훈련 및 테스트 과정에서 모델 성능을 평가할 지표로 '정확도(accuracy)'를 설정합니다.

# 3. 모델 훈련 (에포크 5회)
print("--- 모델 훈련 시작 ---") # 터미널 창에 모델 훈련이 시작됨을 알리는 문자열을 출력합니다.
model.fit(x_train, y_train, epochs=5) # 훈련 데이터(x_train)와 레이블(y_train)을 사용하여 5번 반복(epochs=5) 학습시킵니다.

# 4. 모델 정확도 평가
print("\n--- 모델 성능 평가 ---") # 터미널 창에 성능 평가가 시작됨을 알리는 문자열을 줄바꿈하여 출력합니다.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) # 테스트 데이터와 레이블을 사용하여 모델을 평가하고 손실값과 정확도를 반환받습니다.
print(f'테스트 정확도: {test_acc:.4f}') # 평가 결과로 나온 테스트 정확도를 소수점 넷째 자리까지 포맷팅하여 출력합니다.
```

### 💡 핵심 함수 정리

* `mnist.load_data()`: MNIST 데이터셋을 자동으로 다운로드하고 훈련 세트와 테스트 세트로 나누어 반환합니다.
* `Sequential(...)`: 여러 신경망 레이어를 순서대로 층층이 쌓아 올릴 수 있는 직관적인 모델 구조를 생성합니다.
* `Dense(units, activation)`: 완전 연결 층(Fully Connected Layer)을 생성합니다. `units` 파라미터로 출력 노드의 수를 지정하고, `activation`으로 사용할 활성화 함수(예: relu, softmax)를 설정합니다.

---

## 02. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

### 📌 과제 설명

10가지 클래스로 구성된 32x32 크기의 컬러 이미지 데이터셋인 CIFAR-10을 로드하여 전처리합니다. `Conv2D`와 `MaxPooling2D` 레이어를 여러 겹 조합하여 이미지의 공간적 특징을 추출하는 합성곱 신경망(CNN)을 설계 및 훈련시킵니다. 이후 테스트 세트로 성능을 평가하고, 외부에서 준비한 강아지 사진(`dog.jpg`)을 모델에 입력하여 클래스를 정확히 예측하는지 확인합니다.

### 🖼️ 중간 및 최종 결과물

* **CNN 모델 학습 및 외부 이미지(dog.jpg) 예측 결과:**
 <img width="1360" height="643" alt="2번과제 결과" src="https://github.com/user-attachments/assets/aa1cb160-3f69-43c7-9f65-d8066de9b487" />


### 💻 소스 코드

```python
import tensorflow as tf # 텐서플로우 라이브러리를 tf 별칭으로 불러옵니다.
from tensorflow.keras.datasets import cifar10 # 케라스 데이터셋에서 10가지 사물 이미지인 cifar10 데이터를 가져옵니다.
from tensorflow.keras.models import Sequential # 레이어를 순서대로 쌓기 위한 Sequential 클래스를 불러옵니다.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # CNN 구성을 위한 합성곱, 풀링, 평탄화, 완전 연결 레이어를 불러옵니다.
from tensorflow.keras.preprocessing import image # 이미지 파일 로드 및 처리를 위한 image 모듈을 불러옵니다.
import numpy as np # 행렬 및 배열 연산을 위해 넘파이 라이브러리를 np 별칭으로 불러옵니다.

# CIFAR-10 클래스 이름 (예측 결과 출력을 위해 정의)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', # 인덱스 0~4에 해당하는 클래스 이름(비행기, 자동차, 새, 고양이, 사슴)을 리스트에 저장합니다.
               'dog', 'frog', 'horse', 'ship', 'truck'] # 인덱스 5~9에 해당하는 클래스 이름(개, 개구리, 말, 배, 트럭)을 리스트에 저장합니다.

# 1. CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # CIFAR-10 데이터를 다운로드하여 훈련용 및 테스트용 데이터와 레이블로 분리합니다.

# 2. 데이터 전처리: 픽셀 값을 0~1 범위로 정규화
x_train = x_train / 255.0 # 훈련 이미지의 모든 픽셀 값을 255로 나누어 0~1 사이의 값으로 스케일링합니다.
x_test = x_test / 255.0 # 테스트 이미지의 모든 픽셀 값도 255로 나누어 0~1 사이의 값으로 스케일링합니다.

# 3. CNN 모델 설계
model = Sequential([ # CNN 레이어를 순차적으로 쌓을 Sequential 모델을 생성합니다.
    # 첫 번째 합성곱 층 및 풀링 층
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # 3x3 크기의 필터 32개를 사용하고, 입력 크기가 32x32 컬러(채널 3)인 첫 번째 합성곱 층을 추가합니다.
    MaxPooling2D((2, 2)), # 2x2 크기의 영역에서 가장 큰 값만 추출하여 이미지 크기를 줄이는 첫 번째 풀링 층을 추가합니다.
    
    # 두 번째 합성곱 층 및 풀링 층
    Conv2D(64, (3, 3), activation='relu'), # 3x3 크기의 필터 64개를 사용하는 두 번째 합성곱 층을 추가합니다.
    MaxPooling2D((2, 2)), # 2x2 크기로 이미지의 공간 차원을 절반으로 줄이는 두 번째 풀링 층을 추가합니다.
    
    # 세 번째 합성곱 층
    Conv2D(64, (3, 3), activation='relu'), # 3x3 크기의 필터 64개를 사용하는 세 번째 합성곱 층을 추가합니다.
    
    # 1차원으로 펼친 후 완전 연결 층(Dense)으로 분류
    Flatten(), # 다차원 특징 맵(Feature Map) 배열을 완전 연결 층에 넣기 위해 1차원으로 평탄화합니다.
    Dense(64, activation='relu'), # 노드 64개와 ReLU 활성화 함수를 가진 완전 연결 은닉층을 추가합니다.
    Dense(10, activation='softmax') # 10개의 클래스에 대한 예측 확률을 출력하기 위해 소프트맥스 함수를 사용하는 최종 출력층을 추가합니다.
]) # Sequential 모델 객체 생성을 완료합니다.

# 모델 컴파일
model.compile(optimizer='adam', # 가중치 최적화 방법으로 adam 알고리즘을 사용하도록 지정합니다.
              loss='sparse_categorical_crossentropy', # 다중 분류의 손실을 계산하는 함수를 설정합니다.
              metrics=['accuracy']) # 모델 평가 기준으로 정확도를 사용하도록 설정합니다.

# 4. 모델 훈련
print("--- CNN 모델 훈련 시작 ---") # 모델 훈련 시작을 알리는 안내 문구를 화면에 출력합니다.
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test)) # 훈련 데이터를 10번 반복 학습하며, 매 에포크마다 테스트 데이터로 검증을 수행합니다.

# 5. 모델 성능 평가
print("\n--- 모델 성능 평가 ---") # 성능 평가가 시작됨을 화면에 출력합니다. (앞에 \n을 넣어 줄바꿈을 합니다)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2) # 테스트 데이터를 모델에 통과시켜 최종 손실값과 정확도를 계산합니다.
print(f'테스트 정확도: {test_acc:.4f}') # 최종 계산된 테스트 정확도를 소수점 아래 4자리까지 출력합니다.

# 6. 테스트 이미지(dog.jpg) 예측 수행
print("\n--- 외부 이미지(dog.jpg) 예측 ---") # 외부 이미지 예측 과정 시작을 알리는 문구를 출력합니다.
img_path = 'dog.jpg' # 예측해 볼 외부 이미지 파일의 이름과 경로를 문자열 변수로 지정합니다.

try: # 파일이 없을 경우를 대비하여 예외 처리를 시작합니다.
    # 이미지를 불러오고 CIFAR-10 모델 입력 크기(32x32)에 맞게 조정
    img = image.load_img(img_path, target_size=(32, 32)) # 지정한 경로의 이미지를 불러오며, 모델의 입력 크기에 맞게 32x32 사이즈로 크기를 조정합니다.
    img_array = image.img_to_array(img) # 불러온 이미지 객체를 다차원 넘파이 배열(Numpy Array) 형태로 변환합니다.
    
    # 모델은 배치(batch) 단위로 예측하므로 차원 확장
    img_array = np.expand_dims(img_array, axis=0) # 케라스 모델은 배치 단위 처리를 기본으로 하므로 배열의 첫 번째 차원에 배치 차원(1)을 추가합니다.
    img_array = img_array / 255.0 # 불러온 이미지의 픽셀 값 역시 훈련 데이터와 동일하게 255로 나누어 정규화합니다.
    
    # 예측 수행
    predictions = model.predict(img_array) # 전처리가 완료된 이미지 배열을 모델에 입력하여 예측 확률값들을 계산합니다.
    predicted_class = np.argmax(predictions) # 반환된 예측 확률 배열 중 가장 값이 큰(확률이 가장 높은) 인덱스를 추출합니다.
    
    print(f"예측 결과: 해당 이미지는 '{class_names[predicted_class]}'일 확률이 가장 높습니다.") # 예측된 인덱스에 해당하는 클래스 이름을 찾아 화면에 결과로 출력합니다.
    
except FileNotFoundError: # 만약 'dog.jpg' 파일을 찾지 못해 에러가 발생한 경우 아래 블록을 실행합니다.
    print(f"오류: '{img_path}' 파일을 찾을 수 없습니다.") # 파일을 찾을 수 없다는 안내 문구를 출력하고 프로그램을 종료합니다.
```

### 💡 핵심 함수 정리

* `Conv2D(filters, kernel_size, ...)`: 2차원 이미지 위를 슬라이딩하며 공간적인 특징(Feature Map)을 추출하는 합성곱(Convolution) 층을 생성합니다.
* `MaxPooling2D(pool_size)`: 추출된 특징 맵을 다운샘플링하여 지배적인 특징만 남기고 연산량을 줄여주는 풀링(Pooling) 층을 생성합니다.
* `model.predict(x)`: 학습이 완료된 신경망 모델에 새로운 입력 데이터 `x`를 전달하여 각 클래스에 속할 확률(예측값)을 배열 형태로 반환합니다.
