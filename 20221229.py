# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:18:23 2022

@author: kimyh
"""
'''
    인공신경망(ANN)
        단위 : 퍼셉트론
        
        y = x1w1 + x2w2 + b
        x1,x2 : 입력값, 입력층
        y : 결과값.
        w : 가중치.
        b : 편항. 
'''
import numpy as np
def AND(x1, x2) :
    x = np.array([x1,x2])     #입력값
    w = np.array([0.5,0.5])   #가중치
    b = -0.8                  #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1

for xs in [(0,0),(0,1),(1,0),(1,1),] :
    y = AND(xs[0],xs[1])
    print(xs,"=>",y)
    
#퍼셉트론 OR 게이트 구현
def OR(x1, x2) :
    x = np.array([x1,x2])     #입력값
    w = np.array([0.5,0.5])   #가중치
    b = -0.2                  #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1

for xs in [(0,0),(0,1),(1,0),(1,1),] :
    y = OR(xs[0],xs[1])
    print(xs,"=>",y)
    
#퍼셉트론 NAND 게이트 구현
def NAND(x1, x2) :
    x = np.array([x1,x2])     #입력값
    w = np.array([-0.5,-0.5])   #가중치
    b = 0.8                  #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1

for xs in [(0,0),(0,1),(1,0),(1,1),] :
    y = NAND(xs[0],xs[1])
    print(xs,"=>",y)

'''
    퍼셉트론을 이용하여 XOR 게이트 구현
    단일신경망으로 구현 안됨
    다중신경망으로 구현해야 함
    단일퍼셉트론 : 입력층 - 출력층
    다중퍼셉트론 : 입력층 - 은닉층 - 출력층
'''
def XOR(x1,x2) :
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

for xs in [(0,0),(0,1),(1,0),(1,1),] :
    y = XOR(xs[0],xs[1])
    print(xs,"=>",y)

'''
    Tensorflow 설치
        1. https://www.microsoft.com/ko-kr 연결
        2. 다운로드 센터
        3. 개발자 도구
        4. 05 Visual Studio 2015용 Visual C++ 재배포 가능 패키지 다운
        5. vc_redist.x64.exe 선택 => 다음 => 다운
        6. 파일탐색기에서 vs_redist.x64.exe 실행
        7. anaconda prompt를 관리자로 실행
        8. pip install tensorflow
           tensorflow 버전 확인
           tensorflow 1.*
           tensorflow 2.*
           1, 2버전 사이에 호환이 안됨.

'''
import tensorflow as tf
print(tf.__version__) #2.11.0
#현재 컴퓨터가 GPU?
tf.config.list_physical_devices("GPU")
#[] 로 출력되면 GPU 환경 아님

import pandas as pd
pd.__version__
import numpy as np
np.__version__

#tensorflow를 이용한 AND/OR 게이트 구현
data = np.array([[0,0],[0,1],[1,0],[1,1]])
#label = np.array([[0],[0],[0],[1]]) #결과데이터 AND
#label = np.array([[0],[1],[1],[1]]) #결과데이터 OR
label = np.array([[0],[1],[1],[0]]) #결과데이터 XOR


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

model = Sequential() #딥러닝 모델.
'''
    Dense : 밀집층
      1   : 출력값의 갯수
    input_shape : 입력값의 갯수
    activation : 활성화 함수 알고리즘
        linear : 선형함수
        sigmoid : 0 ~ 1 사이의 값 변형
        relu : 양수인 경우 선형 함수, 음수인 경우 0
'''
model.add(Dense(1, input_shape=(2,), activation='linear'))
'''
    compile : 모델 설정. 모형 설정. 가중치 찾는 방법 설정
        optimizer=SGD() : 경사하강법 알고리즘 설정.
        loss=mse : 손실함수. mse : 평균제곱오차.
                   mse 값이 가장 적은 경우의 가중치와 평향 구함.
        metrics=['acc'] : 평가 방법 지정. acc:정확도
        => 손실함수의 값은 적은 값. 정확도는 1에 가까운 가중치와
           편향의 값을 찾도록 설정
'''
model.compile(optimizer=SGD(), loss=mse, metrics=['acc'])
# 학습하기
'''
    data : 훈련데이터.
    label : 정담.
    epochs=300 : 300번 반복학습. 손실함수가 적고, 정확도가 높아지도록
    verbose=0 : 학습과정 출력 생략
    verbose=1 : 학습과정 상세 출력
    verbose=2 : 학습과정 간략히 출
'''
model.fit(data, label, epochs=300, verbose=0)

print(model.get_weights())
print(model.predict(data))
print(model.evaluate(data, label))
