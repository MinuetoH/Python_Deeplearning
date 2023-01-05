# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:02:50 2023

@author: kimyh
"""

#######################################################
#  CNN : Convolutional Neural Network : 합성곱층
#        Conv2D(컨볼루션층,합성곱층) 층으로 표현
#
#  Dense 층 : 완전연결층. 밀집층
#         - 1차원 배열의 형태로 변형하여 학습.
#         - 이미지 분석시 색상의 관계 표현 못함 
#         - 3차원형태의 이미지를 1차원형태로 분석함. 
#           => 이미지 특성부분을 분석하기 어려움 
########################################################
#Fashion MNIST with CNN
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()
train_x.shape #(60000, 28, 28)
test_x.shape   #(10000, 28, 28)
train_y.shape  #(60000,)
test_y.shape   #(10000,)
train_x = train_x/255.0  #정규화
test_x = test_x/255.0    #정규화
train_x = train_x.reshape(-1,28,28,1)  #4차원배열로 변경
test_x = test_x.reshape(-1,28,28,1)    #4차원배열로 변경
train_x.shape
test_x.shape

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
for c in range(16):
   plt.subplot(4,4,c+1)
   plt.imshow(train_x[c].reshape(28,28), cmap='gray')
plt.show()
#모델 생성
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense
model = Sequential([
   Conv2D(input_shape=(28,28,1), kernel_size=(3,3),filters=3),
   Conv2D(kernel_size=(3,3), filters=32),
   Conv2D(kernel_size=(3,3), filters=64),
   Flatten(),    #1차원형태로 변경
   Dense(units=128, activation='relu'),
   Dense(units=10, activation='softmax')    #출력층
 ])
train_y[:10]
#sparse_categorical_crossentropy : 레이블을 one-hot 인코딩 불필요
model.compile(optimizer="adam", 
  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
'''
              입력값  필터  편향
  Conv2D-1 : (3*3*1) * 3  + 3  = 30
  Conv2D-2 : (3*3*3) * 32 + 32 = 896
  Conv2D-3 : (3*3*32) * 64 + 64 = 18496
  flatten_1 : 22*22*64 = 30976
  Dense_1  : (30976 + 1) + 128 = 3965056
  Dense_2  : (128 + 1) * 10 = 1290
'''
train_y[0]
train_x.shape
history = model.fit(train_x, train_y, epochs=5, \
                validation_split=0.25,batch_size=128)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()
plt.show()

#평가하기
model.evaluate(test_x, test_y) # [0.4205720126628876, 0.8658000230789185]



#풀링층, 드랍아웃층추가하기
#
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense
from tensorflow.keras.layers import MaxPool2D,Dropout
model = Sequential([
   Conv2D(input_shape=(28,28,1), kernel_size=(3,3),filters=3),
   #strides=(2,2) : 
   #MaxPool2D 입력구조 : 26,26,3  
   #MaxPool2D 출력구조 : 13,13,3    
   MaxPool2D(strides=(2,2)), #max풀링층 : 특성맵을 최대값으로 이루어진 맵
   Conv2D(kernel_size=(3,3), filters=32),
   MaxPool2D(strides=(2,2)),
   Conv2D(kernel_size=(3,3), filters=64),
   Flatten(),
   Dense(units=128, activation='relu'),
   Dropout(rate=0.3), #드랍아웃층 : 학습제외 비율. 30%정도 제외
   Dense(units=10, activation='softmax')    
 ])
model.summary()
'''
  1. https://graphviz.org/download/ 사이트 접속
     graphviz-2.49.3 (64-bit) EXE installer [sha256]
     다운 받고 설치하기
  2. 환경 설정  
    - Graphviz : C:\Program Files\Graphviz\bin\dot.exe
    - path 설정 : C:\Program Files\Graphviz\bin 추가
  3. anaconda prompt   
     pip install pyparsing
     pip install graphviz
     pip install pydot
     conda install graphviz => 콘다 upgrade 필요
  4. spyder 재실행     
'''
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)

#학습하기
model.compile(optimizer="adam", 
  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=5, \
                validation_split=0.25,batch_size=128)
#그래프상 과적합 해소.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()
plt.show()
# [0.4205720126628876, 0.8658000230789185]
model.evaluate(test_x, test_y) #[0.353180855512619, 0.8712000250816345]

#데이터 증식 방법
# 비슷한 이미지를 생성.
from tensorflow.keras.preprocessing.image \
    import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
#ImageDataGenerator : 이미지 데이터 변형 해주는 객체
train_datagen = ImageDataGenerator(\
           horizontal_flip = True, #수평방향 뒤집기
           vertical_flip = True,   #수직방향 뒤집기
           shear_range=0.5,        #시계방향으로 이미지 밀기
           brightness_range=[0.5,1.0], #이미지 밝기 변경
           zoom_range=0.2,    #이미지 크기 확대. 확대/축소
           width_shift_range=0.1, #가로방향으로 이미지 이동
           height_shift_range=0.1,#세로방향으로 이미지 이동
           rotation_range=30,  #이미지 회줜
           fill_mode='nearest' #이미지변환시 픽셀 채우는 방법
           )
    
import tensorflow as tf
image_path = tf.keras.utils.get_file\
     ('cat.jpg','http://bit.ly/33U6mH9')
image=plt.imread(image_path)
image.shape
image = image.reshape((1,) + image.shape)
image.shape
#batch_size=1 : 1개씩 이미지 생성.
train_generator = train_datagen.flow(image,batch_size=1)
fig = plt.figure(figsize=(5,5))
fig.suptitle("augmented image")
for i in range(25) : 
    #train_generator에 설정된 이미지를 전달
   data = next(train_generator)
   image = data[0] #이미지데이터
   plt.subplot(5, 5, i+1)
   plt.xticks([]) #x축 제거
   plt.yticks([]) #y축 제거
   plt.imshow(np.array(image,dtype=np.uint8))    
plt.show()
########################
# 1. 캐글 데이터셋 다운받기
#    https://www.kaggle.com/trolukovich/apparel-images-dataset
# 2. archive.zip 파일을 현재 폴더의 clothes_dataset 폴더에
#    압축 풀기
# 3. 다중레이블 데이터 : 폴더의 이름이 레이블 
#      빨강  파랑   신발   드레스 
#        1    0     1      0
#        0    1     0      1
#       활성화 함수 : sigmoid
#       손실함수    : binary_crossentropy
# 4. 다중클래스 데이터 : 다중컬럼데이터
#      신발  가방  드레스 
#       1     0     0
#       0     1     0 
#       활성화 함수 : softmax
#       손실함수 : categorical_crossentropy
