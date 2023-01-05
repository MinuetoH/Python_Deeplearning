# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:03:26 2023

@author: kimyh
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import glob as glob
'''
 glob.glob() : 파일의 목록을 리스트로 리턴 
 './clothes_dataset/*/*.jpg' : 현재폴터의 clothes_dataset 폴더의 모든 하위폴더
                               jpg 파일의 목록
 recursive=True  : 지정된 폴더의 하위 폴더 까지 검색
'''
all_data = np.array(glob.glob('./clothes_dataset/*/*.jpg', recursive=True))
len(all_data)  #11385
all_data.shape #(11385,)
all_data[:5]

#다중레이블 생성을 위한 함수.
def check_cc(color, clothes): #black,dress
#np.zeros(11,) : 요소의 갯수가 11개이고 값이 0인 1차원배열 생성    
    labels = np.zeros(11,)
    if(color == 'black'):       labels[0] = 1
    elif(color == 'blue'):      labels[1] = 1
    elif(color == 'brown'):     labels[2] = 1
    elif(color == 'green'):     labels[3] = 1
    elif(color == 'red'):       labels[4] = 1
    elif(color == 'white'):     labels[5] = 1
    #의류종류 설정
    if(clothes == 'dress'):     labels[6] = 1
    elif(clothes == 'shirt'):   labels[7] = 1
    elif(clothes == 'pants'):   labels[8] = 1
    elif(clothes == 'shorts'):  labels[9] = 1
    elif(clothes == 'shoes'):   labels[10] = 1
    return labels  #[10000010000]
#배열로 설정
all_labels = np.empty((all_data.shape[0], 11))
all_labels.shape # (11385, 11)
all_labels[0]
for i, data in enumerate(all_data):
    #i:인덱스
    #data : 이미지이름
    #./clothes_dataset\\black_dress\\0097960878307e559459d98c9f9eaeeea0db1f94.jpg
    color_and_clothes = all_data[i].split('\\')[1].split('_')
    color = color_and_clothes[0]  #black
    clothes = color_and_clothes[1] #dress
    labels = check_cc(color, clothes)
    all_labels[i] = labels; 
print(all_labels[-10:])     

#훈련데이터 테스트 데이터 분리
from sklearn.model_selection import train_test_split
#shuffle = True : 섞어서 분리.
#test_size = 0.3 : 훈련:7,테스트:3 비율
train_x, test_x, train_y, test_y = train_test_split\
(all_data, all_labels, shuffle = True, test_size = 0.3,random_state = 99)
train_x.shape  #(7969,)
test_x.shape   #(3416,)
train_y.shape  #(7969, 11)
test_y.shape   # (3416, 11)
#훈련데이터 검증데이터 분리
train_x, val_x, train_y, val_y = train_test_split\
(train_x, train_y, shuffle = True, test_size = 0.3,random_state = 99)
train_x.shape  # (5578,)
val_x.shape    # (2391,)
train_y.shape  # (5578, 11)
val_y.shape    # (2391, 11)

#csv 파일을 위한 DataFrame 생성
train_df = pd.DataFrame(
{'image':train_x, 'black':train_y[:, 0], 'blue':train_y[:, 1],
'brown':train_y[:, 2], 'green':train_y[:, 3], 'red':train_y[:, 4],
'white':train_y[:, 5], 'dress':train_y[:, 6], 'shirt':train_y[:, 7],
'pants':train_y[:, 8], 'shorts':train_y[:, 9], 'shoes':train_y[:, 10]})
train_df.info()
val_df = pd.DataFrame(
{'image':val_x, 'black':val_y[:, 0], 'blue':val_y[:, 1],
'brown':val_y[:, 2], 'green':val_y[:, 3], 'red':val_y[:, 4],
'white':val_y[:, 5], 'dress':val_y[:, 6], 'shirt':val_y[:, 7],
'pants':val_y[:, 8], 'shorts':val_y[:, 9], 'shoes':val_y[:, 10]})
val_df.info()
test_df = pd.DataFrame(
{'image':test_x, 'black':test_y[:, 0], 'blue':test_y[:, 1],
'brown':test_y[:, 2], 'green':test_y[:, 3], 'red':test_y[:, 4],
'white':test_y[:, 5], 'dress':test_y[:, 6], 'shirt':test_y[:, 7],
'pants':test_y[:, 8], 'shorts':test_y[:, 9], 'shoes':test_y[:, 10]})
test_df.info()
train_df.to_csv("./clothes_dataset/train.csv",index=None)
val_df.to_csv("./clothes_dataset/val.csv",index=None)
test_df.to_csv("./clothes_dataset/test.csv",index=None)
#########################################################
# 저장된 파일을 읽어서 데이터 분석하기
# console restart
import pandas as pd
train_df = pd.read_csv("./clothes_dataset/train.csv")
train_df.info()
val_df = pd.read_csv("./clothes_dataset/val.csv")
val_df.info()
test_df = pd.read_csv("./clothes_dataset/test.csv")
test_df.info()
#원본이미지 출력
import matplotlib.pyplot as plt
img1 = plt.imread(train_df["image"][0])
img1.shape  #(474, 474, 3)
plt.imshow(img1)
img2 = plt.imread(train_df["image"][1])
img2.shape #(494, 474, 3)
plt.imshow(img2)
img3 = plt.imread(train_df["image"][2])
img3.shape #(640, 474, 3)
plt.imshow(img3)

#1. 분석을 위한 이미지 크기를 동일하게 설정
#2. 5578개의 이미지를 메모리에 로드하여 분석하기 어려움.
# => 이미지제너레이터 방식을 이용함
from tensorflow.keras.preprocessing.image \
                              import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255) #이미지데이터 정규화
val_datagen = ImageDataGenerator(rescale=1./255) #이미지데이터 정규화

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import \
                Dense, Flatten,Conv2D,MaxPool2D,Dropout
'''
input_shape=(112, 112, 3) : 학습데이터의 크기
padding='same' : 입력데이터의 크기와 특징맵의 크기를 동일하게 설정.
                 패딩 추가
MaxPool2D(pool_size=(2,2)) : (2,2) 중 최대값인 데이터만 특징맵으로 설정
              => 특징맵의 크기가 반으로 줌
Dropout(rate=0.5) : 학습 데이터 중 50%만 학습.
                   50%는 학습하지 않음.                 
'''                
model = Sequential([
    Conv2D(input_shape=(112, 112, 3),kernel_size=(3,3),
                   filters=32, padding='same', activation='relu'),
    Conv2D(kernel_size=(3,3), filters=64, padding='same',
                                               activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(rate=0.5), 
    Conv2D(kernel_size=(3,3), filters=128, 
           padding='same',activation='relu'),
    Conv2D(kernel_size=(3,3), filters=256, padding='valid',
           activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(rate=0.5),
    Flatten(), #평탄화층(레이어).1차원형태의 배열로 변경 
    Dense(units=512, activation='relu'),
    Dropout(rate=0.5),
    Dense(units=256, activation='relu'),
    Dropout(rate=0.5),
    Dense(units=11, activation='sigmoid') #출력층. 11개의 값
])                            
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)
model.compile(optimizer='adam',loss='binary_crossentropy', \
              metrics = ['acc'])
class_col = list(train_df.columns[1:])
class_col
batch_size = 32
#학습이미지 설정
train_generator = train_datagen.flow_from_dataframe(
             dataframe=train_df, #dataframe 객체
             #이미지폴더. image 컬럼에 폴더지정.현재는 필요 없음
             directory=None,  #이미지 폴더 설정 안함.
             x_col = 'image',  #분석이미지데이터의 값.  image 컬럼
             y_col = class_col,#레이블데이터. train_df 컬럼명 
             target_size = (112, 112),#생성될이미지의 크기 설정
             color_mode='rgb',  #색상지정. 컬러이미지
             class_mode='raw',  #레이블데이터의 자료형. 배열로 리턴. 
             batch_size=batch_size, #한번에 생성되는 이미지 갯수
             shuffle = True,  #이미지를 랜덤하게 리턴   
             seed=42          #랜덤시드 
)
#검증이미지 설정
val_generator = val_datagen.flow_from_dataframe(
      dataframe=val_df,        directory = None,
      x_col = 'image',         y_col = class_col,
      target_size = (112,112), color_mode='rgb',
      class_mode='raw',        batch_size=batch_size,
      shuffle=True   )

def get_steps(num_samples, batch_size):
   if (num_samples % batch_size) > 0 :
       return (num_samples // batch_size) + 1
   else :
       return num_samples // batch_size
#steps_per_epoch : 한번의 학습마다 진행되는 값
history = model.fit(train_generator, 
           steps_per_epoch=get_steps(len(train_df), batch_size),
           validation_data = val_generator, 
           validation_steps=get_steps(len(val_df), batch_size),
           epochs = 3)

import matplotlib.pyplot as plt
batch = next(train_generator)
image,level = batch[0],batch[1]
image.shape
level.shape
plt.imshow(image[0])
level[0]
# 학습 결과를 그래프로 출력하기
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], 'g-', label='accuracy')
plt.plot(history.history['val_acc'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#테스트 데이터를 이용하여 예측하기
test_datagen = ImageDataGenerator(rescale = 1./255) #정규화
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,     directory=None,
        x_col = 'image',       y_col = None, #레이블 없음.
        target_size = (112, 112),   color_mode='rgb',
        class_mode=None,   #레이블 없음  
        batch_size=batch_size, #32
        shuffle = False)
#예측하기
preds = model.predict(test_generator, steps = 32)
preds[0] #9.9783474e-01, 3.8876748e-04, 8.1772832e-03, ....
off = 16
do_preds = preds[off:off+8] #8개씩 이미지 조회
class_col
for i, pred in enumerate(do_preds):
    plt.subplot(2, 4, i + 1) #2행4열이미지
    prob = zip(class_col,list(pred)) #(black,9.9.... ),...
    #정렬하기. 예측된 결과값으로 내림차순 정렬
    #prob : 11개의 컬럼의 값중 값이 큰 2개의 데이터
    prob = sorted(list(prob),\
                  key=lambda z : z[1], reverse=True)[:2]
    image = plt.imread(test_df["image"][i+off])
    plt.imshow(image) 
    plt.title(f'{prob[0][0]}:\
{round(prob[0][1] * 100,2)}%\n{prob[1][0]}:{round(prob[1][1] *100,2)}%')
    plt.tight_layout()              
plt.show()

#현재 모델을 저장하기
model.save('cloths_model1.h5')

#######################################
from keras.models import load_model
m2= load_model('cloths_model.h5') 

pred2 = m2.predict(test_generator, steps = 32)
pred2
pred2[0]
off = 24
do_pred2 = pred2[off:off+8]
for i, pred in enumerate(do_pred2):
    plt.subplot(2, 4, i + 1)
    prob = zip(class_col,list(pred))
    prob = sorted(list(prob),key=lambda z : z[1], reverse=True)[:2]
    image = plt.imread(test_df["image"][i+off]) 
    plt.imshow(image) 
    plt.title(f'{prob[0][0]}:{round(prob[0][1] * 100,2)}%\n{prob[1][0]}:{round(prob[1][1] *100,2)}%')
    plt.tight_layout()              
plt.show()
