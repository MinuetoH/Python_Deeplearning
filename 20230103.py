# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:05:45 2023

@author: kimyh
"""

from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape,x_test.shape) #(60000, 28, 28) (10000, 28, 28)
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat',
             'Sandal','Shirt','Sneaker','Bag','Ankle boot']

x_train = x_train/255 #minmax 정규화 (x-min)/(max-min)
x_test = x_test/255
# 정답. one-hot 인코딩 => 다중분류. 10가지
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 훈련데이터, 검증데이터 분리
x_train, x_val, y_train, y_val = \
    train_test_split(x_train,y_train,test_size=0.3,random_state=777)
    
model1 = Sequential() 
model1.add(Flatten(input_shape = (28,28))) #28*28=784 1차원배열 변경
model1.add(Dense(64,activation="relu"))
model1.add(Dense(32,activation="relu"))
model1.add(Dense(10,activation="softmax"))
model1.summary()
model1.compile(optimizer='adam',
               loss='categorical_crossentropy',metrics=['acc'])
history1 = model1.fit(x_train,y_train, epochs=30,
                      batch_size=128, validation_data=(x_val,y_val))
history1.history["loss"][29]      # 0.1902388036251068
history1.history["acc"][29]       # 0.9295714497566223
history1.history["val_loss"][29]  # 0.33574315905570984
history1.history["val_acc"][29]   # 0.8887777924537659
# 테스트데이터의 손실함수값, 정확도 출력하기
# [0.39121556282043457, 0.8755999803543091]
model1.evaluate(x_test,y_test)
# 예측하기
results = model1.predict(x_test)
np.argmax(results[:10],axis=-1)
np.argmax(y_test[:10],axis=-1)
# 평가하기
from sklearn.metrics import classification_report, confusion_matrix
# 혼동행렬 출력하기
cm = confusion_matrix(np.argmax(y_test,axis=-1), 
                      np.argmax(results,axis=-1))
cm
# 혼동행렬 heatmap으로 출력하기
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('predicted label',fontsize=15)
plt.ylabel('true label',fontsize=15)
plt.xticks(range(10), class_names, rotation=45) #x축의 값을 class_names 변경
plt.yticks(range(10), class_names, rotation=0) #y축의 값을 class_names 변경
plt.show()

# history1 데이터로 loss, acc - 훈련데이터,검증데이터부분을 선 그래프로 출력하기
import matplotlib.pyplot as plt
his_dict = history1.history
loss = his_dict['loss']  
val_loss = his_dict['val_loss'] 
epochs = range(1, len(loss) + 1) 
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1, 2, 1) #1행 2열의 1번째 그래프영역
ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
ax1.plot(epochs, val_loss, color='orange', label = 'val_loss')
ax1.set_title('train and val loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()
# 정확도 그래프
acc = his_dict['acc'] #훈련데이터 정확도 값
val_acc = his_dict['val_acc'] #검증데이터 정확도 값
ax2 = fig.add_subplot(1, 2, 2) #1행 2열의 2번째 그래프영역
ax2.plot(epochs, acc, color = 'blue', label = 'train_acc')
ax2.plot(epochs, val_acc, color='orange', label = 'val_acc')
ax2.set_title('train and val acc')
ax2.set_xlabel('epochs')
ax2.set_ylabel('acc')
ax2.legend()
plt.show()

# model2 구성하여 학습하기
# 256, 128개의 출력을 가지는 은닉층을 추가
model2 = Sequential() 
model2.add(Flatten(input_shape = (28,28))) #28*28=784 1차원배열 변경
model2.add(Dense(256,activation="relu"))  #256 출력 은닉층 추가
model2.add(Dense(128,activation="relu"))  #128 출력 은닉층 추가
model2.add(Dense(64,activation="relu"))
model2.add(Dense(32,activation="relu"))
model2.add(Dense(10,activation="softmax"))
model2.summary()
model2.compile(optimizer='adam',
               loss='categorical_crossentropy',metrics=['acc'])
history2 = model2.fit(x_train,y_train, epochs=30,
                      batch_size=128, validation_data=(x_val,y_val))
model1.evaluate(x_test,y_test) #[0.39121556282043457, 0.8755999803543091]
model2.evaluate(x_test,y_test) #[0.46417900919914246, 0.8765000104904175]
# 모델 저장하기
model1.save("fashion1.h5")
model2.save("fashion2.h5")
# 저장된 모델 로드
from keras.models import load_model
m1 = load_model("fashion1.h5")
m2 = load_model("fashion2.h5")
m1.evaluate(x_test,y_test) #[0.39121556282043457, 0.8755999803543091]
m2.evaluate(x_test,y_test) #[0.46417900919914246, 0.8765000104904175]

results = m1.predict(x_test)  #예측하기
results2 = m2.predict(x_test) #예측하기
np.argmax(results[:10],axis=-1)
np.argmax(results2[:10],axis=-1)
np.argmax(y_test[:10],axis=-1)

#############################################
# 이항분류 : 분류의 종류가 2종류인 경우
import pandas as pd
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
red = pd.read_csv(url+'winequality-red.csv',sep=';')  #red 와인 정보
white = pd.read_csv(url+'winequality-white.csv', sep=';') #white 와인 정보
red.info()   #1599
white.info() #4898
'''
1 - fixed acidity : 주석산농도
2 - volatile acidity : 아세트산농도
3 - citric acid : 구연산농도
4 - residual sugar : 잔류당분농도
5 - chlorides : 염화나트륨농도
6 - free sulfur dioxide : 유리 아황산 농도
7 - total sulfur dioxide : 총 아황산 농도
8 - density : 밀도
9 - pH : ph
10 - sulphates : 황산칼륨 농도
11 - alcohol : 알코올 도수
12 - quality (score between 0 and 10) : 와인등급
'''
# type 컬럼 추가
# red 와인인 경우 type 컬럼에 0, white와인인 경우 type 컬럼에 1로 저장하기
red["type"]=0
white["type"]=1
# red,white 데이터를 합하여 wine 데이터에 저장하기
wine = pd.concat([red,white])
wine.info() #6497
wine.head()
# wine 데이터를 minmax 정규화하여 wine_norm 데이터에 저장
wine.min()
wine.max()
wine_norm = (wine-wine.min()) / (wine.max()-wine.min())
wine_norm.head()
wine_norm.min()
wine_norm.max()
wine_norm["type"].head()
wine_norm["type"].tail()
# wine_norm 데이터를 섞어 wine_shuffle 데이터에 저장하기
import numpy as np
# sample() : 임의로 표본추출을 위한 함수.
# frac=1 : 표본추출의 비율. 1은 100%.
wine_shuffle = wine_norm.sample(frac=1)
wine_shuffle["type"].head()
wine_shuffle["type"].tail()
wine_shuffle.info()
# wine_shuffle 데이터를 배열데이터 wine_np로 저장
wine_np = wine_shuffle.to_numpy()
type(wine_np)
wine_np.shape
# train(8), test(2) 데이터 분리
# 설명변수, 목표변수(정답)로 분리
train_idx = int(len(wine_np)*0.8)
train_idx
train_x, train_y = \
    wine_np[:train_idx,:-1], wine_np[:train_idx,-1]
train_x.shape #(5197, 12)
train_y.shape #(5197,)
# 훈련데이터 분리
test_x, test_y = \
    wine_np[train_idx:,:-1], wine_np[train_idx:,-1]
test_x.shape #(1300, 12)
test_y.shape #(1300,)
# LABEL을 one-hot 인코딩하기
import tensorflow as tf
train_y = tf.keras.utils.to_categorical(train_y, num_classes=2)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=2)
train_y
test_y
# 모델 생성
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(units=48,activation='relu',input_shape=(12,)),
    Dense(units=24,activation='relu'),
    Dense(units=12,activation='relu'),
    Dense(units=2,activation='sigmoid')
    ])
model.summary()
# binary_crossentropy : 이중분류에서 사용되는 손실함수.
#                       레이블을 one-hot 인코딩 필요
model.compile(optimizer='adam',
               loss='binary_crossentropy',metrics=['accuracy'])
# validation_split=0.25 : 25%의 데이터를 검증데이터로 사용
history = model.fit(train_x, train_y, epochs=25,
                      batch_size=32, validation_split=0.25)
# 학습결과 시각화하기
# 학습데이터와 검증데이터의 loss,accuracy 값을 선그래프로 출력하기
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
# 'b-' : blue, 실선
plt.plot(history.history['loss'], 'b-', label='loss')
# 'r--' : red, 점선
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()
plt.show()
# 과적합 발생 안됨.
# 평가하기.
model.evaluate(test_x,test_y) #[0.04478975757956505, 0.9915384650230408]
# 예측하기
results = model.predict(test_x)
np.argmax(results[:10],axis=-1)
np.argmax(test_y[:10],axis=-1)
# 평가결과 출력하기 : 혼동행렬, heatmap 출력하기
# 혼동행렬(confusion_matrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(test_y,axis=-1), 
                      np.argmax(results,axis=-1))
cm
import seaborn as sns
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('predicted label',fontsize=15)
plt.ylabel('true label',fontsize=15)
plt.xticks(range(2), ['red','white'], rotation=45) 
plt.yticks(range(2), ['red','white'], rotation=0) 
plt.show()

from sklearn.metrics import classification_report
classification_report\
    (np.argmax(test_y,axis=-1),np.argmax(results,axis=-1))
# 컬러 이미지 분석하기
# 인터넷을 통해 이미지 조회하기
# 이미지 다운받기
image_path = tf.keras.utils.get_file('cat.jpg','http://bit.ly/33U6mH9')
image_path
image = plt.imread(image_path) #고양이 이미지의 배열값
image.shape #(241, 320, 3)
import cv2
cv2.imshow("cat",image)
plt.imshow(image)
# 색상별로 분리
bgr = cv2.split(image)
bgr[0].shape #빨강
bgr[1].shape #초록
bgr[2].shape #파랑
plt.imshow(bgr[0])
cv2.imshow("cat",bgr[0])
titles=['RGB','Red','Green','Blue']
# zeros_like : bgr[0] 구조와 같은 배열을 0 값으로 채워 생성
zero = np.zeros_like(bgr[0],dtype='uint8')
red = cv2.merge([bgr[0],zero,zero])
green = cv2.merge([zero,bgr[1],zero])
blue = cv2.merge([zero,zero,bgr[2]])
# images : 이미지배열의 리스트
images = [image,red,green,blue]
# 그래프영역을 1행 4열로 생성. 크기 13,3 설정
# axes : 4개의 영역들
fig, axes = plt.subplots(1, 4, figsize=(13,3))
objs = zip(axes, titles, images)
for ax, title, img in objs :
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks(())  #x축 없음
    ax.set_yticks(())  #y축 없음

