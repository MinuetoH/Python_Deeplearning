# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:06:00 2023

@author: kimyh
"""

###############################################
#  RNN (Recurrent Neural Network) : 순환신경망. 
#       음성인식,문장번역등에 사용.
###############################################
#SimpleRNN 
import tensorflow as tf
import numpy as np
#return_sequences=True : 순환설정
#activation='tanh' : -1 ~ 1사이의 값을 가짐
rnn1 = tf.keras.layers.SimpleRNN\
       (units=1,activation='tanh',return_sequences=True)
X=[] #[[[0],[0.1],[0.2],[0.3]],[0.1,0.2,0.3,0.4]...]
Y=[] #[0.4,0.5]
for i in range(6) : #0 ~ 5
    #lst = [0.1,0.2,0.3,0.4]
    lst = list(range(i,i+4)) #0  ~ 3
    X.append(list(map(lambda c:[c/10],lst)))
    Y.append((i+4)/10)
X
Y
X=np.array(X)  #배열로 변경
Y=np.array(Y)
for i in range(len(X)) :
    #np.squeeze : X[i] => 1차원 배열로 변경
    print(np.squeeze(X[i]),Y[i])
    
X.shape
model = tf.keras.Sequential([
 tf.keras.layers.SimpleRNN
   (units=10, return_sequences=False, input_shape=[4,1]),
 tf.keras.layers.Dense(1)
])    
#mse: 평균제곱오차
model.compile(optimizer='adam', loss='mse')
model.summary()
X
model.fit(X, Y, epochs=1000, verbose=0)
print(model.predict(X)) #학습데이터 예측
#학습되지 않은 데이터 
print(model.predict(np.array([[[0.6],[0.7],[0.8],[0.9]]])))
print(model.predict(np.array([[[-0.1],[0.0],[0.1],[0.2]]])))
#예측 못함
print(model.predict(np.array([[[1],[2],[3],[4]]])))

#####################
# 단어 분석
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
texts = ['You are the Best Thing','You are the Nice']
#Tokenizer : 토큰화 객체
#num_words = 10 : 10개의 토큰으로 분리.
#oov_token = '<OOV>' : 분석된 토큰에 없는 단어의 경우 대체되는 
#                      문자열
tokenizer = Tokenizer(num_words = 10, oov_token = '<OOV>')
#texts 내용을 토큰화.
tokenizer.fit_on_texts(texts)
#texts_to_sequences : texts데이터와 토큰인덱스를 매칭
sequences = tokenizer.texts_to_sequences(texts)
#sequences_to_matrix : texts 데이터를 토큰 인덱스의 위치값을 1로 설정
binary_results = tokenizer.sequences_to_matrix\
                          (sequences, mode = 'binary')
texts                          
print(tokenizer.word_index)
print(sequences)
'''
{'<OOV>': 1, 'you': 2, 'are': 3, 'the': 4, 'best': 5, 'thing': 6, 'nice': 7
    You are the Best Thing
  0. 0. 1. 1. 1. 1. 1. 0. 0. 0.
  
'''
print(binary_results)
'''
You are the One
2    3    4  1
'''
test_text = ['You are the One']
test_seq = tokenizer.texts_to_sequences(test_text)
print(test_seq)                          
#test_text 데이터를 이진매핑으로 출력하기
test_bin = tokenizer.sequences_to_matrix\
    (test_seq,mode="binary")
test_bin
'''
imdb 데이터 셋
 - 영화리뷰에 대한 데이터 5만개
 - 50%씩 긍정리뷰,부정리뷰 
 - 전처리가 완료 상태. -> 내용이 숫자로 변환됨. 
'''
from tensorflow.keras.datasets import imdb
num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
print(X_train.shape, X_test.shape)
print(X_train[0])
print(y_train[0])
print(len(X_train[0])) #단어의 갯수 218
print(len(X_train[1])) #단어의 갯수 189
list(imdb.get_word_index().items())[0] #('fawn', 34701)(단어,토큰인덱스)
#토큰인덱스의 값이 작은 경우 빈번한 단어임.
#key : 단어
#value : 토큰인덱스
imdb_get_word_index = {}
for key, value in imdb.get_word_index().items():
   imdb_get_word_index[value] = key #imdb_get_word_index[34701]=fawn
for i in range(1, 11): #1~10 토큰인덱스값 출력
   print('{} 번째로 가장 많이 쓰인 단어 = {}'.format\
         (i, imdb_get_word_index[i]))
       
#훈련데이터의 문장 길이의 평균, 중간값,최대값,최소값 출력하기
import numpy as np
#훈련데이터의 문장의 길이를 배열로 저장
lengths = np.array([len(x) for x in X_train])
lengths[:10]
#평균, 중간값,최대값,최소값
np.mean(lengths) #평균
np.median(lengths) #중간값
np.max(lengths) #최대값
np.min(lengths) #최소값
#단어의 갯수를 히스토그램으로 출력하기
import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel("length")
plt.ylabel("frequency")       

y_train[:10] #0:부정, 1:긍정
#딥러닝을 위해서는 데이터의 길이가 동일해야 함. 
# -> 패딩 작업이 필요함 : 데이터의 길이가 지정한 길이보다 작으면 0으로 채움.
#                        지정한 길이보다 크면, 지정길이로 잘라냄.
#패딩방법
from tensorflow.keras.preprocessing.sequence import pad_sequences
a1 = [[1,2,3]]
a2 = [[1,2,3,4,5,6,7,8]]
#maxlen : 지정길이.
# padding='pre'  : 앞쪽을 0으로 채움. 기본값
# padding='post' : 뒤쪽을 0으로 채움
a1_pre = pad_sequences(a1, maxlen=5) #3->5. 앞쪽 0으로 채움
a2_pre = pad_sequences(a2, maxlen=5) #8->5  앞쪽 3개잘라냄
print(a1_pre) # [0 0 1 2 3]
print(a2_pre) # [4 5 6 7 8]
#3->5. 뒤쪽 0으로 채움
a1_post = pad_sequences(a1, maxlen=5,padding='post') 
#8->5  앞쪽 3개잘라냄
a2_post = pad_sequences(a2, maxlen=5,padding='post') 
print(a1_post) # [1 2 3 0 0]
print(a2_post) # [4 5 6 7 8]

max_len = 500
pad_X_train = pad_sequences(X_train,maxlen=max_len,padding='pre')
len(pad_X_train[0])
len(pad_X_train[1])
pad_X_train[0]

#모델 설정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
'''
Embedding : RNN의 가장 기본층. 첫번째 층으로 사용됨.
input_dim=num_words : input 갯수
output_dim=32 : 출력 갯수
input_length=max_len : 토큰 갯수
'''
num_words #10000

model = Sequential()
model.add(Embedding(input_dim=num_words,output_dim=32,input_length=max_len))
model.add(Flatten()) 
model.add(Dense(1, activation = 'sigmoid')) #0 또는 1
model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['acc'])
model.summary()
history = model.fit(pad_X_train, y_train,batch_size = 32, 
                    epochs = 30, validation_split = 0.2)

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

#테스트 데이터로 평가하기
pad_X_test = pad_sequences(X_test,maxlen=max_len,padding='pre')
model.evaluate(pad_X_test, y_test)#[0.8257220387458801, 0.8691999912261963]

#### 네이버 영화 리뷰 데이터 분석하기
import konlpy
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

train_file = tf.keras.utils.get_file('ratings_train.txt',\
origin=\
  'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',\
  extract=True)
train_file  #파일의 위치
#train_file 파일을 데이터프레임으로 읽기
train = pd.read_csv(train_file, sep='\t')
train.info()
#label 별 갯수 조회하기
train.label.value_counts()
sns.countplot(x="label",data=train)
#결측값의 갯수 조회하기
train.isnull().sum()
#결측값 레코드 삭제하기
train = train.dropna()
train.info()
len(train.document[0])
len(train.document[1])
#label 0:부정, 1:긍정
#긍정의 글자수, 부정의 글자수를 히스토그램으로 출력하기
#긍정의 글자수목록
#train[train['label']==1] : label 컬럼의 값이 1인 레코드만 조회.
#train[train['label']==1]['document'] : label이 1인 document 컬럼의 값들
#p_len : label이 1인 document 컬럼의 값의 길이들
p_len = train[train['label']==1]['document'].str.len()
p_len[:10]
p_len.mean() #34.599959906448376
#부정의 글자수목록
n_len = train[train['label']==0]['document'].str.len()
n_len.mean() #35.80631901024345
#히스토그램으로 출력하기 => 차이가 없다
fig = plt.figure(figsize = (10, 5))
ax1 = plt.subplot(1,2, 1)
ax1.set_title("positive")
ax1.hist(p_len)
ax2 = plt.subplot(1,2, 2)
ax2.hist(n_len)
ax2.set_title("negative")
fig.suptitle("Number of characters")
plt.show()
# 한글 형태소 분석하기 
# document 컬럼의 내용 중 한글,영문, 공백을 제외하고 문자 제거
train["document"].head()
'''
[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ] : 대문자,소문자,한글,자음,모음이 아닌 글자
    ^   : not
    A-Z : 대문자
    a-z : 소문자
    가-힣 : 한글
    ㄱ-ㅎ : 한글자음
    ㅏ-ㅣ : 한글모음
     : 공백
    
'''
train["document"] = train["document"].str.replace\
    ("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
train["document"].head()

#한글 불용어 제거. 형태소 분석
okt = Okt()
def word_tokenization(text) : #text : 한개의 문장
    stop_words = ["는","을","를","이","가","의","던",\
                  "고","하","다","은","에","들","지","게","도"]
    #okt.morphs(text) : 형태소 분리.
    #word : 분리된 형태소 한개의 값     
    return [word for word in okt.morphs(text) if word not in stop_words]    

import time
start = time.time()
#x : 한개의 문장
data = train['document'].apply((lambda x: word_tokenization(x)))
print("실행시간:",time.time()-start) # 403.1246769428253
data.head()
    
#전체 단어의 갯수 출력
#data 정보를 토큰화 하기
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

print("총 단어 갯수 : ",len(tokenizer.word_index)) #102194
