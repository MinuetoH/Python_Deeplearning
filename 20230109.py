# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:42:31 2023

@author: kimyh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import konlpy
from konlpy.tag import Okt
import time
#### 네이버 영화 리뷰 데이터 분석하기
train_file = tf.keras.utils.get_file('ratings_train.txt', \
origin='https://raw.githubusercontent.com/e9t/nsmc/master/\
    ratings_train.txt',extract=True)
train_file    #다운 받은 파일의 위치 
#sep='\t' : 셀구분자각 탭(\t)
train = pd.read_csv(train_file, sep='\t')
train.info()
#실습을 위해서 데이터 줄이기 : 150000건 => 15000건
train = train.iloc[:15000]
train.shape
train.info()
#레이블 별 데이터 건수
train["label"].value_counts()
#한글 형태소 분석
okt = Okt()
#데이터 전처리
#document 컬럼의 내용 중 한글, 영문, 공백을
#  제외한 모든 문자들 제거하기
train["document"] = \
train["document"].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
train["document"].head()

#제외되는 단어들 제외
def word_tokenization(text): 
  stop_words = ["는", "을", "를", '이', '가', '의', '던', \
    '고', '하', '다', '은', '에', '들', '지', '게', '도'] 
  #okt.morphs(text) : text 내용을 형태소 분석하여 단어들 추출   
  #                   품사는 표시안함
  return [word for word in okt.morphs(text) \
          if word not in stop_words]

start = time.time()
#x : train['document'] 데이터 한개.
data = train['document'].apply((lambda x: word_tokenization(x)))
print("실행시간:",time.time()-start)
data.head()
#data 정보를 토큰화 하기
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence \
    import pad_sequences
oov_tok="<OOV>" #기타 토큰
vocab_size=15000
tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(data) #토큰화

print("총 단어 갯수 : ",len(tokenizer.word_index)) #27072
tokenizer.word_index #단어:토큰인덱스
tokenizer.word_counts #단어:갯수
list(tokenizer.word_counts.items())[:10]
list(tokenizer.word_counts.values())[:10]
len(tokenizer.word_counts.values())
#5회이상 사용된 단어 갯수
cnt = 0
for x in tokenizer.word_counts.values() :
    if x >= 5 :  #단어의 사용 갯수
        cnt += 1
print("5회이상 사용된 단어:",cnt) #4350
len(data)
#훈련 데이터(12000건), 검증 데이터(3000건) 분리 
train_size=12000
train_data = data[:train_size]
valid_data = data[train_size:]
len(train_data)
len(valid_data)
#label
train_y = train['label'][:train_size]
valid_y = train['label'][train_size:]
train_y[:10]
valid_y[:10]
train_data[:10]

#훈련/검증 데이터 토큰화 하기
train_data = tokenizer.texts_to_sequences(train_data)
valid_data = tokenizer.texts_to_sequences(valid_data)
train_data[:10]
valid_data[:10]
#문장의 단어의 최대 갯수
max(len(x) for x in valid_data) #검증 데이터 최대 단어 갯수 :63
max_len = max(len(x) for x in train_data)
print("문장 최대 길이:",max_len)  #훈련데이터의 최대 단어 갯수 : 59
#최대 단어 갯수로 패딩
train_pad = pad_sequences(train_data,padding="post", maxlen=max_len)
valid_pad = pad_sequences(valid_data,padding="post", maxlen=max_len)
len(train_pad[0]) #59
len(train_pad[1]) #59
train_pad[0]

#모델 구성하기
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers \
    import Dense, LSTM, Embedding, Bidirectional
vocab_size #15000
def create_model():
    model = Sequential([
            Embedding(vocab_size, 32), #RNN의 첫번째 층. 
            #LSTM(Long Short Term Memory)
            #return_sequences=True : 상태값 전달.
            #Bidirectional : 양방향 RNN
            Bidirectional(LSTM(32, return_sequences=True)),    
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
    ])
    model.compile\
    (loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model
model = create_model()
model.summary()
#콜백함수 설정하기
checkpoint_path = 'best_performed_model.ckpt' #파일명
'''
딥러닝 모델의 구조가 복잡하고, 데이터가 크면 학습시간이 길어짐
=> 오랜시간 학습한 모델의 값을 저장
ModelCheckpoint : 가중치값을 저장 
 save_weights_only=True : 가중치만 저장함.
 save_best_only=True    : 좋은 가중치값만 저장
 monitor='val_loss'    : 평가기준. 좋은 가중치 판단 기준.
                         검증데이터의 손실함수값
'''
checkpoint = tf.keras.callbacks.ModelCheckpoint\
            (checkpoint_path, #저장 파일명
             save_weights_only=True, 
             save_best_only=True, 
             monitor='val_loss', #평가기준
             verbose=1)
'''
  학습시 성능이 개선되지 않을 경우 학습을 중단
  monitor='val_loss' : 성능 판단기준.
  patience=2 : 2번 epochs 까지 개선되지 않으면 학습 중단. 
'''            
early_stop = tf.keras.callbacks.EarlyStopping\
            (monitor='val_loss', patience=2)
'''
콜백함수 : 호출된 함수에서 호출하는 함수
callbacks=[early_stop, checkpoint] : 
'''
history = model.fit(train_pad, train_y, 
                validation_data=(valid_pad, valid_y), 
                callbacks=[early_stop, checkpoint], #콜백함수 등록 
                batch_size=64, epochs=10, verbose=1)

def plot_graphs(history, metric):
  plt.figure()  
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
  
plot_graphs(history, 'acc') #정확도 그래프
plot_graphs(history, 'loss') #손실함수 그래프

'''
1. console 초기화
2. test 데이터 읽기.속도를 위해 1000건의 데이터만 처리
'''
import tensorflow as tf
import pandas as pd
test_file = tf.keras.utils.get_file('ratings_test.txt',\
 origin='https://raw.githubusercontent.com\
/e9t/nsmc/master/ratings_test.txt', extract=True)
test = pd.read_csv(test_file, sep='\t')
test.shape
test = test.iloc[:1000] 
test.shape
test.info()
# 테스트 데이터 전처리
# - 한글,영문,공백을 제외한 모든 문자 제거
# - 결측값 제거 : 행 제거
# - 테스트 데이터 레이블 부분 분리
# - 테스트 데이터 불용어 제거 (형태소분석)
# - tokenizer를 이용하여 분석 가능한 데이터로 변경
# - 패딩하기. 59
import konlpy
from konlpy.tag import Okt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
okt = Okt()
def word_tokenization(text):
  stop_words = ["는", "을", "를", '이', '가', '의', '던', \
    '고', '하', '다', '은', '에', '들', '지', '게', '도'] 
  return [word for word in okt.morphs(text) \
          if word not in stop_words]

#토큰화를 위한 Train 데이터 읽기
train_file = tf.keras.utils.get_file('ratings_train.txt', \
origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',\
    extract=True)
train = pd.read_csv(train_file, sep='\t')
train = train.iloc[:15000]
train['document'] = train['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
data = train['document'].apply((lambda x: word_tokenization(x)))
oov_tok = "<OOV>"
vocab_size = 15000
tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(data)
data[:3]

#테스트 데이터 전처리 함수 
def preprocessing(df):
  #영문,한글,공백 이외 문자 제거  
  df['document'] = df['document'].str.replace\
      ("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
  #결측값 행을 제거    
  df = df.dropna() 
  #라벨 분리
  test_label = df['label']
  #형태소 분리. 불용어 제거
  test_data = df['document'].apply((lambda x: word_tokenization(x))) 
  #tokenizer를 이용하여 분석 가능한 데이터로 변경
  test_data = tokenizer.texts_to_sequences(test_data)
  #패딩하기
  test_data = pad_sequences(test_data, padding="post", maxlen=59)
  return test_data, test_label

test_data, test_label = preprocessing(test)
len(test_data)
test_data[0]
test_label[:10]
len(test_data[0])
len(test_label)
test_data[0]
#3. 저장된 가중치 읽어서 모델에 적용
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import \
    Dense, LSTM, Embedding, Bidirectional
vocab_size
def create_model():
    model = Sequential([
            Embedding(vocab_size, 32), #첫번째 층 
            Bidirectional(LSTM(32, return_sequences=True)),    
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', \
                  optimizer='adam', metrics=['acc'])
    return model
model = create_model()
model.summary()
model.evaluate(test_data,test_label) #학습안된 상태의 평가
checkpoint_path = 'best_performed_model.ckpt' #파일명
#저장된 가중치 값을 로드
model.load_weights(checkpoint_path)
model.evaluate(test_data,test_label) #가중치값을 로드한 후 평가

mydata = pd.DataFrame({'id':[1,2],
                  "document":["영화 재밌어요",'영화 재미없어요'],
                  'label':[1,0]})
X, Y = preprocessing(mydata)
X
model.evaluate(X,Y)
pred= model.predict(X)
pred[0]
pred[1]
np.mean(pred[0])
np.mean(pred[1])
