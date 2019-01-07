## 특징

1. brand와 maker은 특수문자를 기준으로 분해했고, model과 product는 형태소 기준으로 분해함
2. 분해된 텍스트를 통틀어서 많이 쓰인 단어들 N개가 example 별로 각각 몇 번 쓰였는지 세서 새로운 feature columns를 만듦
3. price가 제공되지 않은 곳은 median price로 채움
4. target은 b, m, s, d 별로 one hot encoding함
5. text feature N개와 img_feature 2048개를 각각 Dense Layer에 통과시키고, 이를 concatenate 한 뒤 다시 Dense Layer에 통과시킴. 그 결과를 카테고리 b, m, s, d 별로 별개의 Dense Layer with softmax activation에 넣어서 카테고리별 인덱스를 예측함

## 훈련된 모델

https://1drv.ms/u/s!Ag3hKYdTEaOMqMJV8Ut1-uH30xh3EQ

## 재현하는 순서

1. 'raw_data' 폴더 생성후 카카오에서 제공된 train, dev, test chunk 모두 넣기
2. 다음과 같은 이름의 폴더들 만들기: data, info, len_chunks, model, predictions
3. explore.py 실행 (쓰인 단어들의 개수와 median price를 각 chunk마다 계산)
4. info.py 실행 (explore.py에서 생성된 정보들을 기반으로 고려할 단어들과 median price를 산출)
5. make_data.py 실행 (모델의 input으로 들어갈 수 있도록 데이터 전처리)
    * 메모리가 부족하다면 코드 상단의 n_chunk 개수를 늘릴 것
    * train, dev, test 중 필요한 것으로 코드 상단의 case를 수정할 것
6. validation_merge.py 실행 (생성된 validation chunk들을 x_val 하나와 y_val 하나로 합쳐줌)
7. train.py 실행 (모델 생성: model 폴더에 h5 형식으로 모델이 저장됨)
    * data 폴더에 있는 train chunk 개수로 n_chunks 수정할 것
8. predict.py 실행 (예측: predictions 폴더에 tsv 형식으로 저장됨)
    * data 폴더에 있는 chunk 개수로 코드 상단의 n_chunks 수정할 것
    * dev, test 중 필요한 것으로 코드 상단의 case를 수정할 것
