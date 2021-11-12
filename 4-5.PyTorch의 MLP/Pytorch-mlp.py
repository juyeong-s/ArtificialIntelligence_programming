# -*- coding: utf-8 -*-
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)    # MNIST정의

# MNIST 데이터 정의
X = mnist.data/255
y = mnist.target

plt.imshow(X[0].reshape(28, 28), cmap='gray')   # plt에 반영
plt.show()  # plt창 띄우기
print("이미지 레이블 : {}".format(y[0]))    # 이미지 레이블 출력


# 데이터 집합을 훈련 데이터와 테스트 데이터로 분류
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=0)
X_train = torch.Tensor(X_train)  # 훈련 데이터 텐서 변환
X_test = torch.Tensor(X_test)  # 테스트 데이터 텐서 변환
y_train = torch.LongTensor(list(map(int, y_train)))  # 훈련 데이터 텐서 변환
y_test = torch.LongTensor(list(map(int, y_test)))  # 테스트 데이터 텐서 변환

# 텐서 데이터 집합
ds_train = TensorDataset(X_train, y_train)  # 훈련 데이터 집합
ds_test = TensorDataset(X_test, y_test) # 테스트 데이터 집합

# 데이터 로드
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)    # 훈련 데이터 로드
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False) # 테스트 데이터 로드

# 모델 구성
model = nn.Sequential() # 연속적으로 연결하는 nn모델
model.add_module('fc1', nn.Linear(28*28*1, 100))    # 28*28*1 선형 모델 추가
model.add_module('relu1', nn.ReLU())    # Rectified Linear Unit
model.add_module('fc2', nn.Linear(100, 100))    # 선형
model.add_module('relu2', nn.ReLU())    # Rectified Linear Unit
model.add_module('fc3', nn.Linear(100, 10)) # 선형

loss_fn = nn.CrossEntropyLoss()  # 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam 최적화

# 학습모드 함수
def train(epoch):
    model.train()  # 학습 모드로 변환
    for data, targets in loader_train:
        optimizer.zero_grad()  # 그레디언트 초기화
        outputs = model(data)   # 출력 데이터
        loss = loss_fn(outputs, targets)    # 손실 계산
        loss.backward() # 손실 Backward
        optimizer.step()    # 단계 최적화
    print('에포크 {}: 완료'.format(epoch))  # 에포크

# 테스트모드 함수
def test(head):
    model.eval()  # 테스트 모드로 변환
    correct = 0
    with torch.no_grad():   #그레디언트 계산 불필요
        for data, targets in loader_test:
            outputs = model(data)   # 출력 데이터
            _, predicted = torch.max(outputs.data, 1)   # torch 최대값
            correct += predicted.eq(targets.data.view_as(predicted)).sum()
    data_num = len(loader_test.dataset) # 정확도 계산
    print('{} 정확도: {}/{}({:.Of}%)'.format(head,  ## 시작, 학습중, 학습후 정확도 출력
          correct, data_num, 100.*correct/data_num))


test('시작')    # 테스트 시작
for epoch in range(3):  # 3번 반복
    train(epoch)    # 에포크 학습중
    test('학습중')
test('학습 후') # 학습 후

index = 10  # 테스트 데이터 중에서 확인해볼 데이터의 인덱스
model.eval()  # 모델을 테스트 모드로 전환
data = X_test[index]    # X 학습데이터
output = model(data)  # 모델 적용
print('{} 번째 학습데이터의 테스트 결과 : {}'.format(index, output)) # index번째 학습데이터의 테스트 결과 출력
_, predicted = torch.max(output.data, 0)    # 예측 torch 최대값
print('{} 번째 데이터의 예측 : {}'.format(index, predicted))    # index번째 데이터의 예측 결과 출력
X_test_show = (X_test[index]).numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')    # 창 열어서 보여줌
print('실제 레이블: {}'.format(y_test[index]))  # 실제 레이블 출력
