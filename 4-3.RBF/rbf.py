from numpy import mgrid, random
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
import numpy as np

# RBF 클래스 정의
class RBF:
    def __init__(self, indim, numCenters, outdim):  # 초기화 함수
        self.indim = indim  # 입력층 정의
        self.outdim = outdim    # 출력층 정의
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim)    # 중심 벡터 정의
                        for i in range(numCenters)]
        self.beta = 8   # 베타=8 로 정의
        self.W = random.random((self.numCenters, self.outdim))  # 가중치 랜덤결정

    def basisFunc(self, c, d):  # 기본 함수
        assert len(d) == self.indim  # 입력층 길이
        return np.exp(-self.beta * norm(c - d) ** 2)    # 은닉층 구하기

    def activationFunc(self, X):    # 활성화 함수
        G = np.zeros((X.shape[0], self.numCenters), float)  # RBF 활성화 계산
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self.basisFunc(c, x)    # 가우시안
        return G

    def train(self, X, Y):  # 훈련 함수
        rnd_idx = random.permutation(
            X.shape[0])[:self.numCenters]  # 훈련 집합에서 임의의 중심 벡터 선택
        self.centers = [X[i, :] for i in rnd_idx]   # 중심벡터
        G = self.activationFunc(X)  # 가중치 출력 계산하기
        self.W = np.dot(pinv(G), Y)  # 가중치

    def predict(self, X):   # 예측 함수
        G = self.activationFunc(X)  # 활성화 함수 호출
        Y = np.dot(G, self.W)   # Y 계산
        return Y

# n, x, y 정의
n = 100
x = mgrid[-1:1:complex(0, n)].reshape(n, 1) # x그리드 정의
y = np.sin(3*(x+0.5)**3 - 1)    # y설정 후 랜덤 노이즈 추가

# RBF 회귀
rbf = RBF(1, 10, 1) # RBF정의
rbf.train(x, y) # x,y 훈련
z = rbf.predict(x)  # z rbf 예측

# plot 오리지날 데이터
plt.figure(figsize=(6, 4))  # 사이즈 대입
plt.plot(x, y, 'k-', label='ground truth')  # ground truth 추가
plt.plot(x, z, 'r-', linewidth=2, label='prediction')   # prediction 추가
plt.plot(rbf.centers, np.zeros(rbf.numCenters),
         'gs', label='centers of RBFs')  # plot RBFs

# RF 예측
for c in rbf.centers:   # 중심벡터 반복
    cx = np.arange(c-0.7, c+0.7, 0.01)  # x축
    cy = [rbf.basisFunc(np.array([cx_]), np.array([c])) for cx_ in cx]  # y축
    plt.plot(cx, cy, '-', color='gray', linewidth=0.2)  # plot 추가

plt.xlim(-1.2, 1.2)  # x범위
plt.legend()
plt.show()  # 창 출력
