# -*- coding: utf-8 -*-
import numpy as np

def Conv2D(X, W, w0, p=(0, 0), s=(1, 1)):
    n1 = X.shape[0] + 2*p[0]    # 패딩 반영
    n2 = X.shape[1] + 2*p[1]
    X_p = np.zeros(shape=(n1, n2))  # 0으로 채움
    X_p[p[0]:p[0]+X.shape[0], p[1]:p[1]+X.shape[1]] = X    # 입력 X 복사
    res = []
    # 2차원 행렬-2중 for문
    for i in range(0, int((X_p.shape[0] - W.shape[0])/s[0])+1, s[0]):
        res.append([])
        for j in range(0, int((X_p.shape[1] - W.shape[1])/s[1])+1, s[0]):
            X_s = X_p[i:i+W.shape[0], j:j+W.shape[1]]   # 컨볼루션 영역 계산
            res[-1].append(np.sum(X_s*W) + w0)   # 컨볼루션 영역 append하기
    return (np.array(res))  # 결과 리턴


# 입력과 필터 w0 정의
X = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [
             0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
W = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
w0 = 1

conv = Conv2D(X, W, w0, p=(0, 0), s=(1, 1))  # 패딩이 0이고, 스트라이드가 1
print('X = ', X)    # 2차원 행렬
print('\nW = ', W)  # 컨볼루션 필터
print('\n컨볼루션 결과 p=(0,0), s=(1,1) \n', conv)
conv = Conv2D(X, W, w0, p=(1, 1), s=(1, 1))   # 패딩이 1이고, 스트라이드가 1
print('\n컨볼루션 결과 p=(1,1), s=(1,1) \n', conv)
conv = Conv2D(X, W, w0, p=(1, 1), s=(2, 2))   # 패딩이 1이고, 스트라이드가 2
print('\n컨볼루션 결과 p=(1,1), s=(1,1) \n', conv)
