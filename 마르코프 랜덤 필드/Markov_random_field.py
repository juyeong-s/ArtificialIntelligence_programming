import numpy as np
from Markov_model import model
from pgmpy.inference import VariableElimination

pf_value = model.get_partition_function()   # 분할 함수의 값
print('\n분할 함수의 값: ', pf_value)

infer = VariableElimination(model) # 추론 객체 생성

phi_ABCD = infer.query(['A', 'B', 'C', 'D']) # 전체 분포 phi(A, B, C, D) 출력
print('phi(A, B, C, D)')
print(phi_ABCD)
P_ABCD = phi_ABCD.values/pf_value # 확률 = (팩터의 곱) / (분할 함수의 값)
PABCD = np.reshape(P_ABCD, -1)
for val in PABCD: # 확률의 출력
    print(val, '\n')

A_dist = infer.query(['A']) # A의 분포 phi(A) 출력
print('phi(A)')
print(A_dist)
P_A = A_dist.values/np.sum(A_dist.values)   # A가 0또는 1 일 때 각각의 확률 확률 구하기 A(0 or 1) / A 전체 합
for val in P_A:     # 0과 1일 경우의 확률 출력
    print(val, '\n')

AIBOC1_dist = infer.query(['A'], evidence = {'B':0, 'C':1}) # B=0, C=1일 때 A의 분포 phi(A|B=0, C=1) 출력
print('phi(A|B=0, C=1)')
print(AIBOC1_dist)
P_AIBOC1 = AIBOC1_dist.values/np.sum(AIBOC1_dist.values)    # A가 0또는 1 일 때 각각의 확률 확률 구하기 A(0 or 1) / A 전체 합
for val in P_AIBOC1:     # 0과 1일 경우의 확률 출력
    print(val, '\n')