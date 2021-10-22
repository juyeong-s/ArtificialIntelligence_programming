from pgmpy.factors.discrete import TabularCPD
import numpy as np

# 지진(Earthquake) 발생 확률 분포
P_E = TabularCPD('E', 2, [[0.9], [0.1]], state_names={'E': ['F', 'T']}) # 지진 발생확률 0.1, 발생하지 않을 확률 0.9
print('P(E)')   # 지진 발생 확률 분포표 출력
print(P_E)

# 절도(Burglary) 발생 확률분포
P_B = TabularCPD('B', 2, [[0.7], [0.3]], state_names={'B': ['F', 'T']}) # 절도 발생확률 0.3, 발생하지 않을 확률 0.7
print('P(B)')   # 절도 발생 확률 분포표 출력
print(P_B)

# 경보(Alarm) 발생 확률 분포
P_A_I_EB = TabularCPD('A', 2, [[0.99, 0.1, 0.3, 0.01], [0.01, 0.9, 0.7, 0.99]], # 지진 발생 X, 절도 발생 X일 경우, 경보 발생확률 0.01
            evidence = ['E', 'B'], evidence_card = [2, 2],                      # 지진 발생 X, 절도 발생 O일 경우, 경보 발생확률 0.9
            state_names = {'A': ['F', 'T'], 'E': ['F', 'T'], 'B': ['F', 'T']})  # 지진 발생 O, 절도 발생 X일 경우, 경보 발생확률 0.7
                                                                                # 지진 발생 O, 절도 발생 O일 경우, 경보 발생확률 0.99
print('P(A|EB)')   # 경보 발생 확률 분포표 출력
print(P_A_I_EB)

# 이웃(Neighbor) 전화 확률 분포
P_N_I_A = TabularCPD('N', 2,
            np.array([[0.9, 0.2], [0.1, 0.8]]),         # 경보 발생 X 일 경우, 이웃이 전화할 확률 0.1
            evidence = ['A'], evidence_card = [2],       # 경보 발생 O 일 경우, 이웃이 전화할 확률 0.8
            state_names = {'N': ['F', 'T'], 'A': ['F', 'T']})
print('P(N|A)')   # 이웃 전화 확률 분포표 출력
print(P_N_I_A)