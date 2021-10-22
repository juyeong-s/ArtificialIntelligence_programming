from pgmpy.inference import VariableElimination # 변수제거 패키지
from Bayesian_register import model # 베이지안 망 구조 임포트

# 베이지안 망의 추론
infer = VariableElimination(model)  # 구조 가져오기
A_dist = infer.query(['A'])     # 경보 확률 분포 출력
print('P(A)')  
print(A_dist) 

N_I_EF_BT = infer.query(['N'], evidence = {'E': 'F', 'B': 'T'}) # 지진이 발생하지 않고, 절도가 일어날 때 이웃 전화 확률 분포
print('P(N | E=F, B=T)')    
print(N_I_EF_BT)

N_I_AF_BT = infer.query(['N'], evidence = {'A': 'F', 'B': 'T'}) # 경보가 울리지 않고, 절도가 일어날 때 이웃 전화 확률 분포
print('P(N | A=F, B=T)')
print(N_I_AF_BT)