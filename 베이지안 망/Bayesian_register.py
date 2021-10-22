from pgmpy.models import BayesianNetwork
from Bayesian_prob_distribute import P_E, P_B, P_A_I_EB, P_N_I_A    # 확률분포 임포트

# 베이지안 망 구조 정의
model = BayesianNetwork([('E', 'A'), ('B', 'A'), ('A', 'N')])
model.add_cpds(P_E, P_B, P_A_I_EB, P_N_I_A) # 베이지안 구조로 부터 확률분포 등록
print(model.check_model())  # 정의된 구조 출력