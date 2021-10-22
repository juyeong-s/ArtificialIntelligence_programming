import numpy as np  # 수학적 패키지
import skfuzzy as fuzz  # 퍼지 추론 방법 지원 패키지
import matplotlib.pyplot as plt # 그래프를 보여줄 패키지

# 전역 변수 정의
x_qual = np.arange(0, 11, 1)    # 음식 품질의 범위 : [0, 11) 1간격
x_serv = np.arange(0, 11, 0.2)  # 서비스 만족도 범위 : [0, 11) 0.2간격
x_tip = np.arange(0, 31, 1)  # 팁의 범위 : [0, 31) 1격

# 소속함수 정의

# 음식 품질 . 사다리꼴 함수
qual_poor = fuzz.trapmf(x_qual, [0, 0, 1, 3])    # 품질이 안좋다 : 0 0 1 3 벡터
qual_amazing = fuzz.trapmf(x_qual, [7, 9, 10, 10])   # 맛있다 : 7 9 10 10 벡터

 # 서비스 만족도. 가우시안 소속함수
serv_poor = fuzz.gaussmf(x_serv, 0, 1)   # 서비스가 안좋다 : 0 평균 1 시그마
serv_acceptable = fuzz.gaussmf(x_serv, 5, 1)    # 서비스 좋음 : 5 평균 1 시그마
serv_amazing = fuzz.gaussmf(x_serv, 10, 1)  # 서비스가 훌륭 : 10 평균 1 시그마

# 팁의 규모. 삼각 소속함수
tip_low = fuzz.trimf(x_tip, [0, 5, 10])  # 팁 적게 : 0 5 10 벡터
tip_medium = fuzz.trimf(x_tip, [10, 15, 20])    # 팁 중간 : 10 15 20 벡터
tip_high = fuzz.trimf(x_tip, [20, 25, 30])  # 팁 많이 : 20 25 30 벡터

# 소속함수 그리기
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))    # 객체 생성

# 음식 품질 그래프
ax0.plot(x_qual, qual_poor, 'b', linewidth=1.5, label='Poor')   # 파란색으로 낮은 음식 품질을 나타냄
ax0.plot(x_qual, qual_amazing, 'r', linewidth=1.5, label='Amazing') # 빨간색으로 좋은 음식 품질을 나타냄
ax0.set_title('Food quality')   # '음식 품질' 제목
ax0.legend()    # 그래프 그리기

# 서비스 품질 그래프
ax1.plot(x_serv, serv_poor, 'b', linewidth=1.5, label='Poor')   # 파란색으로 낮은 서비스 품질을 나타냄
ax1.plot(x_serv, serv_acceptable, 'g', linewidth=1.5, label='Acceptable')   # 초록색으로 높은 서비스 품질을 나타냄
ax1.plot(x_serv, serv_amazing, 'r', linewidth=1.5, label='Amazing')   # 빨간색으로 훌륭한 서비스 품질을 나타냄
ax1.set_title('Service quality')   # '서비스 품질' 제목
ax1.legend()    # 그래프 그리기

# 팁의 amount 그래프
ax2.plot(x_tip, tip_low, 'b', linewidth=1.5, label='Low')   # 파란색으로 적은 팁을 나타냄
ax2.plot(x_tip, tip_medium, 'g', linewidth=1.5, label='Medium')   # 초록색으로 보통의 팁을 나타냄
ax2.plot(x_tip, tip_high, 'r', linewidth=1.5, label='High')   # 빨간색으로 많은 팁을 나타냄
ax2.set_title('Tip amount')   # '팁의 amount' 제목
ax2.legend()    # 그래프 그리기

# 상단/오른쪽 축 없애기
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False) # 상단 축 없애기
    ax.spines['right'].set_visible(False)   # 우측 축 없애기
    ax.get_xaxis().tick_bottom()    # 눈금을 좌표축 맨 아래로 이동 - x좌표
    ax.get_yaxis().tick_left()  # 눈금을 좌표축 맨 왼쪽으로 이동 - y좌표

# 함수 보여주기
plt.tight_layout()

# 범위 domain에서 정의된 소속함수 mf의 val에 대한 값
def membership(domain, mf, val):
    return fuzz.interp_membership(domain, mf, val)  # domain에 대한 멤버십 정도를 찾아줌

# 퍼지 규칙을 적용한 food quality가 qual_val, service 점수가 serv_val일 때 tip 계산
def compute_tip_amount(qual_val, serv_val):
    qual_level_poor = fuzz.interp_membership(x_qual, qual_poor, qual_val)   # 낮은 음식 품질 점수
    qual_level_amazing = fuzz.interp_membership(x_qual, qual_amazing, qual_val)   # 좋은 음식 품질 점수

    serv_level_poor = fuzz.interp_membership(x_serv, serv_poor, serv_val)   # 낮은 음식 서비스 점수
    serv_level_acceptable = fuzz.interp_membership(x_serv, serv_acceptable, serv_val)   # 좋은 서비스 품질 점수
    serv_level_amazing = fuzz.interp_membership(x_serv, serv_amazing, serv_val)   # 훌륭한 서비스 품질 점수

    # Rule 1: IF service = poor OR food = poor THEN tip = low
    # 만약 서비스가 안좋거나 음식이 별로일 경우 tip을 적게 준다.
    satisfaction_rule1 = np.fmax(qual_level_poor, serv_level_poor)
    # 해당 출력에서 ​​상단을 잘라내어 적용
    tip_activation_low = np.fmin(satisfaction_rule1, tip_low)

    # Rule 2: IF service = acceptable THEN tip = medium
    # 만약 서비스가 좋을 경우 tip을 보통으로 준다.
    tip_activation_medium = np.fmin(serv_level_acceptable, tip_medium)

    # Rule 3: IF service = amazing OR food = amazing THEN tip = high
    # 만약 서비스가 좋거나 음식이 맛있을 경우 tip을 많이 준다.
    satisfaction_rule3 = np.fmax(qual_level_amazing, serv_level_amazing)
    tip_activation_high = np.fmin(satisfaction_rule3, tip_high)
    tip0 = np.zeros_like(x_tip)

    # 각 규칙의 추론결과 결합
    aggregated = np.fmax(tip_activation_low, 
                    np.fmax(tip_activation_medium, tip_activation_high))    # 세가지 멤버십 함수 모두 집계

    # 비퍼지화 결과 계산
    tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
    return tip  # 결과 팁 리턴

# 음식 품질 점수가 6.6이고, 서비스 점수가 9일 때 팁을 출력
print('food quality score = 6.6, service score = 9일 때 팁 : ',
      compute_tip_amount(6.6, 9))

### 3차원 그래프

d_qual = np.arange(0, 10, 0.5)  # 음식 품질의 범위
d_serv = np.arange(0, 10, 0.5)  # 서비스 만족도 범위

Q, S = np.meshgrid(d_qual, d_serv)  # 1차원 그래프에 격자그리기
T = np.zeros_like(Q)    # Q크기만큼 0으로 가득 찬 배열

for i in range(20):
    for j in range(20):
        T[i, j] = compute_tip_amount(Q[i, j], S[i, j])  # 1차원 그래프로부터 팁 구하기

fig = plt.figure(figsize=(14, 10))  # 14:10인치로 창의 크기
ax = plt.axes(projection='3d')  # 3차원
ax.plot_surface(Q, S, T, rstride=1, cstride=1, cmap='viridis',
                linewidth=0.4, antialiased=True)    # 3차원 표면 그리기

ax.set_xlabel('food quality')   # x축은 음식 품질
ax.set_xlim(0, 10)  # 0부터 10 범위
ax.set_ylabel('service')   # y축은 서비스 품질
ax.set_ylim(0, 10)  # 0부터 10 범위
ax.set_zlabel('tip')   # z축은 팁의 amount
ax.set_zlim(0, 30)  # 0부터 30 범위
ax.set_title('fuzzy inference-based tip computation')   # 그래프의 제목

plt.show()  # 그래프 그리기
