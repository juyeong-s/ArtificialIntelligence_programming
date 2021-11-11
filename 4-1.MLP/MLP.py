import numpy as np  # 행렬 사용
import matplotlib.pyplot as plt


class MLP:
    # 신경망 초기화하기
    def __init__(self, hidden_node=3):
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.input_node = 1
        self.hidden_node = hidden_node
        self.output_node = 1
        # 가중치 생성
        self.w1 = np.random.rand(self.hidden_node, self.input_node)
        self.b1 = np.random.rand(self.hidden_node, 1)
        self.w2 = np.random.rand(self.output_node, self.hidden_node)
        self.b2 = np.random.rand(self.output_node, 1)

    # 시그모이드 함수
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # x에 대해 미분한 시그모이드 함수
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # 신경망 학습시키기
    def train(self, train_x, train_y, alpha=0.1, max_iter=500):
        np.random.seed(0)
        # 입력, 은닉, 출력 노드의 수
        input_node = self.input_node
        hidden_node = self.hidden_node
        output_node = self.output_node
        alpha = alpha
        max_iter = max_iter
        # 최대 반복 횟수까지 반복
        for iter in range(1, max_iter):
            # n번 훈련
            for i in range(n_train):
                z1 = np.dot(self.w1, train_x[i].reshape(
                    1, 1)) + self.b1  # 은닉층 계산
                a1 = self.sigmoid(z1)    # 시그모이드 함수의 파라미터로 넘기기
                z2 = np.dot(self.w2, a1) + self.b2  # 출력 값
                y_hat = z2  # 출력 값 y
                y_hat_list[i] = y_hat  # 출력 값 y목록
                e = 0.5 * (train_y[i] - y_hat) ** 2  # 오차함수
                dy = - (train_y[i] - y_hat)  # y 미분
                dz2 = 1  # z2 미분
                dw2 = a1.T  # w2 미분
                delta_w2 = dy * dz2 * dw2   # 델타 w2
                delta_b2 = dy * dz2  # 델타 b2
                da1 = self.w2.T  # 시그모이드 함수 값 미분
                dz1 = self.d_sigmoid(z1)    # 미분한 시그모이드 함수의 파라미터로 넘기기
                dw1 = train_x[i].T  # w1 미분
                delta_w1 = dy * dz2 * da1 * dz1 * dw1  # 델타 w1
                delta_b1 = dy * dz2 * da1 * dz1  # 델타 b1
                # 가중치 계산
                self.w2 -= alpha * delta_w2
                self.b2 -= alpha * delta_b2
                self.w1 -= alpha * delta_w1
                self.b1 -= alpha * delta_b1

    # 신경망 예측하기
    def predict(self, test_x):
        # test횟수 만큼 반복
        for i in range(n_test):
            z1 = np.dot(self.w1, test_x[i].reshape(1, 1)) + self.b1  # 은닉 층 계산
            a1 = self.sigmoid(z1)   # 시그모이드 함수 계산-은닉 층 계산
            z2 = np.dot(self.w2, a1) + self.b2  # 출력 값 계산
            y_hat = z2  # y^으로 넣어주기
            y_hat_list[i] = y_hat   # y리스트
        return y_hat_list   # y의 리스트 리턴


n_train = 20    # 20번 훈련
train_x = np.linspace(0, np.pi * 2, n_train)  # x축
train_y = np.sin(train_x)  # y축

n_test = 60  # 60번 테스트
test_x = np.linspace(0, np.pi * 2, n_test)  # x축
test_y = np.sin(test_x)  # y축
y_hat_list = np.zeros(n_test)   # 60개 칸 공간

mlp = MLP(hidden_node=4)    # 은닉 노드 수 4개
mlp.train(train_x, train_y, max_iter=600)   # 최대 반복 횟수 600번-훈련
plt.plot(test_x, test_y, label='ground truth')  # ground truth 그래프 출력

y_hat_list = mlp.predict(test_x)    # 신경망 예측하기
plt.plot(test_x, y_hat_list, '-r', label='prediction')  # prediction 그래프 출력
plt.legend()
plt.show()  # 그래프 보여주기
