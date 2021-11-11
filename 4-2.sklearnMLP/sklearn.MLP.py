import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 와인 데이터 로드
wine = load_wine()
# 와인 데이터의 feature, target으로 데이터 프레임을 만든다.
data = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
print(data.head())

X = wine.data   # 와인에서 feature로 되어 있는 데이터 가져옴
y = wine.target  # 와인 데이터에서 target 데이터를 가져옴
X_train, X_test, y_train, y_test = train_test_split(X, y)   # 테스트 데이터 와인 분류

scaler = StandardScaler()   # 스케일링 표준화
scaler.fit(X_train)  # 학습용 데이터를 이용해 학습하기
StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)  # 훈련 데이터 스케일 조정
X_test = scaler.transform(X_test)   # 테스트 데이터 스케일 조정

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)  # MLP객체 생성
mlp.fit(X_train, y_train)   # mlp학습 시행
predictions = mlp.predict(X_test)   # X_test로 예측 시행
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
