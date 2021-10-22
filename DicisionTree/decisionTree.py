import matplotlib.pyplot as plt # 그래프를 보여줄 패키지
from sklearn.datasets import load_iris  # 데이터셋을 불러올 패키지
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 트리 생성을 위한 패키지

iris = load_iris()  # 데이터 로드
decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=3)    # 결정트리 모델 생성 : 트리 깊이 3까지 분류
decision_tree = decision_tree.fit(iris.data, iris.target)   # DecisionTreeClassifier 학습하기

plt.figure()    # 객체 생성
plot_tree(decision_tree, filled=True)   # 결정 트리 
plt.show()  # 창 열기