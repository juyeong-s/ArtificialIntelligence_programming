import networkx as nx
from IPython.core.display import Image
from networkx.drawing.nx_pydot import to_pydot

g1 = nx.Graph()     # 그래프형성
g1.add_edge("A", "B")   # A,B간의 간선 추가
g1.add_edge("D", "A")   # D,A간의 간선 추가
g1.add_edge("B", "C")   # B,C간의 간선 추가
g1.add_edge("C", "D")   # C,D간의 간선 추가

d1 = to_pydot(g1)   # 그래프 그리기
d1.set_dpi(300) # 크기 300
d1.set_margin(0.5)  # 마진 0.5설정
Image(d1.create_png(), width=300)   # 그래프 이미지로 출력
