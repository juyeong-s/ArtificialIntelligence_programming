import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from IPython.core.display import Image

g = nx.Graph()  # 그래프형성
g.add_edge("E", "A")   # E,A간의 간선 추가
g.add_edge("B", "A")   # B,A간의 간선 추가
g.add_edge("A", "N")   # A,N간의 간선 추가
d = to_pydot(g) # 그래프 그리기
d.get_node("N")[0].set_fillcolor("gray")    # 회색으로 채움
d.get_node("N")[0].set_style("filled")  # 스타일 색상 꽉 채움
d.set_dpi(300)  # 크기 300
d.set_margin(0.2) # 마진 0.2 설정
Image(d.create_png(), width=200)   # 그래프 이미지로 출력
