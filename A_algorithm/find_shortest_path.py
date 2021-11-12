import numpy as np  # 다차원 배열을 처리할 라이브러리
import heapq  # 데이터를 정렬된 상태로 저장하기 위한 라이브러리
import matplotlib.pyplot as plt # 데이터를 시각화하기 위한 라이브러리
from matplotlib.pyplot import figure # matplotlib에서 figure를 만들고 편집할 수 있게 해주는 함수

# 맵을 나타내는 2차원 배열
# 0은 지나갈 수 있는 공간, 1은 지나갈 수 없는 벽을 나타낸다
grid = np.array([

    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],

    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],

    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],

    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# start point and goal

start = (7, 0)  # 시작지점 좌표
goal = (0, 10)  # 도착지점 좌표

# 휴리스틱 함수 : a와 b사이의 유클리드 거리
# 현재 위치에서 목적지까지의 직선거리를 계산
def heuristic(a, b):
    # 두 점 사이의 거리를 알기위한 피타고라스 정리 적용
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) # 각 x, y좌표 사이의 거리의 제곱을 더한 후, 루트씌움

def astar(array, start, goal):  # 인자로 맵, 시작좌표, 목적지 좌표를 받고, 탐색 시작
    # 이동가능한 이웃 리스트 -> 총 8개의 방향 : 위, 위/오른쪽, 오른쪽, 아래/오른쪽, 아래, 아래/왼쪽, 왼쪽, 위/왼쪽
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                 (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()  # 탐색이 종료된 위치들의 집합
    came_from = {} # 각 반복에서 사용할 모든 경로 
    gscore = {start: 0}  # 시작 위치부터 현재 위치까지의 비용을 나타냄
    fscore = {start: heuristic(start, goal)}  # 시작 위치에서 목적지까지의 전체비용
    oheap = []  # min-heap-최단 경로를 찾는 것으로 간주되는 모든 위치를 포함하는 변수
    heapq.heappush(oheap, (fscore[start], start))  # (거리, 시작위치)를 min-heap에 저장한다

    while oheap:  # 옵션이 남아있지 않을 때까지 이동할 수 있는 위치 확인
        current = heapq.heappop(oheap)[1]  # f()값이 최소인 노드 추출
        if current == goal:  # 목적지에 도착할 경우
            data = []
            while current in came_from:  # 목적지에서 역순으로 경로를 추출한다
                data.append(current) # 역순으로 가면서 현재 위치를 data에 저장
                current = came_from[current] # 현재위치를 새로고침해줌

            return data # 역순 경로

        close_set.add(current)  # current 위치를 탐색이 종료된 set에 저장

        for i, j in neighbors:  # current 위치의 각 이웃 위치에 대해 f() 값 계산
            neighbor = current[0] + i, current[1] + j  # 이웃 위치 = 현재 위치에 i,j좌표씩 더해줌
            tentative_g_score = gscore[current] + heuristic(current, neighbor) # g^(n)=g(c)+h((c,n))     

            # 이웃이 grid 외부에 있을경우
            if 0 <= neighbor[0] < array.shape[0]: # array.shape[0]는 grid행의 수 - 위나 아래 bound를 넘을 경우
                if 0 <= neighbor[1] < array.shape[1]: # array.shape[1]는 grid열의 수 - 왼쪽이나 오른쪽 bound를 넘을 경우
                    if array[neighbor[0]][neighbor[1]] == 1:  # 벽을 만났을 경우
                        continue # 무시하고 loop진행
                else:  # y축의 경계를 벗어난 경우
                    # array bound y walls
                    continue # 무시하고 loop진행

            else:  # x축의 경계를 벗어난 경우
                # array bound x walls
                continue # 무시하고 loop진행

            # 이웃이 닫힌 집합에 포함돼있고, g^점수가 기존 g점수보다 클 경우
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue  # 무시하고 loop진행

            # g^(n) < g(n) 이거나, n을 처음 방문한 경우
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current # 이웃에 도달한 최선의 경로에서 직전 위치는 current
                gscore[neighbor] = tentative_g_score  # g(n) = g^(n)
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)  # f(n) = g() + h()
                heapq.heappush(oheap, (fscore[neighbor], neighbor)) # min heap에 (f(), neighbor) 저장

    return False # false 리턴

route = astar(grid, start, goal)  # 탐색 시작
route = route + [start]  # 시작 위치 더하기
route = route[::-1]  # 역순으로 변환
print(route)  # 이동한 좌표 출력

# route의 x와y 좌표를 저장할 변수
x_coords = [] # x좌표
y_coords = [] # y좌표

# route에서 x, y좌표 추출하는 반복문
for i in (range(0, len(route))):
    x = route[i][0] # x좌표 추출
    y = route[i][1] # y좌표 추출
    x_coords.append(x) # x좌표 저장
    y_coords.append(y) # y좌표 저장

# 맵과 경로 그리기
fig, ax = plt.subplots(figsize=(20, 20)) # 여러개의 sub그래프 생성
ax.imshow(grid, cmap=plt.cm.Dark2) # 맵을 이미지로 출력
ax.scatter(start[1], start[0], marker="*", color="yellow", s=200) # 시작위치 그리기
ax.scatter(goal[1], goal[0], marker="*", color="red", s=200) # 목적지 위치 그리기
ax.plot(y_coords, x_coords, color="black") # 경로 그리기
plt.show()  # 맵을 보여줄 창 띄우기
