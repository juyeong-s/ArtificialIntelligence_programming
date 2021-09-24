from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS:
    "Monte Carlo tree searcher. 먼저 rollout한 다음, 위치(move)선택"

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # 노드별 이긴 횟수(reward) 값을 0으로 초기화
        self.N = defaultdict(int)  # 노드별 방문횟수(visit count)를 0으로 초기화
        self.children = dict()  # 노드의 자식노드
        self.exploration_weight = exploration_weight  # UCT 계산에 사용되는 계수

    def choose(self, node):
        "node의 최선인 자식 노드 선택"
        if node.is_terminal(): # node가 단말인 경우 오류
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children: # node가 children에 포함되지 않으면 무작위 선택
            return node.find_random_child()

        def score(n):  # 점수 계산
            if self.N[n] == 0:
                return float("-inf")  # 한번도 방문하지 않은 노드인 경우 - 선택 배제
            return self.Q[n] / self.N[n]  # 평균 점수

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "게임 트리에서 한 층만 더 보기"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node): # 선택 단계
        "node의 아직 시도해보지 않은 자식 노드 찾기"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node의 child나 grandchild가 아닌 경우: 아직 시도해보지 않은 것 또는 단말 노드
                return path
            unexplored = self.children[node] - self.children.keys() # 차집합
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # 한 단게 내려가기

    def _expand(self, node): # 확장 단계
        "children에 node의 자식노드 추가"
        if node in self.children:
            return  # 이미 확장된 노드
        self.children[node] = node.find_children() # 선택가능 move들을 node의 children에 추가

    def _simulate(self, node): # 시뮬레이션 단계
        "node의 무작위 시뮬레이션에 대한 결과(reward) 반환"
        invert_reward = True
        while True:
            if node.is_terminal(): # 단말에 도달하면 승패여부 결정
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child() # 선택할 수 있는 것 중에서 무작위로 선택
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward): # 역전파 단계
        "단말 노드의 조상 노드들에게 보상(reward) 전달"
        for node in reversed(path): # 역순으로 감ㄴ서 Monte Carlo 시뮬레이션 결과 반영
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 자신에게는 1 상대에게는 0, 또는 그 반대

    def _uct_select(self, node): # UCB 정책 적용을 통한 노드 확장 대상 노드 선택
        "탐험(exploration)과 이용(exploitation)의 균형을 맞춰 node의 자식 노드 선택"

        # node의 모든 자신 노드가 이미 확장되었는지 확인
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "UCB(Upper confidence bound) 점수 계산"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    " 게임 트리의 노드로서 보드판의 상태 표현"
    @abstractmethod
    def find_children(self):
        "해당 보드판 상태의 가능한 모든 후속 상태"
        return set()

    @abstractmethod
    def find_random_child(self):
        "현 보드에 대한 자식 노드 무작위 선택"
        return None

    @abstractmethod
    def is_terminal(self):
        "자식 노드인지 판단"
        return True

    @abstractmethod
    def reward(self):
        "점수 계산"
        return 0

    @abstractmethod
    def __hash__(self):
        "노드에 해시적용 가능하도록(hashable) 함"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "노드는 서로 비교 가능해야 함"
        return True
