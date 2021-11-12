from collections import namedtuple # 네임드 튜플 임포트
from random import choice # 랜덤으로 하나 선택해줌
from monte_carlo_tree_search import MCTS, Node # 몬테카를로 임포트

TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal") # 네임드 튜플 선언

class TicTacToeBoard(TTTB, Node): # TTTB의 속성들도 상속
    def find_children(board): # 전체 가능한 move집합으로 반환
        if board.terminal:  # 게임이 끝날 경우
            return set() # 아무것도 하지 않음
        return {  # 그렇지 않으면, 비어있는 곳에 각각 시도
            board.make_move(i) for i, value in enumerate(board.tup) if value is None
        }

    def find_random_child(board):  # 무작위로 move 선택
        if board.terminal: # 게임이 끝날 경우
            return None  # 아무것도 하지 않음
        empty_spots = [i for i, value in enumerate(board.tup) if value is None] # 빈 곳 찾기
        return board.make_move(choice(empty_spots)) # 아무 빈곳이나 하나 뽑아줌

    def reward(board):  # 점수 계산
        if not board.terminal: # 게임이 끝나지 않았으면
            raise RuntimeError(f"reward called on nonterminal board {board}") # 런타임에러
        if board.winner is board.turn: # 본인 차례이면서 본인이 이긴 상황
            raise RuntimeError(f"reward called on unreachable board {board}") # 런타임에러
        if board.turn is (not board.winner):  # 상대가 이긴 상황임
            return 0 # 0점 획득
        if board.winner is None:  # 비긴 상황임
            return 0.5 # 반점만 획득
        raise RuntimeError(f"board has unknown winner type {board.winner}") # 위어를 알 수 없음

    def is_terminal(board):  # 게임 종료 여부 판단
        return board.terminal

    def make_move(board, index):  # index 위치에 board.turn 표시하기
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1:] # 움직일 곳을 정하기 위한 현재 보드
        turn = not board.turn  # 순서 바꾸기
        winner = _find_winner(tup)  # 승자 또는 종료 여부 판단
        is_terminal = (winner is not None) or not any(v is None for v in tup) # winner값으로 종료 여부 판단
        return TicTacToeBoard(tup, turn, winner, is_terminal)  # 보드 상태 반환

    def to_pretty_string(board):  # 보드 상태 출력
        def to_char(v): return ( # 문자로 변환해서 출력
            "X" if v is True else ("O" if v is False else " ")) # v가 True일 경우 X, False일 경우 O 출력
        rows = [ # row값
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return ( # 새로 반영된 보드판 출력
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row)
                        for i, row in enumerate(rows)) # row출력
            + "\n"
        )


def play_game():  # 게임 시작
    tree = MCTS() # monte carlo tree search를 tree로 받아옴
    board = new_Board() # 비어있는 보드판 생성
    print(board.to_pretty_string()) # 보드판 출력
    while True: # 무한루프
        row_col = input("enter row,col: ") # row, col입력받기
        row, col = map(int, row_col.split(",")) # int형으로 데이터 변환
        index = 3 * (row - 1) + (col - 1) # 입력한 위치의 index
        if board.tup[index] is not None:  # 비어있는 위치가 아닌 경우
            raise RuntimeError("Invalid move") # 런타임 에러
        board = board.make_move(index)  # index 위치의 보드 상태를 변경함
        print(board.to_pretty_string()) # 변경된 상태 출력
        if board.terminal:  # 게임 종료
            break # 무한루프 빠져나옴
        # 진행하면서 훈련하거나 처음에만 훈련할 수 있습니다.
        # 여기에서 우리는 매 턴마다 50번의 롤아웃을 하면서 훈련합니다.
        for _ in range(50):  # 매번 50번의 rollout을 수행
            tree.do_rollout(board) # MCTS의 rollout함술 호출
        board = tree.choose(board)  # 최선의 값을 갖는 move 선택해서 보드에 반영
        print(board.to_pretty_string()) # 새로 반영된 보드 출력
        if board.terminal: # 게임 종료
            break # 무한루프 빠져나옴


def _winning_combos():  # 이기는 배치 조합
    for start in range(0, 9, 3):  # 행에 3개 연속 - 0부터 9까지 3칸씩 증가
        yield (start, start + 1, start + 2) # 오른쪽으로 가는 가로선
    for start in range(3):  # 열에 3개 연속 - 0부터 3까지
        yield (start, start + 3, start + 6) # 아래로 가는 세로선
    yield (0, 4, 8)  # 오른쪽 아래로 가는 대각선 3개
    yield (2, 4, 6)  # 왼쪽 아래로 가는 대각선 3개


def _find_winner(tup): # X가 이기면 True, 0이 이기면 False, 미종료 상태이면 None 반환
    for i1, i2, i3 in _winning_combos(): # 이기는 배치 조합에서 받아온 i1, i2, i3
        v1, v2, v3 = tup[i1], tup[i2], tup[i3] # 이기는 배치조합을 훑어서
        if False is v1 is v2 is v3: # v1, v2, v3가 False일 경우
            return False # False 리턴
        if True is v1 is v2 is v3: # v1, v2, v3가 True일 경우
            return True # True 리턴
    return None # 모두 아닐 경우 None리턴


def new_Board():  # 비어있는 보드판 생성
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False) # 3X3보드판


if __name__ == "__main__":
    play_game() # 게임 시작
