from pyswip import *

N = 3   # 원반 개수

def notify(t):  # 
    print("%s --> %s" % tuple(t))   # 원반 이동 방향 출력
notify.arity = 1        # notify함수의 애러티를 1로 설정

prolog = Prolog()
registerForeign(notify) # 외부 함수 prolog에 등록
prolog.consult("Prolog/hanoi.pl")  # prolog 프로그램 파일 읽어들이기
list(prolog.query("hanoi(%d)" % N))  # hanoi함수에 N=3파라미터로 전송

""" hanoi.pl
hanoi(N) :- move(N, left, right, center).   # move함수 호출
move(0, _, _, _) :- !.  # N이 0이 될 경우 종료
move(N,A,B,C) :-
    M is N-1,           # M=N-1
    move(M,A,C,B),      # 재귀로 A,C,B순서로 M대입
    notify([A,B]),      # notify함수에 A,B 출력하도록
    move(M,C,B,A).      # 재귀로 C,B,A순서로 M대입
"""
