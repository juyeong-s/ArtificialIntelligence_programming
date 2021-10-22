from pyswip import *

p = Prolog()
# 사실 정의
p.assertz("vegeterian(cinderella)") # cinderella는 채식주의자
p.assertz("vegeterian(snow_white)") # snow_white는 채식주의자
p.assertz("vegetable(사과)")        # 사과는 채소
p.assertz("vegetable(가지)")        # 가지는 채소
p.assertz("vegetable(당근)")        # 당근은 채소
p.assertz("likes(X,Y) :- vegeterian(X), vegetable(Y)")  # X가 채식주의자이고 Y가 채소이면, X는 Y를 좋아함
p.assertz("likes(X,오이) :- vegeterian(X)") # X가 채식주의자이면, X는 오이를 좋아한다.
p.assertz("likes(cinderella, 달걀)")        # cinderella은 달걍을 좋아한다.

# 규칙 정의
for ans in p.query("likes(X,가지)"):    # X는 가지를 좋아한다.
    print(ans["X"], 'likes 가지')       # cinderella likes 가지

for ans in p.query("likes(snow-white,Y)"):    # snow-white는 Y를 좋아한다.
    print('snow-white likes', ans["Y"])       # snow_white likes 가지

for ans in p.query("likes(cinderella,Y)"):    # cinderella
    print('cinderella likes', ans["Y"])       # cinderella likes 가지

vegetable = Functor("vegetable", 1)     # 1개의 파라미터
likes = Functor("likes", 2)             # 2개의 파라미터
Y = Variable()      # Y는 채소

q = Query(likes("cinderella", Y), vegetable(Y)) # cinderella는 Y를 좋아함 / Y는 채소
while q.nextSolution(): # 쿼리 반복
    print("cinderella 좋아하는 채소:", Y.value) # cinderella 좋아하는 채소 : 사과, 가지, 당근
q.closeQuery()  # 쿼리 종료