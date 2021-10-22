from pyswip import *

p = Prolog()

# 사실(fact)의 정의
p.assertz("female(심청)")   # 사실 추가 :여자 심청
p.assertz("female(홍련)")   # 사실 추가 :여자 홍련
p.assertz("female(춘향)")   # 사실 추가 :여자 춘향
p.assertz("female(월매)")   # 사실 추가 :여자 월매
p.assertz("female(몽룡모)")   # 사실 추가 :여자 몽룡모
p.assertz("male(배무룡)")   # 사실 추가 :남자 배무룡
p.assertz("male(심학규)")   # 사실 추가 :남자 심학규
p.assertz("male(성참판)")   # 사실 추가 :남자 성참판
p.assertz("male(이몽룡)")   # 사실 추가 :남자 이몽룡
p.assertz("male(이한림)")   # 사실 추가 :남자 이한림
p.assertz("male(변학도)")   # 사실 추가 :남자 변학도
p.assertz("female(숙향)")   # 사실 추가 :여자 숙향
p.assertz("father(심학규,심청)")   # 사실 추가 :심청의 아빠 심학규
p.assertz("father(성참판,춘향)")   # 사실 추가 :춘향의 아빠 성참판
p.assertz("father(배무룡,홍련)")   # 사실 추가 :홍련의 아빠 배무룡
p.assertz("father(이한림,이몽룡)")   # 사실 추가 : 이몽룡의 아빠 이한림
p.assertz("father(이몽룡,숙향)")   # 사실 추가 : 숙향의 아빠 이몽룡
p.assertz("mother(월매,춘향)")   # 사실 추가 : 춘향의 엄마 이몽룡
p.assertz("spouse(이한림,몽룡모)")   # 사실 추가 :몽룡모과 이한림은 서로의 배우자
p.assertz("healthy(이몽룡)")   # 사실 추가 :건강한 이몽룡
p.assertz("wealthy(이몽룡)")   # 사실 추가 :풍족한 이몽룡
p.assertz("healthy(변학도)")   # 사실 추가 :건강한 변학도
p.assertz("healthy(심청)")   # 사실 추가 :건강한 심청
p.assertz("wealthy(변학도)")   # 사실 추가 :풍족한 변학도
p.assertz("healthy(홍련)")   # 사실 추가 :건강한 홍련

# 규칙(rule)의 정의
p.assertz("grandfather(X,Y) :- father(X,Z),father(Z,Y)")    # Z의 아빠가 X이고,  Y의 아빠가 Z이면 Y의 할아버지는 X
p.assertz("husband(X,Y) :- father(X,Z),mother(Y,Z)")    # Z의 아빠가 X이고,  Z의 엄마가 Y이면 Y의 남편은 X
p.assertz("wife(X,Y) :- mother(X,Z),father(Y,Z)")    # Z의 엄마가 X이고,  Z의 아빠가 Y이면 Y의 아내는 X
p.assertz("wife(X,Y) :- spouse(Y,X), female(X)")    # X와 Y가 서로 배우자이고,  X가 여자이면 X는 Y의 아내
p.assertz("wife(X,Y) :- spouse(X,Y), female(X)")    # Y와 X가 서로 배우자이고,  X가 여자이면 X는 Y의 아내
p.assertz("traveler(X) :- healthy(X), wealthy(X)")    # X가 건강하고, 풍족하면,  X는 여행자
p.assertz("canTravel(X) :- traveler(X)")    # X가 여행자이면, X는 여행할 수 있다.

# 질의(query) 정의 및 결과 추출
for ans in p.query("grandfather(이한림,X)"):    # X의 할아버지는 이한림
    print('이한림의 손주:', ans["X"])   # 이한림의 손주 : 숙향

for sol in p.query("traveler(Y)"):  # Y는 여행자
    print('여행자:', sol["Y"])  # 여행자 Y : 이몽룡, 변학도

for ans in p.query("healthy(X), wealthy(X)"):   # X가 건강하고 풍족한 사람
    print('건강하고 여유있는 사람:', ans["X"])  # 건강하고 여유있는 사람 : 이몽룡, 변학도

for ans in p.query("husband(X,Y)"): # Y의 남편 X
    print('부부:', ans["X"], ans["Y"])  # 성참판 월매

for ans in p.query("spouse(X,Y)"):  # Y와 X는 배우자
    print('부부:', ans["X"], ans["Y"])  # 이한림 몽룡모

for ans in p.query("wife(X,Y)"):    # Y의 아내 X
    print(ans["Y"], '아내:', ans["X"])  # 성참판의 아내 월매, 이한림의 아내 몽룡모

print('이한림은 여행자이다:', bool(list(p.query("traveler(이한림)"))))  # 이한림이 여행자인지 확인
print('변학도는 여행자이다:', bool(list(p.query("traveler(변학도)"))))  # 변학도가 여행자인지 확인
