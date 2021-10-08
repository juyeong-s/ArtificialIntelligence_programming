from durable.lang import *

with ruleset('testRS'): # 규칙집합
    # antecedent(조건부). @when_any를사용하여 표기
    @when_all(m.subject == 'World') # m: rule이 적용되는 데이터
    def say_hello(c):
        print('Hello {0}'.format(c.m.subject))

post('testRS', {'subject': 'World'}) # 규칙 집합에 데이터 'subject': 'World' 전달

