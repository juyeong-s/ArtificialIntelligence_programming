from durable.lang import *

with ruleset('risk'):
    @when_all(c.first << m.t == 'purchase',
              c.second << m.location != c.first.location)
    def fraud(c):
        print('이상거래 탐지 -> {0}, {1}'.format(c.first.location, c.second.location))
        
post('risk', {'t': 'purchase', 'location': 'US'})
post('risk', {'t': 'purchase', 'location': 'CA'})