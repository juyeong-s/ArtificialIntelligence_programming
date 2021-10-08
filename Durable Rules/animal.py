from durable.lang import *

with ruleset('animal'):
    @when_all(c.first << (m.predicate == 'eats') & (m.object == 'flies'), # << 해당 조건을 만족하는 대상 지시하는 이름
              (m.predicate == 'lives') & (m.object == 'water') & (m.subject == c.first.subject))
    def frog(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'frog'})
        # 사실(fact)의 추가

    @when_all(c.first << (m.predicate == 'eats') & (m.object == 'potato'),
              (m.predicate == 'lives') & (m.object == 'land') & (m.subject == c.first.subject))
    def pig(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'pig'})

    @when_all(c.first << (m.predicate == 'eats') & (m.object == 'worms'))
    def bird(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'bird'})

    @when_all(c.first <<(m.predicate == 'eats') & (m.object == 'grass'),
              (m.predicate == 'lives') & (m.object == 'woods') & (m.subject == c.first.subject))
    def rabbit(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'rabbit'})

    @when_all(c.first <<(m.predicate == 'eats') & (m.object == 'fish'),
              (m.predicate == 'lives') & (m.object == 'lake') & (m.subject == c.first.subject))
    def duck(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'duck'})

    @when_all(c.first <<(m.predicate == 'lives') & (m.object == 'Frozen'),
              (m.predicate == 'eats') & (m.object == 'earthworm') & (m.subject == c.first.subject))
    def lizard(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'lizard'})

    @when_all(c.first <<(m.predicate == 'eats') & (m.object == 'honey'),
              (m.predicate == 'lives') & (m.object == 'woods') & (m.subject == c.first.subject))
    def bear(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'bear'})

    @when_all(c.first <<(m.predicate == 'eats') & (m.object == 'milk'),
              (m.predicate == 'lives') & (m.object == 'my house'), (m.predicate == 'howl') & (m.object == 'bowwow') & (m.subject == c.first.subject))
    def puppy(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'puppy'})

    @when_all(c.first <<(m.predicate == 'is') & (m.object == 'duck')|(m.predicate == 'is') & (m.object == 'puppy'))
    def white(c):
        c.assert_fact({'subject': c.first.subject, 'predicate': 'is', 'object': 'white'})

    @when_all((m.predicate == 'is') & (m.object == 'rabbit')|(m.predicate == 'is') & (m.object == 'pig'))
    def pink(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': 'is', 'object': 'pink'})

    @when_all((m.predicate == 'is') & (m.object == 'lizard'))
    def skyblue(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': 'is', 'object': 'skyblue'})

    @when_all((m.predicate == 'is') & (m.object == 'frog'))
    def green(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': 'is', 'object': 'green'})

    @when_all((m.predicate == 'is') & (m.object == 'bird'))
    def black(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': 'is', 'object': 'black'})

    @when_all((m.predicate == 'is') & (m.object == 'bear'))
    def yellowbrown(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': 'is', 'object': 'yellowbrown'})

    @when_all(+m.subject) # m.subject가 한번 이상
    def output(c):
        print('Fact: {0} {1} {2}'.format(c.m.subject, c.m.predicate, c.m.object))

assert_fact('animal', {'subject': 'Kermit', 'predicate': 'eats', 'object': 'flies'})
assert_fact('animal', {'subject': 'Kermit', 'predicate': 'lives', 'object': 'water'})
assert_fact('animal', {'subject': 'Piglet', 'predicate': 'eats', 'object': 'potato'})
assert_fact('animal', {'subject': 'Piglet', 'predicate': 'lives', 'object': 'land'})
assert_fact('animal', {'subject': 'Tweety', 'predicate': 'eats', 'object': 'worms'})
assert_fact('animal', {'subject': 'Donald', 'predicate': 'lives', 'object': 'lake'})
assert_fact('animal', {'subject': 'Donald', 'predicate': 'eats', 'object': 'fish'})
assert_fact('animal', {'subject': 'Bruni', 'predicate': 'lives', 'object': 'Frozen'})
assert_fact('animal', {'subject': 'Bruni', 'predicate': 'eats', 'object': 'earthworm'})
assert_fact('animal', {'subject': 'Snow ball', 'predicate': 'lives', 'object': 'woods'})
assert_fact('animal', {'subject': 'Snow ball', 'predicate': 'eats', 'object': 'grass'})
assert_fact('animal', {'subject': 'Pooh', 'predicate': 'eats', 'object': 'honey'})
assert_fact('animal', {'subject': 'Pooh', 'predicate': 'lives', 'object': 'woods'})
assert_fact('animal', {'subject': 'Snoopy', 'predicate': 'eats', 'object': 'milk'})
assert_fact('animal', {'subject': 'Snoopy', 'predicate': 'lives', 'object': 'my house'})
assert_fact('animal', {'subject': 'Snoopy', 'predicate': 'howl', 'object': 'bowwow'})
