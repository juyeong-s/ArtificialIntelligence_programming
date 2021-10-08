from durable.lang import *

with ruleset('bookstore'):
    # this rule will trigger for events with status
    @when_all(+m.status)
    def event(c):
        print('bookstore-> Reference {0} status {1}'.format(c.m.reference, c.m.status))

    @when_all(+m.name)
    def fact(c):
        print('bookstore-> Added {0}'.format(c.m.name))
        
    # this rule will be triggered when the fact is retracted
    @when_all(none(+m.name))
    def empty(c):
        print('bookstore-> No books')

# will not throw because the fact assert was successful 
assert_fact('bookstore', {
    'name': 'The new book',
    'seller': 'bookstore',
    'reference': '75323',
    'price': 500
})

# will throw MessageObservedError because the fact has already been asserted 
try:
    assert_fact('bookstore', {
        'reference': '75323',
        'name': 'The new book',
        'price': 500,
        'seller': 'bookstore'
    })
except BaseException as e:
    print('bookstore expected {0}'.format(e.message))

# will not throw because a new event is being posted
post('bookstore', {
    'reference': '75323',
    'status': 'Active'
})

# will not throw because a new event is being posted
post('bookstore', {
    'reference': '75323',
    'status': 'Active'
})

retract_fact('bookstore', {
    'reference': '75323',
    'name': 'The new book',
    'price': 500,
    'seller': 'bookstore'
})
