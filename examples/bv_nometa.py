from qwerty import *

#@qpu
#def kernel() -> bit[2]:
#    return (('0'+'1')*('0'+'1')
#            | {'1'*'0', '1'*'1'} >> {-'1'*'0', -'1'*'1'}
#            | {'0'+'1','0'-'1'}*{'0'+'1','0'-'1'} >> {'0','1'}*{'0','1'}
#            | ({'0','1'}*{'0','1'}).measure)

# Easier to debug issue with this smaller example
@qpu
def kernel() -> bit[2]:
    return ('0'*'0'
            | {'0'+'1','0'-'1'}*{'0'+'1','0'-'1'} >> {'0','1'}*{'0','1'}
            | ({'0','1'}*{'0','1'}).measure)


histogram(kernel(shots=1024))
