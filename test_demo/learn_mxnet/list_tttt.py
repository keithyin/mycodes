class A(object):
    def __init__(self, i):
        self.i = i

a = A(1)
a2 = A(2)
list_A  = [a, a2]
for v in list_A:
    v = A(4)

for v in list_A:
    print(v.i)