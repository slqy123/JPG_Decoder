k = set(range(10))
with open('test.jpg', 'rb') as f:
    while (a:=f.read(1)):
        assert a not in k
