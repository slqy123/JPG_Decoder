from bit_reader import BitReader

data = open('test.jpg', 'rb').read()

br = BitReader(bytearray(data))

# br.get_all()
while True:
    # br.r1()
    br.read1()
