class BitReader:
    def __init__(self, array: bytearray) -> None:
        self.array = array
        self.c_bit = 0
        self.c_byte = 0

    def read1(self):
        res = (self.array[self.c_byte] >> (7 - self.c_bit)) & 1
        self.c_bit += 1
        if self.c_bit == 8:
            self.c_byte += 1
            self.c_bit = 0
        return res

    def read_iter(self):
        res = 0
        while True:
            res = (res << 1) + self.read1()
            yield res

    def read(self, n):
        res = 0
        for _ in range(n):
            res = (res << 1) + self.read1()
        return res
    

    def align(self):
        if self.c_bit > 0:
            self.c_bit = 0
            self.c_byte += 1

