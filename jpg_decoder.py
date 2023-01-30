import struct
from itertools import product
from pathlib import Path
from io import BufferedReader
from typing import List, Optional, Tuple, Dict, Literal

import click
import numpy as np

from consts import *
from bit_reader import BitReader
from bmp_reader import save_bmp

from zigzag import inverse_zigzag_single

class JPGImage:
    def __init__(self, f: BufferedReader) -> None:
        self.f = f

        self.sof_flag = False

        self.quantization_tables = np.empty((4, 8, 8), dtype=np.uint16)
        self.quantization_tables_mask = [False] * 4

        self.width: int
        self.height: int
        self.mcus: Optional[np.ndarray] = None

        self.color_components: Dict[int, ColorComponent] = {}
        self.components_num: int

        self.huffman_tables_DC: Dict[int, HuffmanTable] = {}
        self.huffman_tables_AC: Dict[int, HuffmanTable] = {}

        self.start_of_selection = 0
        self.end_of_selection = 63
        self.successive_approximation = 0

        self.huffman_data = bytearray()
        self.comments = []

        self.restart_interval = None

        self.read_header()
        self.read_data()
        self.decode_huffman_data()
    
    @staticmethod
    def divide(b: int) -> Tuple[int, int]:
        """将一字节的整型数据拆分成高半字节和低半字节的整型数据"""
        assert 0 <= b <= 255
        return b >> 4, b & 0xF

    def read_marker(self):
        prev, marker = struct.unpack('>BB', self.f.read(2))
        assert prev == 0xFF
        while marker == 0xFF:
            marker, = struct.unpack('>B', self.f.read(1))
        return marker
    
    def read_length(self) -> int:
        length, = struct.unpack('>H', self.f.read(2))
        return length - 2
    
    def read_one_DQT_map(self) -> int:
        dqt_info, = struct.unpack('B', self.f.read(1))
        table_type, table_id = self.divide(dqt_info)
        assert table_type in (0,1)  # 0代表量化表元素的长度为一字节，1代表两字节
        assert 0 <= table_id < 4  # 最多只能有四个量化表
        table_size = table_type * 64 + 64  # 整个量化表的长度，因为一共8x8的元素，每个元素占一个或两个字节
        
        self.quantization_tables_mask[table_id] = True
        if table_type == 0:
            array = np.array(struct.unpack('B'*64, self.f.read(64)))
            self.quantization_tables[table_id] = inverse_zigzag_single(array)
        else:
            array = np.array(struct.unpack('H'*64, self.f.read(128)))
            self.quantization_tables[table_id] = inverse_zigzag_single(array)
        return table_size + 1

    def read_SOF0(self):
        assert not self.sof_flag  # sof只能有一个
        self.sof_flag = True

        length = self.read_length()
        precision, self.height, self.width, self.components_num = struct.unpack('>BHHB', self.f.read(6))
        assert precision == 8  # 精度必须为8，代表每个通道的颜色信息用8bit表示
        assert self.height > 0
        assert self.width > 0
        assert self.components_num in (1,3)  # 通道数，1个代表灰度，3个代表YCbCr
        assert length == 6 + self.components_num * 3  # 第一个unpack有6个字节，后面一个for循环，每次3个字节
        
        for _ in range(self.components_num):
            component_id, sampling_factor, quantization_table_id= struct.unpack('BBB', self.f.read(3))
            assert 0 < component_id < 4  # id 是从1开始的，而4,5是YIQ图片，不打算支持
            color_component = ColorComponent(
                *self.divide(sampling_factor), quantization_table_id)
            self.color_components[component_id] = color_component
    
    def read_DRI(self):
        length = self.read_length()
        assert length == 2
        self.restart_interval, = struct.unpack('>H', self.f.read(2))

    def read_DHT(self):
        """
        可能有多个Huffman Table
        symbol 是一个字节，分高字节和低字节
        高字节表示后面0的个数，范围是0-15
        低字节表示该系数对应编码的长度(例如5(0b101)长度就是3)，范围是1-10
        至此有了160个symbol，再加上两个特殊的:
            0xF0: 跳过16个0，然后不读取任何编码
            0x00: 后面全部都是0，不管有多少位
        因此共有162个symbols
        """
        length = self.read_length()
        while length > 0:
            table_info, = struct.unpack('>B', self.f.read(1))
            is_AC, table_id = self.divide(table_info) # ac和dc的table_id互相独立
            assert 0<= table_id <= 3
            if is_AC:
                table = self.huffman_tables_AC[table_id] = HuffmanTable()
            else:
                table = self.huffman_tables_DC[table_id] = HuffmanTable()
            
            self.symbol_counts = struct.unpack('>'+'B'*16, self.f.read(16))
            length -= 17
            code = 0
            for index, count in enumerate(self.symbol_counts):
                if count == 0:
                    code *= 2
                    continue
                symbols = struct.unpack('>'+'B'*count, self.f.read(count))
                for symbol in symbols:
                    table.symbols[(index+1, code)] = symbol
                    code += 1
                code *= 2
                length -= count
        assert length == 0
    
    def read_SOS(self):
        length = self.read_length()
        assert self.components_num > 0 # 必须要先读到components
        components_num, = struct.unpack('>B', self.f.read(1))
        assert components_num <= self.components_num
        for _ in range(components_num):
            component_id, huffman_table_id = struct.unpack('>BB', self.f.read(2))
            component = self.color_components[component_id]
            assert not component.used_in_scan  # color components 不会重复
            component.used_in_scan = True
            component.huffman_DC_table_id, component.huffman_AC_table_id = self.divide(huffman_table_id)
        self.start_of_selection, \
        self.end_of_selection, \
        self.successive_approximation = struct.unpack('>BBB', self.f.read(3))
        assert self.start_of_selection == 0 and self.end_of_selection == 63 and self.successive_approximation == 0
        assert (length - 4 - 2 * components_num) == 0

    def read_comment(self):
        length = self.read_length()
        comment = self.f.read(length)
        self.comments.append(comment)

    def print_DQT_table(self):
        for i in range(4):
            if self.quantization_tables_mask[i]:
                print(f'table id={i}:')
                print(self.quantization_tables[i])
    
    def print_SOF0(self):
        for i, item in self.color_components.items():
            if item is not None:
                print(f'component id = {i}')
                print(item)
    
    def print_DHT(self):
        for table_id, table in self.huffman_tables_DC.items():
            print(f'DC Huffman Table id={table_id}')
            for key, value in table.symbols.items():
                keylen, code = key
                print(f'code: {bin(code)[2:].zfill(keylen)}\tsymbol: %.2x' % value)
        for table_id, table in self.huffman_tables_AC.items():
            print(f'AC Huffman Table id={table_id}')
            for key, value in table.symbols.items():
                keylen, code = key
                print(f'code: {bin(code)[2:].zfill(keylen)}\tsymbol: %.2x' % value)

    def read_header(self):
        assert self.read_marker() == SOI

        while True:
            marker = self.read_marker()
            
            if APP0 <= marker <= APP15:
                print(f'Read APP{marker & 0x0F} marker')
                length = self.read_length()
                # print('data is:', self.f.read(length))
                self.f.read(length)
            elif marker == DQT:
                print(f'Read DQT marker')
                length = self.read_length()
                while length > 0:
                    read_size = self.read_one_DQT_map()
                    length -= read_size
                assert length == 0
            elif marker == SOF0:
                print('Read SOF0 marker')
                self.read_SOF0()
            elif marker == DRI:
                print('Read DRI marker')
                self.read_DRI()
            elif marker == DHT:
                print('Read DHT marker')
                self.read_DHT()
            elif marker == SOS:
                print('Read SOS marker')
                self.read_SOS()
                break
            elif marker == COM:
                print('Read COM marker')
                self.read_comment()
            elif (JPG0 <= marker <= JPG13) or marker in (DNL, DHP, EXP):
                print('Read useless marker')
                length = self.read_length()
                self.f.read(length)
            elif marker == TEM:
                continue
            else:
                print('unkow marker: 0x%.2x' % marker)
                exit(-1)
        print('width:', self.width, ' ', 'height:', self.height)
        self.print_DQT_table()
        self.print_SOF0()
        self.print_DHT()

    def read_data(self):
        next: int = struct.unpack('>B', self.f.read(1))[0]
        while True:
            current = next
            next, = struct.unpack('>B', self.f.read(1))
            if current != 0xFF:
                self.huffman_data.append(current)
                continue
            
            # for current == 0xFF
            if next == EOI:
                print('End of Image')
                print(f'Huffman data size = {len(self.huffman_data)} bytes')
                break
            elif next == 0:  # 0xff0x00 代表这里的ff是huffman编码不是marker
                self.huffman_data.append(current)
                next, = struct.unpack('>B', self.f.read(1))
            elif RST0 <= next <= RST7:  # restart marker 解码时不重要
                next, = struct.unpack('>B', self.f.read(1))
            elif next == 0xFF:
                pass  # 可能多个ff
            else:
                print('Unexpected marker:', '%.2x' % next)
                exit(-1)

    def decode_huffman_data(self):
        def read_symbol(htable: HuffmanTable):
            for i, code in zip(range(16), br.read_iter()):
                if (key:=(i+1, code)) in htable.symbols:
                    # print("%d|%.2x" % (code, htable.symbols[key]), end=' ')
                    return htable.symbols[key]
            else:
                print('failed to match any symbol')
                exit(-1)
        def read_coefficient(length):
            if length == 0:
                return 0
            coeff = br.read(length)
            if coeff < 2**(length-1):
                coeff = coeff + 1 - 2**(length)
            return coeff


        def decode_one_mcu(color_component_id: int) -> np.ndarray:
            # print(f'\n{self.color_components[color_component_id]}')
            dc_table_id = self.color_components[color_component_id].huffman_DC_table_id
            ac_table_id = self.color_components[color_component_id].huffman_AC_table_id
            dc_table = self.huffman_tables_DC[dc_table_id]
            ac_table = self.huffman_tables_AC[ac_table_id]
            mcu = np.empty(64, dtype=int)

            dc_symbol = read_symbol(dc_table)
            assert dc_symbol <= 11
            coeff = read_coefficient(dc_symbol)  # dc symbol 只有后半部分，也就是要读的长度，而没有前面的要读多少个零
            # print(dc_symbol, coeff)
            mcu[0] = coeff + previous_coeff4DC.get(color_component_id, 0)
            previous_coeff4DC[color_component_id] = mcu[0]

            i = 1
            while i < 64:
                ac_symbol = read_symbol(ac_table)
                assert ac_symbol != 0xFF
                if ac_symbol == 0x00:
                    # 结束了，后面全是零
                    while i < 64:
                        mcu[i] = 0
                        i += 1
                else:
                    zero_num, coeff_len = self.divide(ac_symbol)
                    if ac_symbol == 0xF0:  # 特殊符号，表示跳过16个，不读取任何一个
                        zero_num = 16
                    # 补充零
                    for _ in range(zero_num):
                        mcu[i] = 0
                        i += 1

                    assert coeff_len < 11  # ac table 的长度不会超过10，但dc可以为11
                    if coeff_len > 0:
                        coeff = read_coefficient(coeff_len)
                        mcu[i] = coeff
                        i += 1
            # res = inverse_zigzag_single(mcu)
            # print(mcu)
            # print(res)
            # input()
            # return res
            return inverse_zigzag_single(mcu)


        mcu_width = (self.width+7)//8
        mcu_height = (self.height+7)//8  # 8是最小单位，图片宽度不够要补全
        self.mcus = np.zeros((mcu_height, mcu_width, self.components_num, 8, 8), dtype=int)
        previous_coeff4DC = {}  # 这里保存的是3个components里的上一个coeff的值，当前的coeff=之前的coeff+读取到的值
        br = BitReader(self.huffman_data)

        for i, j, k in product(range(mcu_height), range(mcu_width), range(self.components_num)):
            if (self.restart_interval is not None) and ((i*mcu_width + j) % self.restart_interval == 0):
                print('reset interval')
                previous_coeff4DC = {}
                br.align()
            self.mcus[i, j, k] = decode_one_mcu(color_component_id=k+1)
        print(np.max(self.mcus))

    def save(self):
        assert isinstance(self.mcus, np.ndarray)
        save_bmp(self.mcus.transpose((0,3,1,4,2)).reshape((
            self.mcus.shape[0] * self.mcus.shape[3], self.mcus.shape[1] * self.mcus.shape[4], self.mcus.shape[2]
            )))

@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(path):
    with open(path, 'rb') as f:
        jpg = JPGImage(f)
        jpg.save()

if __name__ == '__main__':
    main()
