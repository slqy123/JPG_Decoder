import  numpy as np
import struct
import matplotlib.pyplot as plt
from dataclasses import dataclass, astuple
from PIL import Image
import sys
# f=open(sys.argv[1],'rb')

# Bitmap File Header (14 bytes)

@dataclass
class BitmapFileHeader:
    bfType: int  # 2 19778，必须是BM字符串，对应的十六进制为0x4d42,十进制为19778，否则不是bmp格式文件
    bfSize: int  # 4 文件大小 以字节为单位(2-5字节)
    bfReserved1: int  # 2 保留，必须设置为0 (6-7字节)                           
    bfReserved2: int  # 2 保留，必须设置为0 (8-9字节)                               
    bfOffBits: int  # 4 从文件头到像素数据的偏移  (10-13字节)



# BItmap Information Header

@dataclass
class BItmapInformationHeader:
    biSize: int  # 4 此结构体的大小 (14-17字节)
    biWidth: int  # 4 图像的宽  (18-21字节)
    biHeight: int  # 4 图像的高  (22-25字节)
    biPlanes: int  # 2 表示bmp图片的平面属，显然显示器只有一个平面，所以恒等于1 (26-27字节)
    biBitCount: int  # 2 一像素所占的位数，一般为24   (28-29字节)
    biCompression: int  # 4 说明图象数据压缩的类型，0为不压缩。 (30-33字节)
    biSizeImage: int  # 4 像素数据所占大小, 这个值应该等于上面文件头结构中bfSize-bfOffBits (34-37字节)
    biXPelsPerMeter: int  # 4 说明水平分辨率，用象素/米表示。一般为0 (38-41字节)
    biYPelsPerMeter: int  # 说明垂直分辨率，用象素/米表示。一般为0 (42-45字节)
    biClrUsed: int  # 4 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）。 (46-49字节)
    biClrImportant: int  # 4 说明对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。(50-53字节)

def read_bmp(f):
    header = BitmapFileHeader(*struct.unpack('<HIHHI', f.read(14)))
    info = BItmapInformationHeader(*struct.unpack('<IllHHIIllII', f.read(40)))
    print(header, info)

    size = header.bfSize - header.bfOffBits
    assert info.biBitCount % 8 == 0
    bit_count = info.biBitCount//8
    data = np.array(struct.unpack('B'*size, f.read(size))).reshape((info.biHeight, info.biWidth, bit_count))
    rev = [2,1,0]
    if rev == 4:
        rev.append(3)
    data = data[::-1,:, rev]
    # data = data[:,:,:3]
    # data = data[::-1, :, ::-1]
    print(data, data.dtype)

    i = Image.fromarray(np.uint8(data))
    i.save('./out.png')

def save_bmp(data: np.ndarray):
    # 把长和宽扩大为最小的4的倍数
    pad = (4 - (data.shape[1] * data.shape[2])%4)%4
    extra_size = data.shape[0] * pad
    print(pad, extra_size)
    header = BitmapFileHeader(0x4d42, data.size+extra_size+54, 0, 0, 54)
    info = BItmapInformationHeader(40, data.shape[1], data.shape[0], 1, data.shape[2]*8, 0, data.size, 0, 0, 0, 0)

    header = astuple(header)
    info = astuple(info)

    header = struct.pack('<HIHHI', *header)
    info =struct.pack('<IllHHIIllII', *info)

    rev = [2,1,0]
    if rev == 4:
        rev.append(3)
    data = data[::-1,:, rev].reshape(data.shape[0], -1)
    data = np.pad(data, ((0,0), (0,pad)), mode='constant', constant_values=0)

    # print(data, data.dtype, np.max(data), np.min(data))
    raw_data = struct.pack('B'*data.size, *(data.flatten()%256))

    with open('out.bmp', 'wb') as f:
        f.write(header)
        f.write(info)
        f.write(raw_data)

if __name__ == '__main__':
    read_bmp(open(sys.argv[1], 'rb'))
