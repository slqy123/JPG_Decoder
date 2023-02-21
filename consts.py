from math import cos, pi
from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Dict

def l(length, default=lambda:None):
    return [default() for _ in range(length)]

M_PI = pi
# Start of Frame markers, non-differential, Huffman coding
SOF0 = 0xC0 # Baseline DCT
SOF1 = 0xC1 # Extended sequential DCT
SOF2 = 0xC2 # Progressive DCT
SOF3 = 0xC3 # Lossless (sequential)

# Start of Frame markers, differential, Huffman coding
SOF5 = 0xC5 # Differential sequential DCT
SOF6 = 0xC6 # Differential progressive DCT
SOF7 = 0xC7 # Differential lossless (sequential)

# Start of Frame markers, non-differential, arithmetic coding
SOF9 = 0xC9 # Extended sequential DCT
SOF10 = 0xCA # Progressive DCT
SOF11 = 0xCB # Lossless (sequential)

# Start of Frame markers, differential, arithmetic coding
SOF13 = 0xCD # Differential sequential DCT
SOF14 = 0xCE # Differential progressive DCT
SOF15 = 0xCF # Differential lossless (sequential)

# Define Huffman Table(s)
DHT = 0xC4

# JPEG extensions
JPG = 0xC8

# Define Arithmetic Coding Conditioning(s)
DAC = 0xCC

# Restart interval Markers
RST0 = 0xD0
RST1 = 0xD1
RST2 = 0xD2
RST3 = 0xD3
RST4 = 0xD4
RST5 = 0xD5
RST6 = 0xD6
RST7 = 0xD7

# Other Markers
SOI = 0xD8 # Start of Image
EOI = 0xD9 # End of Image
SOS = 0xDA # Start of Scan
DQT = 0xDB # Define Quantization Table(s)
DNL = 0xDC # Define Number of Lines
DRI = 0xDD # Define Restart Interval
DHP = 0xDE # Define Hierarchical Progression
EXP = 0xDF # Expand Reference Component(s)

# APPN Markers
APP0 = 0xE0
APP1 = 0xE1
APP2 = 0xE2
APP3 = 0xE3
APP4 = 0xE4
APP5 = 0xE5
APP6 = 0xE6
APP7 = 0xE7
APP8 = 0xE8
APP9 = 0xE9
APP10 = 0xEA
APP11 = 0xEB
APP12 = 0xEC
APP13 = 0xED
APP14 = 0xEE
APP15 = 0xEF

# Misc Markers
JPG0 = 0xF0
JPG1 = 0xF1
JPG2 = 0xF2
JPG3 = 0xF3
JPG4 = 0xF4
JPG5 = 0xF5
JPG6 = 0xF6
JPG7 = 0xF7
JPG8 = 0xF8
JPG9 = 0xF9
JPG10 = 0xFA
JPG11 = 0xFB
JPG12 = 0xFC
JPG13 = 0xFD
COM = 0xFE
TEM = 0x01

@dataclass
class HuffmanTable:
    symbol_counts: Tuple = field(default_factory=tuple)
    symbols: Dict[Tuple[int, int], int] = field(default_factory=dict)  # key为code长度和code的uint整形

@dataclass
class ColorComponent:
    horizontal_sampling_factor: int
    vertical_sampling_factor: int
    quantization_table_id: int
    huffman_DC_table_id: int = -1
    huffman_AC_table_id: int = -1
    used_in_frame = False
    used_in_scan = False


YCbCr2RG = np.array([
    [1, 0, 1.402],
    [1, -0.344, -0.714],
    [1, 1.772, 0]
    ])

# class JPGImage:
#     QuantizationTable quantizationTables[4]

#     ColorComponent colorComponents[3]

#     frameType = 0
#     height = 0
#     width = 0
#     numComponents = 0
#     zeroBased = False

#     componentsInScan = 0
#     startOfSelection = 0
#     endOfSelection = 0
#     successiveApproximationHigh = 0
#     successiveApproximationLow = 0

#     restartInterval = 0

#     Block* blocks = nullptr

#     valid = True

#     blockHeight = 0
#     blockWidth = 0
#     blockHeightReal = 0
#     blockWidthReal = 0

#     horizontalSamplingFactor = 0
#     verticalSamplingFactor = 0


# class BMPImage:
#     height = 0
#     width = 0

#     Block* blocks = nullptr

#     blockHeight = 0
#     blockWidth = 0


# zigZagMap = [
#     0,   1,  8, 16,  9,  2,  3, 10,
#     17, 24, 32, 25, 18, 11,  4,  5,
#     12, 19, 26, 33, 40, 48, 41, 34,
#     27, 20, 13,  6,  7, 14, 21, 28,
#     35, 42, 49, 56, 57, 50, 43, 36,
#     29, 22, 15, 23, 30, 37, 44, 51,
#     58, 59, 52, 45, 38, 31, 39, 46,
#     53, 60, 61, 54, 47, 55, 62, 63
# ]

# # IDCT scaling factors
# m0 = 2.0 * cos(1.0 / 16.0 * 2.0 * M_PI)
# m1 = 2.0 * cos(2.0 / 16.0 * 2.0 * M_PI)
# m3 = 2.0 * cos(2.0 / 16.0 * 2.0 * M_PI)
# m5 = 2.0 * cos(3.0 / 16.0 * 2.0 * M_PI)
# m2 = m0 - m5
# m4 = m0 + m5

# s0 = cos(0.0 / 16.0 * M_PI) / sqrt(8)
# s1 = cos(1.0 / 16.0 * M_PI) / 2.0
# s2 = cos(2.0 / 16.0 * M_PI) / 2.0
# s3 = cos(3.0 / 16.0 * M_PI) / 2.0
# s4 = cos(4.0 / 16.0 * M_PI) / 2.0
# s5 = cos(5.0 / 16.0 * M_PI) / 2.0
# s6 = cos(6.0 / 16.0 * M_PI) / 2.0
# s7 = cos(7.0 / 16.0 * M_PI) / 2.0

# # standard tables

# const QuantizationTable qTableY50 = {
#     {
#         16,  11,  10,  16,  24,  40,  51,  61,
#         12,  12,  14,  19,  26,  58,  60,  55,
#         14,  13,  16,  24,  40,  57,  69,  56,
#         14,  17,  22,  29,  51,  87,  80,  62,
#         18,  22,  37,  56,  68, 109, 103,  77,
#         24,  35,  55,  64,  81, 104, 113,  92,
#         49,  64,  78,  87, 103, 121, 120, 101,
#         72,  92,  95,  98, 112, 100, 103,  99
#     },
#     True


# const QuantizationTable qTableCbCr50 = {
#     {
#         17, 18, 24, 47, 99, 99, 99, 99,
#         18, 21, 26, 66, 99, 99, 99, 99,
#         24, 26, 56, 99, 99, 99, 99, 99,
#         47, 66, 99, 99, 99, 99, 99, 99,
#         99, 99, 99, 99, 99, 99, 99, 99,
#         99, 99, 99, 99, 99, 99, 99, 99,
#         99, 99, 99, 99, 99, 99, 99, 99,
#         99, 99, 99, 99, 99, 99, 99, 99
#     },
#     True


# const QuantizationTable qTableY75 = {
#     {
#         16/2,  11/2,  10/2,  16/2,  24/2,  40/2,  51/2,  61/2,
#         12/2,  12/2,  14/2,  19/2,  26/2,  58/2,  60/2,  55/2,
#         14/2,  13/2,  16/2,  24/2,  40/2,  57/2,  69/2,  56/2,
#         14/2,  17/2,  22/2,  29/2,  51/2,  87/2,  80/2,  62/2,
#         18/2,  22/2,  37/2,  56/2,  68/2, 109/2, 103/2,  77/2,
#         24/2,  35/2,  55/2,  64/2,  81/2, 104/2, 113/2,  92/2,
#         49/2,  64/2,  78/2,  87/2, 103/2, 121/2, 120/2, 101/2,
#         72/2,  92/2,  95/2,  98/2, 112/2, 100/2, 103/2,  99/2
#     },
#     True


# const QuantizationTable qTableCbCr75 = {
#     {
#         17/2, 18/2, 24/2, 47/2, 99/2, 99/2, 99/2, 99/2,
#         18/2, 21/2, 26/2, 66/2, 99/2, 99/2, 99/2, 99/2,
#         24/2, 26/2, 56/2, 99/2, 99/2, 99/2, 99/2, 99/2,
#         47/2, 66/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2,
#         99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2,
#         99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2,
#         99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2,
#         99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2, 99/2
#     },
#     True


# const QuantizationTable qTableY100 = {
#     {
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1
#     },
#     True


# const QuantizationTable qTableCbCr100 = {
#     {
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1
#     },
#     True


# const QuantizationTable* const qTables50[]  = {  &qTableY50,  &qTableCbCr50,  &qTableCbCr50 }
# const QuantizationTable* const qTables75[]  = {  &qTableY75,  &qTableCbCr75,  &qTableCbCr75 }
# const QuantizationTable* const qTables100[] = { &qTableY100, &qTableCbCr100, &qTableCbCr100 }

# HuffmanTable hDCTableY = [
#     { 0, 0, 1, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12 },
#     { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b },
#     {},
#     False]


# HuffmanTable hDCTableCbCr = [
#     { 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12 },
#     { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b },
#     {},
#     False]


# HuffmanTable hACTableY = [
#     { 0, 0, 2, 3, 6, 9, 11, 15, 18, 23, 28, 32, 36, 36, 36, 37, 162 },
#     {
#         0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
#         0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
#         0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
#         0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
#         0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
#         0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
#         0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
#         0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
#         0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
#         0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
#         0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
#         0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
#         0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
#         0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
#         0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
#         0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
#         0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
#         0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
#         0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
#         0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
#         0xf9, 0xfa
#     },
#     {},
#     False]


# HuffmanTable hACTableCbCr = [
#     { 0, 0, 2, 3, 5, 9, 13, 16, 20, 27, 32, 36, 40, 40, 41, 43, 162 },
#     {
#         0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
#         0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
#         0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
#         0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
#         0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
#         0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
#         0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
#         0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
#         0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
#         0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
#         0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
#         0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
#         0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
#         0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
#         0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
#         0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
#         0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
#         0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
#         0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
#         0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
#         0xf9, 0xfa
#     },
#     {},
#     False]


# HuffmanTable* const dcTables[] = { &hDCTableY, &hDCTableCbCr, &hDCTableCbCr }
# HuffmanTable* const acTables[] = { &hACTableY, &hACTableCbCr, &hACTableCbCr }

