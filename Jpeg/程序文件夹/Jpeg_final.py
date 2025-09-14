import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from skimage.metrics import structural_similarity as ssim


def convert_rgb_to_ycbcr(rgb):
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.4187, -0.0813]
    ])
    orig_shape = rgb.shape
    if len(orig_shape) == 3:
        rgb = rgb.reshape((orig_shape[0] * orig_shape[1], 3))
    ycbcr = np.dot(rgb, transform_matrix.T)
    return ycbcr.reshape(orig_shape)


def convert_ycbcr_to_rgb(ycbcr):
    transform_matrix = np.array([
        [1, 0, 1.402],
        [1, -0.344, -0.714],
        [1, 1.772, 0]
    ])
    orig_shape = ycbcr.shape
    if len(orig_shape) == 3:
        ycbcr = ycbcr.reshape((orig_shape[0] * orig_shape[1], 3))
    rgb = np.dot(ycbcr, transform_matrix.T)
    return rgb.reshape(orig_shape)


def HTable(nr_codes, std_table, huffman_table):
    current_index = 0
    code_val = 0
    for length in range(1, 17):
        for count in range(1, nr_codes[length - 1] + 1):
            huffman_table[std_table[current_index]] = bin(code_val)[2:].rjust(length, '0')
            current_index += 1
            code_val += 1
        code_val <<= 1


def RHTable(nr_codes, nr_values, reverse_huffman_table):
    pos = 0
    code_val = 0
    for bits in range(1, 17):
        for j in range(1, nr_codes[bits - 1] + 1):
            reverse_huffman_table[bin(code_val)[2:].rjust(bits, '0')] = nr_values[pos * 2:pos * 2 + 2]
            pos += 1
            code_val += 1
        code_val <<= 1



def calculate_compression(txt_file, png_file):
    txt_size = os.path.getsize(txt_file)
    png_size = os.path.getsize(png_file)
    ratio = txt_size / png_size
    print(f"文本大小: {txt_size} 字节")
    print(f"PNG大小: {png_size} 字节")
    print(f"压缩率: {ratio:.2f}")



def process_block_encoding(block, zigzag_order, quant_table, dc_table, ac_table, bitstream, prev_dc):
    block = block.astype(float)
    dct_block = cv2.dct(block)
    quant_block = np.round(dct_block / quant_table)

    zigzag_arr = [0] * 64
    non_zero_count = 0
    for idx in range(64):
        value = int(quant_block[int(idx / 8)][idx % 8])
        zigzag_arr[zigzag_order[idx]] = value
        if value != 0:
            non_zero_count += 1

    # DC系数处理
    if zigzag_arr[0] != 0:
        non_zero_count -= 1
    dc_diff = zigzag_arr[0] - prev_dc[0]
    prev_dc[0] = zigzag_arr[0]
    abs_value = bin(abs(dc_diff))[2:]
    size_bits = len(abs_value)
    if dc_diff < 0:
        abs_value = bin(abs(dc_diff) ^ (2 ** size_bits - 1))[2:].rjust(size_bits, '0')
    if dc_diff == 0:
        abs_value = ''
        size_bits = 0
    bitstream += dc_table[size_bits]
    bitstream += abs_value

    # AC系数处理
    zero_run = 0
    for idx in range(1, 64):
        if non_zero_count == 0:
            bitstream += ac_table[0]
            break
        if zigzag_arr[idx] == 0 and zero_run < 15:
            zero_run += 1
        else:
            ac_value = zigzag_arr[idx]
            abs_bits = bin(abs(ac_value))[2:]
            size_val = len(abs_bits)
            if ac_value < 0:
                abs_bits = bin(abs(ac_value) ^ (2 ** size_val - 1))[2:].rjust(size_val, '0')
            if ac_value == 0:
                size_val = 0
                abs_bits = ''
            bitstream += ac_table[zero_run * 16 + size_val]
            bitstream += abs_bits
            zero_run = 0
            if int(zigzag_arr[idx]) != 0:
                non_zero_count -= 1
    return bitstream



def decode_huffman_block(quant_table, zigzag_order, dc_reverse_table, ac_reverse_table, bitstream, prev_dc, pos, output,
                         row, col):
    coeffs = [0]
    num_coeffs = 0

    # DC系数解码
    for bits in range(11, 1, -1):
        code = dc_reverse_table.get(bitstream[pos[0]:pos[0] + bits])
        if code:
            pos[0] += bits
            num_coeffs += 1
            if code == '00':
                coeffs[0] = 0 + prev_dc[0]
                prev_dc[0] = coeffs[0]
                break
            size_val = int(code[1], 16)
            value_bits = bitstream[pos[0]:pos[0] + size_val]
            if value_bits and value_bits[0] == '0':
                value = -(int(value_bits, 2) ^ (2 ** size_val - 1))
            else:
                value = int(value_bits or '0', 2)
            coeffs[0] = value + prev_dc[0]
            prev_dc[0] = coeffs[0]
            pos[0] += size_val
            break

    # AC系数解码
    while num_coeffs < 64:
        for bits in range(16, 1, -1):
            code = ac_reverse_table.get(bitstream[pos[0]:pos[0] + bits])
            if code:
                pos[0] += bits
                if code == '00':
                    coeffs += [0] * (64 - num_coeffs)
                    num_coeffs = 64
                    break
                run_length = int(code[0], 16)
                size_val = int(code[1], 16)
                value_bits = bitstream[pos[0]:pos[0] + size_val]
                pos[0] += size_val
                if value_bits and value_bits[0] == '0':
                    value = -(int(value_bits, 2) ^ (2 ** size_val - 1))
                else:
                    value = int(value_bits or '0', 2)
                num_coeffs += run_length + 1
                coeffs += [0] * run_length
                coeffs.append(value)
                break

    # 重构量化块
    quant_block = np.zeros((8, 8))
    for idx in range(64):
        quant_block[int(idx / 8)][idx % 8] = coeffs[zigzag_order[idx]]

    # 逆量化和逆DCT
    dequant_block = quant_block * quant_table
    idct_block = cv2.idct(dequant_block)
    output[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = idct_block



def compress(img_data, qs=10):
    # 获取图像数据流宽高
    height, width, _ = img_data.shape
    YT = np.zeros([8, 8], dtype=np.uint8)
    CT = np.zeros([8, 8], dtype=np.uint8)

    # 标准亮度量化表
    LQT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                             [12, 12, 14, 19, 26, 58, 60, 55],
                                             [14, 13, 16, 24, 40, 57, 69, 56],
                                             [14, 17, 22, 29, 51, 87, 80, 62],
                                             [18, 22, 37, 56, 68, 109, 103, 77],
                                             [24, 35, 55, 64, 81, 104, 113, 92],
                                             [49, 64, 78, 87, 103, 121, 120, 101],
                                             [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)

    # 标准色度量化表
    CQT = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                               [18, 21, 26, 66, 99, 99, 99, 99],
                                               [24, 26, 56, 99, 99, 99, 99, 99],
                                               [47, 66, 99, 99, 99, 99, 99, 99],
                                               [99, 99, 99, 99, 99, 99, 99, 99],
                                               [99, 99, 99, 99, 99, 99, 99, 99],
                                               [99, 99, 99, 99, 99, 99, 99, 99],
                                               [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.uint8)
    # Z字型
    ZZ = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63])

    DC_Lu_NR = [0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    DC_Lu_Val = [4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11]

    AC_Lu_NR = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
    AC_Lu_Val = [0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
                                    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
                                    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
                                    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
                                    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
                                    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
                                    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
                                    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                                    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                                    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
                                    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
                                    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                                    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
                                    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
                                    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
                                    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
                                    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
                                    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
                                    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
                                    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                                    0xf9, 0xfa]

    DC_Chrom_NR = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    DC_Chrom_Val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    AC_Chrom_NR = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
    AC_Chrom_Val = [0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
                                      0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
                                      0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
                                      0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
                                      0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
                                      0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
                                      0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
                                      0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
                                      0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
                                      0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
                                      0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
                                      0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                                      0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
                                      0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
                                      0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
                                      0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
                                      0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
                                      0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
                                      0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
                                      0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                                      0xf9, 0xfa]

    for i in range(64):
        r, c = divmod(i, 8)
        # 亮度量化表
        tmp = int((LQT[r][c] * qs + 50) / 100)
        YT[r][c] = max(1, min(255, tmp))

        # 色度量化表
        tmp = int((CQT[r][c] * qs + 50) / 100)
        CT[r][c] = max(1, min(255, tmp))

    # 初始化哈夫曼编码表
    Y_DC = [0] * 12
    Y_AC = [0] * 256
    CbCr_DC = [0] * 12
    CbCr_AC = [0] * 256

    HTable(DC_Lu_NR, DC_Lu_Val, Y_DC)
    HTable(AC_Lu_NR, AC_Lu_Val, Y_AC)
    HTable(DC_Chrom_NR, DC_Chrom_Val, CbCr_DC)
    HTable(AC_Chrom_NR, AC_Chrom_Val, CbCr_AC)

    # 转成float类型
    img_data = img_data.astype(np.float64)

    # 存储最后的哈夫曼编码
    result = ''

    # 色彩空间转换
    YCbCr_data = convert_rgb_to_ycbcr(img_data)
    YCbCr_data = YCbCr_data.astype(int)
    Y_data, Cb_data, Cr_data = cv2.split(YCbCr_data)
    Y_data = Y_data - 128
    prev_DC_Y = [0]
    prev_DC_Cb = [0]
    prev_DC_Cr = [0]

    # CbCr降采样
    Cb_data=Cb_data[::2,::2]
    Cr_data=Cr_data[::2,::2]
    # 三个通道分别编码成独立数据流
    h, w = Y_data.shape
    # 分成8*8的块
    for i in range(h // 8):
        for j in range(w // 8):
            block = Y_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            result = process_block_encoding(block, ZZ, YT, Y_DC, Y_AC, result,
                                        prev_DC_Y)

    c_h, c_w = Cb_data.shape
    for i in range(c_h // 8):
        for j in range(c_w // 8):
            block = Cb_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            result = process_block_encoding(block, ZZ, CT, CbCr_DC, CbCr_AC,
                                        result, prev_DC_Cb)
            block = Cr_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            result = process_block_encoding(block, ZZ, CT, CbCr_DC, CbCr_AC,
                                        result, prev_DC_Cr)

    # 补足为8的整数倍
    pad_len = (8 - len(result) % 8) % 8
    result += '0' * pad_len
    res_data = ''
    for i in range(0, len(result), 8):
        temp = int(result[i:i + 8], 2)
        res_data += hex(temp)[2:].rjust(2, '0').upper()
        # FF需要添加00区分
        if temp == 255:
            res_data += '00'
    result = res_data
    res = ''

    # 添加jpeg文件头
    # SOI(文件头)
    res += 'FFD8'
    # APP0
    res += 'FFE000104A46494600010100000100010000'
    # DQT
    res += 'FFDB008400'

    for i in range(64):
        res += hex(YT[int(i / 8)][i % 8])[2:].rjust(2, '0')
    res += '01'
    for i in range(64):
        res += hex(CT[int(i / 8)][i % 8])[2:].rjust(2, '0')

    # SOF0
    res += 'FFC0001108'
    res += hex(height)[2:].rjust(4, '0')
    res += hex(width)[2:].rjust(4, '0')

    res += '03012200021101031101'
    # DHT定义huffman表
    res += 'FFC401A200'
    for i in DC_Lu_NR:
        res += hex(i)[2:].rjust(2, '0')
    for i in DC_Lu_Val:
        res += hex(i)[2:].rjust(2, '0')
    res += '10'
    for i in AC_Lu_NR:
        res += hex(i)[2:].rjust(2, '0')
    for i in AC_Lu_Val:
        res += hex(i)[2:].rjust(2, '0')
    res += '01'
    for i in DC_Chrom_NR:
        res += hex(i)[2:].rjust(2, '0')
    for i in DC_Chrom_Val:
        res += hex(i)[2:].rjust(2, '0')
    res += '11'
    for i in AC_Chrom_NR:
        res += hex(i)[2:].rjust(2, '0')
    for i in AC_Chrom_Val:
        res += hex(i)[2:].rjust(2, '0')
    # SOS扫描行开始，10个字节
    res += 'FFDA000C03010002110311003F00'
    # 压缩的图像数据
    res += result
    # EOI文件尾0
    res += 'FFD9'
    return res


def decompress(img):
    with open(img, 'rb') as f:
        img_data = f.read()
    res = ''
    for i in img_data:
        res += hex(i)[2:].rjust(2, '0').upper()
    ZigZag = [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63]

    rYTable = np.zeros((8, 8))
    for i in range(64):
        rYTable[int(i / 8)][i % 8] = int(res[50 + i * 2:52 + i * 2], 16)
    rCbCrTable = np.zeros((8, 8))
    for i in range(64):
        rCbCrTable[int(i / 8)][i % 8] = int(res[180 + i * 2:182 + i * 2], 16)

    # 获取SOF0宽高
    h = int(res[318:322], 16)
    w = int(res[322:326], 16)


    # 获取DHT定义huffman表
    DC_Lu_NR = [0] * 16
    for i in range(16):
        DC_Lu_NR[i] = int(res[356 + i * 2:358 + i * 2], 16)
    DC_Lu_Val = res[388:412]

    AC_Lu_NR = [0] * 16
    for i in range(16):
        AC_Lu_NR[i] = int(res[414 + i * 2:416 + i * 2], 16)
    AC_Lu_Val = res[446:770]

    DC_Ch_NR = [0] * 16
    for i in range(16):
        DC_Ch_NR[i] = int(res[772 + i * 2:774 + i * 2], 16)
    DC_Ch_Val = res[804:828]

    AC_Ch_NR = [0] * 16
    for i in range(16):
        AC_Ch_NR[i] = int(res[830 + i * 2:832 + i * 2], 16)
    AC_Ch_Val = res[862:1186]

    # 生成逆huffman编码表
    Reverse_Y_DC= {}
    Reverse_Y_AC= {}
    Reverse_CbCr_DC= {}
    Reverse_CbCr_AC= {}
    RHTable(DC_Lu_NR, DC_Lu_Val,Reverse_Y_DC)
    RHTable(AC_Lu_NR, AC_Lu_Val,Reverse_Y_AC)
    RHTable(DC_Ch_NR, DC_Ch_Val,Reverse_CbCr_DC)
    RHTable(AC_Ch_NR, AC_Ch_Val,Reverse_CbCr_AC)

    # 获取压缩的图像数据
    tmp_result = res[1214:-4]
    result = ''
    i = 0
    while i < len(tmp_result):
        tmp0 = tmp_result[i:i + 2]
        result += tmp0
        i += 2
        if (tmp0 == 'FF'):
            i += 2
    # 得到哈夫曼编码后的01字符串
    result = bin(int(result, 16))[2:].rjust(len(result) * 4, '0')
    prev_DC_Y = [0]
    prev_DC_Cb = [0]
    prev_DC_Cr = [0]
    pos = [0]
    # 逆huffman编码
    Y_data = np.zeros((h, w), dtype=int)
    Cb_data = np.zeros((h//2, w//2), dtype=int)
    Cr_data = np.zeros((h//2, w//2), dtype=int)
    for j in range(h // 8):
        for k in range(w // 8):
            decode_huffman_block(rYTable, ZigZag, Reverse_Y_DC, Reverse_Y_AC, result,
                               prev_DC_Y, pos, Y_data, j, k)
    cb_h, cb_w = Cb_data.shape
    for j in range(cb_h // 8):
        for k in range(cb_w // 8):
            decode_huffman_block(rCbCrTable, ZigZag, Reverse_CbCr_DC, Reverse_CbCr_AC,
                               result, prev_DC_Cb, pos, Cb_data, j, k)
            decode_huffman_block(rCbCrTable, ZigZag, Reverse_CbCr_DC, Reverse_CbCr_AC,
                               result, prev_DC_Cr, pos, Cr_data, j, k)

    Y_data = Y_data + 128
    Cb_data = np.repeat(np.repeat(Cb_data, 2, axis=0), 2, axis=1)
    Cr_data = np.repeat(np.repeat(Cr_data, 2, axis=0), 2, axis=1)
    YCbCr_data = cv2.merge([Y_data, Cb_data, Cr_data])
    img_data = convert_ycbcr_to_rgb(YCbCr_data)
    img_data = img_data.astype(np.uint8)
    return img_data


def execute_pipeline():

    # 图像处理流程
    input_img = './sga.png'
    original = cv2.imread(input_img)[:, :, (2, 1, 0)]
    cv2.imwrite('./reference.jpg', original)

    compressed_data = compress(original, 10)
    with open("result.txt", "w") as f:
        f.write(compressed_data)

    compressed_img = './compressed_img.jpg'
    with open(compressed_img, 'wb') as f:
        f.write(base64.b16decode(compressed_data.upper()))

    decompressed_img = decompress(compressed_img)
    rgb_img = cv2.cvtColor(decompressed_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./decompressed.jpg', rgb_img)

    calculate_compression("result.txt", input_img)

    # 质量评估
    orig_float = original.astype(np.float64) / 255.0
    decomp_float = decompressed_img.astype(np.float64) / 255.0
    mse_val = np.mean((orig_float - decomp_float) ** 2)
    psnr_val = 20 * np.log10(1.0 / np.sqrt(mse_val)) if mse_val > 0 else float('inf')
    ssim_val = ssim(orig_float, decomp_float, win_size=7, channel_axis=-1, data_range=1.0)

    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    axes[1].imshow(decompressed_img)
    axes[1].set_title('JPEG重建')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    execute_pipeline()