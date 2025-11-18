#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os


# def gen_golden_data():
#     M = 1024
#     K = 256
#     N = 512
#     E = 4
    

#     input_a = np.random.randint(1, 10, [M, K]).astype(np.float16)
#     input_b = np.random.randint(1, 10, [K, N]).astype(np.float16)
#     input_bias = np.random.randint(1, 10, [N]).astype(np.float32)
#     alpha = 0.001
#     golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
#     golden = np.where(golden >= 0, golden, golden * alpha)
#     os.system("mkdir -p input")
#     os.system("mkdir -p output")
#     input_a.tofile("./input/x1_gm.bin")
#     input_b.tofile("./input/x2_gm.bin")
#     input_bias.tofile("./input/bias.bin")
#     golden.tofile("./output/golden.bin")

def gen_test_data():  
    M = 64 
    K = 32 
    N = 16 
    E = 4 
    Group_list = np.array([16,16,16,16], dtype=np.int64)  
    
    # 生成指定大小的输入矩阵  
    input_X = np.random.randint(1, 10, [M, K]).astype(np.int8)


    input_W = np.random.randint(1, 10, [E, K, N]).astype(np.int8)
    # print("Input X:\n")
    # for i in range(64):
    #     for j in range(32):
    #         print(f"{input_X[i][j]} ", end='')
    #     print("\n")
    
    # print("Input W:\n")
    # for i in range(32):
    #     for j in range(16):
    #         print(f"{input_W[0][i][j]} ", end='')
    #     print("\n")
    input_X_Scale = np.random.randn(M).astype(np.float32)
    input_W_Scale = np.random.randn(E, N).astype(np.float32)
    # 计算矩阵  
    Y = GroupedMatmul(input_X, input_W, input_X_Scale, input_W_Scale,Group_list)   

    # 创建输出文件夹  
    os.makedirs("input", exist_ok=True)  
    os.makedirs("output", exist_ok=True)  

    # 保存
    input_X.tofile("./input/x.bin")  
    input_W.tofile("./input/w.bin")  
    input_X_Scale.tofile("./input/x_scale.bin")  
    input_W_Scale.tofile("./input/w_scale.bin")  
    Group_list.tofile("./input/group_list.bin")

    Y.tofile("./output/golden.bin")  
    Y_float = Y.astype(np.float32)  
    # 使用 np.savetxt 将转换后的数组保存为 TXT 文件  
    np.savetxt("./output/golden.txt", Y_float, fmt='%.4f')  # fmt='%.4f' 指定保留四位小数  

def GroupedMatmul(X, W, X_Scale, W_Scale, Group_list):
    M, K = X.shape
    E, K_w, N = W.shape
    assert K == K_w, "X和W的K维度必须相等"
    assert np.sum(Group_list) == M, "Group_list的和必须等于M"

    Res = np.zeros((M, N), dtype=np.float16)

    current_row = 0
    for expert_idx in range(E):
        # 获取当前专家的行数
        m_i = Group_list[expert_idx]
        
        if m_i == 0:
            continue
            
        # 从X中截取当前专家的数据
        x_i = X[current_row:current_row + m_i, :]  # shape (m_i, K)
        
        # 获取当前专家的权重
        w_i = W[expert_idx]  # shape (K, N)

        matmul_int32 = np.dot(x_i.astype(np.int32), w_i.astype(np.int32))
        matmul_float32 = matmul_int32.astype(np.float32)
        
        # 应用缩放因子
        x_scale_i = X_Scale[current_row:current_row + m_i]  # shape (m_i,)
        w_scale_i = W_Scale[expert_idx]  # shape (N,)
        
        # 广播缩放因子并应用
        scale_matrix = x_scale_i[:, np.newaxis] * w_scale_i[np.newaxis, :]  # shape (m_i, N)
        scaled_result = matmul_float32 * scale_matrix
        
        # 存储到输出张量
        Res[current_row:current_row + m_i, :] = scaled_result.astype(np.float16)
        current_row += m_i
        # print("======\n")
        # print(Step1Res)
        # print("======\n")
    
    return Res

if __name__ == "__main__":
    gen_test_data()
