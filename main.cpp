/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_W4A4GroupMatmul_custom.h"
#else
#include "tikicpulib.h"
extern "C" void W4A4GroupMatmul_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t * , uint8_t *, uint8_t *, uint8_t *);
#endif
extern void GenerateTilingMatmul(const char *socVersion, uint8_t *tilingBuf);
extern void GenerateTilingScamul(const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    // size_t aFileSize = 262144 * sizeof(int16_t);
    // size_t bFileSize = 163840 * sizeof(int16_t);
    // size_t cFileSize = 655360 * sizeof(float);
    // size_t biasFileSize = 640 * sizeof(float);

    //
    int32_t M = 64;
    int32_t K = 32;
    int32_t N = 16;
    int32_t E = 4;

    //
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    //==============
    size_t input_X_Size = M * K * sizeof(int8_t);
    size_t input_W_Size = E * K * N * sizeof(int8_t);
    size_t input_X_scale_Size = M * sizeof(float);
    size_t input_W_scale_Size = E * N * sizeof(float);
    size_t input_group_list_Size = E * sizeof(int64_t);
    size_t y_Size = M * N * sizeof(half);
    //==============

    size_t tilingFileSize = sizeof(TCubeTiling);

    uint8_t *tilingBufMatmul = (uint8_t *)malloc(tilingFileSize);

    uint8_t *tilingBufScamul = (uint8_t *)malloc(tilingFileSize);

    GenerateTilingMatmul(socVersion, tilingBufMatmul);

    GenerateTilingScamul(socVersion, tilingBufScamul);

#ifdef CUSTOM_ASCEND310P//NOTUSE
    uint32_t blockDim = 2;
#else
    uint32_t blockDim = 1;
#endif

//uint32_t blockDim = 1;

#ifdef ASCENDC_CPU_DEBUG//=============USE
    //=============
        //printf("TestBlock1\n");
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(input_X_Size);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(input_W_Size);
    uint8_t *x_scale = (uint8_t *)AscendC::GmAlloc(input_X_scale_Size);
    uint8_t *w_scale = (uint8_t *)AscendC::GmAlloc(input_W_scale_Size);
    uint8_t *group_list = (uint8_t *)AscendC::GmAlloc(input_group_list_Size);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(y_Size);
    //=============

    uint8_t *tilingMatmul = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *tilingScamul = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    ///
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    //=============?

    ReadFile("./input/x.bin", input_X_Size, x, input_X_Size);
    ReadFile("./input/w.bin", input_W_Size, w, input_W_Size);
    ReadFile("./input/x_scale.bin", input_X_scale_Size, x_scale, input_X_scale_Size);
    ReadFile("./input/w_scale.bin", input_W_scale_Size, w_scale, input_W_scale_Size);
    ReadFile("./input/group_list.bin", input_group_list_Size, group_list, input_group_list_Size);

        //printf("TestBlock2\n");

    // ReadFile("./input/bias.bin", biasFileSize, bias, biasFileSize);

    memcpy_s(tilingMatmul, tilingFileSize, tilingBufMatmul, tilingFileSize);
    memcpy_s(tilingScamul, tilingFileSize, tilingBufScamul, tilingFileSize);
    ICPU_RUN_KF(W4A4GroupMatmul_custom, blockDim, x, w, x_scale, w_scale, group_list, y, workspace, tilingMatmul, tilingScamul);
    WriteFile("./output/output.bin", y, y_Size);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)w);
    AscendC::GmFree((void *)x_scale);
    AscendC::GmFree((void *)w_scale);
    AscendC::GmFree((void *)group_list);
    AscendC::GmFree((void *)y);


    AscendC::GmFree((void *)tilingMatmul);
    AscendC::GmFree((void *)tilingScamul);

    AscendC::GmFree((void *)workspace);

#else//==================================================================================================?????????
//This must for NO:CUSTOM_ASCEND310P&&NO:ASCENDC_CPU_DEBUG
//NOT   USED
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputAHost;
    uint8_t *inputADevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputAHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputBHost;
    uint8_t *inputBDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputCHost;
    uint8_t *outputCDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&outputCHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *inputBiasHost;
    uint8_t *inputBiasDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBiasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBiasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
    CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(matmul_leakyrelu_custom)
    (blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(inputADevice));
    CHECK_ACL(aclrtFreeHost(inputAHost));
    CHECK_ACL(aclrtFree(inputBDevice));
    CHECK_ACL(aclrtFreeHost(inputBHost));
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", outputCHost, cFileSize);
    CHECK_ACL(aclrtFree(outputCDevice));
    CHECK_ACL(aclrtFreeHost(outputCHost));
    CHECK_ACL(aclrtFree(inputBiasDevice));
    CHECK_ACL(aclrtFreeHost(inputBiasHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBufMatmul);
    free(tilingBufScamul);
    return 0;
}