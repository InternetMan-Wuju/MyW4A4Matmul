/**
 * @file matmul_leakyrelu_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

/**
  * @brief  Generate matmul tiling.
  * @param  socVersion: Platform socversion.
  * @param  tilingBuf data buffer.
  */
void GenerateTilingMatmul(const char *socVersion, uint8_t *tilingBuf)
{
  //未来需要支持动态Shape
  //No tiling for now , 25.11.4
    constexpr int32_t M = 64;
    constexpr int32_t K = 32;
    constexpr int32_t N = 16;
    
    //int E = 4;?

    TPosition XPosition = TPosition::VECOUT;
    CubeFormat XFormat = CubeFormat::ND;
    DataType XDtype = DataType::DT_INT8;
    bool isTransX = false;

    TPosition WPosition = TPosition::VECOUT;
    CubeFormat WFormat = CubeFormat::ND;
    DataType WDtype = DataType::DT_INT8;
    bool isTransW = false;


    // TPosition GroupListPosition = TPosition::GM;
    // CubeFormat GroupListFormat = CubeFormat::ND;
    // DataType GroupListDtype = DataType:::DT_INT64;
    // bool isTransGroupList = false;

    // TPosition YPosition = TPosition::GM;
    // CubeFormat YFormat = CubeFormat::ND;
    // DataType YDtype = DataType:::DT_FLOAT16;
    //bool isTransY = false;

    TPosition MatmulPosition = TPosition::VECIN;
    CubeFormat MatmulFormat = CubeFormat::ND;
    DataType MatmulDtype = DataType::DT_INT32;
    //bool isTransMatmul = false;

    // TPosition biasPosition = TPosition::GM;
    // CubeFormat biasFormat = CubeFormat::ND;
    // DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = false;

    //int usedCoreNum = 1;
    //先用单核吧
    // int baseM = 64;
    // int baseN = 32;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MatmulApiTiling tilingApi(*ascendcPlatform);

    //tilingApi.SetDim(usedCoreNum); // Set the number of cores that participate in multi-core computaion is 2.
    tilingApi.SetAType(XPosition, XFormat, XDtype, isTransX);
    tilingApi.SetBType(WPosition, WFormat, WDtype, isTransW);
    tilingApi.SetCType(MatmulPosition, MatmulFormat, MatmulDtype);
    //No bias Now
    //tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);

    tilingApi.SetOrgShape(M, N, K);  
    tilingApi.SetShape(M, N, K); 
    //tiling.SetSingleShape(M, N, K);?
    tilingApi.SetBias(isBias);
    //No tiling now
    //tilingApi.SetTraverse(MatrixTraverse::FIRSTM); // Set the matmul travse is FIRSTM.
    //tilingApi.SetFixSplit(baseM, baseN, -1); // Set the fixed baseM=128, baseN=256.
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t step1MulRes = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    //No tiling now
    // tilingData.set_stepM(1); // Set the matmul tiling stepM=1.
    // tilingData.set_stepN(1); // Set the matmul tiling stepN=1.
    if (step1MulRes == -1) {
        std::cout << "gen tiling failed, this code is from tiling.cpp" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    //uint64_t localMemSize;
    //ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    //*reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
    return;
}

void GenerateTilingScamul(const char *socVersion, uint8_t *tilingBuf)
{
  //未来需要支持动态Shape
  //这里先直接全怼进去
  //Scamul:那个x_scale和w_scale相乘的操作，相当于vector mul,不过我们还是直接用matmul，后面再调,25.11.4
    constexpr int32_t M = 32;
    constexpr int32_t K = 1;
    constexpr int32_t N = 32;
    
    //int E = 4;


    TPosition xScalePosition = TPosition::VECOUT;
    CubeFormat xScaleFormat = CubeFormat::ND;
    DataType xScaleDtype = DataType::DT_FLOAT;
    bool isTransxScale = false;

    TPosition wScalePosition = TPosition::VECOUT;
    CubeFormat wScaleFormat = CubeFormat::ND;
    DataType wScaleDtype = DataType::DT_FLOAT;
    bool isTranswScale = false;

    TPosition ScamulPosition = TPosition::VECIN;
    CubeFormat ScamulFormat = CubeFormat::ND;
    DataType ScamulDtype = DataType::DT_FLOAT;
    //bool isTransScamul = false;
    // TPosition GroupListPosition = TPosition::GM;
    // CubeFormat GroupListFormat = CubeFormat::ND;
    // DataType GroupListDtype = DataType:::DT_INT64;
    // bool isTransGroupList = false;

    // TPosition YPosition = TPosition::GM;
    // CubeFormat YFormat = CubeFormat::ND;
    // DataType YDtype = DataType:::DT_FLOAT16;
    // bool isTransY = false;


    // TPosition biasPosition = TPosition::GM;
    // CubeFormat biasFormat = CubeFormat::ND;
    // DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = false;

    //int usedCoreNum = 1;
    //先用单核吧
    // int baseM = 64;
    // int baseN = 32;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MatmulApiTiling tilingApi(*ascendcPlatform);

    //tilingApi.SetDim(usedCoreNum); // Set the number of cores that participate in multi-core computaion is 2.
    tilingApi.SetAType(xScalePosition, xScaleFormat, xScaleDtype, isTransxScale);
    tilingApi.SetBType(wScalePosition, wScaleFormat, wScaleDtype, isTranswScale);
    tilingApi.SetCType(ScamulPosition, ScamulFormat, ScamulDtype);
    //tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);   
    tilingApi.SetBias(isBias);
    //tilingApi.SetTraverse(MatrixTraverse::FIRSTM); // Set the matmul travse is FIRSTM.
    //tilingApi.SetFixSplit(baseM, baseN, -1); // Set the fixed baseM=128, baseN=256.
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t step1MulRes = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    // tilingData.set_stepM(1); // Set the matmul tiling stepM=1.
    // tilingData.set_stepN(1); // Set the matmul tiling stepN=1.
    if (step1MulRes == -1) {
        std::cout << "gen tiling failed, this code is from tiling.cpp" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

    //uint64_t localMemSize;
    //ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    return;
}
