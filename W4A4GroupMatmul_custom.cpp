/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

 //tiling ERROR:
//  /CYT_fileSys_2/Code1/W4A4GroupMatmul/W4A4GroupMatmul.cpp:144:97: error: 'const struct TCubeTiling' has no member named 'K'
//   144 |     xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x), tilingMatmul_M * tilingMatmul_K);

//USE : REPLACE:
// tilingMatmul_M tilingMatmul_N tilingMatmul_K tilingMatmul_E 
// tilingScamul_M tilingScamul_N tilingScamul_K tilingScamul_E
//
//
constexpr int32_t  tilingMatmul_M=64; constexpr int32_t tilingMatmul_N=16; constexpr int32_t tilingMatmul_K=32; constexpr int32_t tilingMatmul_E=4;
constexpr int32_t  tilingScamul_M=64; constexpr int32_t tilingScamul_N=16; constexpr int32_t tilingScamul_K=1; constexpr int32_t tilingScamul_E=4;



/**
  * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
  * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
  * @param  tilingGM: tiling gm addr.
  * @retval None
  */
__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

class W4A4GroupMatmul {
public:
    __aicore__ inline W4A4GroupMatmul(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR x_scale, GM_ADDR w_scale, GM_ADDR group_lists, GM_ADDR y, 
      const TCubeTiling &tilingMatmul,const TCubeTiling &tilingScamul, AscendC::TPipe *pipe);
    __aicore__ inline void Process(GM_ADDR workspace);
 
    //===================================
    Matmul<MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, int32_t>> mmMatmul;

    Matmul<MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>,
               MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>> mmScamul;

    //Global内存，对应你的每一个输入输出
    AscendC::GlobalTensor<int8_t> xGlobal;
    AscendC::GlobalTensor<int8_t> wGlobal;
    AscendC::GlobalTensor<float> xScaleGlobal;
    AscendC::GlobalTensor<float> wScaleGlobal;
    AscendC::GlobalTensor<int64_t> groupListGlobal; 
    AscendC::GlobalTensor<half> yGlobal;

    //===========================TWO TILING
    TCubeTiling tilingMatmul;
    TCubeTiling tilingScamul;
    //TQue&BUFFER,对应你需要的输入输出和中间结果的队列,暂存
    //VECIN
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();

    AscendC::TQue<AscendC::TPosition::VECIN, 1> xInQueue_;     
    AscendC::TQue<AscendC::TPosition::VECIN, 1> wInQueue_;       
    AscendC::TQue<AscendC::TPosition::VECIN, 1> MatMulInQueue_;//      
    AscendC::TQue<AscendC::TPosition::VECIN, 1> GroupListsInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xScaleInQueue_;       
    AscendC::TQue<AscendC::TPosition::VECIN, 1> wScaleInQueue_;       
    AscendC::TQue<AscendC::TPosition::VECIN, 1> ScaMulInQueue_;//
    //VECCALC
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCALC_Buf_;     
    AscendC::TBuf<AscendC::TPosition::VECCALC> wCALC_Buf_;       
    AscendC::TBuf<AscendC::TPosition::VECCALC> MatMulCALC_Buf_;//
    //
    AscendC::TBuf<AscendC::TPosition::VECCALC> MatMul_FLOAT_CALC_Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_CALC_Buf_;
    //
    AscendC::TBuf<AscendC::TPosition::VECCALC> GroupListsCALC_Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xScaleCALC_Buf_;       
    AscendC::TBuf<AscendC::TPosition::VECCALC> wScaleCALC_Buf_;       
    AscendC::TBuf<AscendC::TPosition::VECCALC> ScaMulCALC_Buf_;//

    //VECOUT
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> xOutQueue_;     
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> wOutQueue_;       
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> MatMulOutQueue_;//      
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> GroupListsOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> xScaleOutQueue_;       
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> wScaleOutQueue_;       
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> ScaMulOutQueue_;//

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yOutQueue_;//

    //...
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  x: A matrix gm addr.M*K int4
  * @param  w: B matrix gm addr.E*K*N int4
  * @param  x_scale:  gm addr.M float32
  * @param  w_scale:  gm addr.E*N float32
  * @param  Group_list:  gm addr.E int64
  * @param  y:  M*N float16
  * @param  bias: Bias gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */

__aicore__ inline void W4A4GroupMatmul::Init(GM_ADDR x, GM_ADDR w, GM_ADDR x_scale, GM_ADDR w_scale, GM_ADDR group_lists, GM_ADDR y, 
      const TCubeTiling &tilingMatmul,const TCubeTiling &tilingScamul, AscendC::TPipe *pipe)
{
    this->tilingMatmul = tilingMatmul;
    this->tilingScamul = tilingScamul;
    //this->workspace = workspace[GetBlockIdx() * tiling.singleCoreM * tiling.singleCoreN];
    //WorkSpace 要不要管？

    //About tiling:
    //
    //constexpr int32_t xTotal = tilingMatmul_M * tilingMatmul_K;
    //以此类推，替换下方

    //
    xGlobal.SetGlobalBuffer((__gm__ int8_t *)x, tilingMatmul_M * tilingMatmul_K);
    wGlobal.SetGlobalBuffer((__gm__ int8_t *)w, tilingMatmul_E * tilingMatmul_K * tilingMatmul_N);
    yGlobal.SetGlobalBuffer((__gm__ half *)y,  tilingMatmul_M * tilingMatmul_N);
    groupListGlobal.SetGlobalBuffer((__gm__ int64_t *)group_lists, tilingMatmul_E);
    xScaleGlobal.SetGlobalBuffer((__gm__ float *)x_scale, tilingScamul_M);
    //warning:这里不能再用Scamul.e,因为我们全部拿进来,以后再改
    wScaleGlobal.SetGlobalBuffer((__gm__ float *)w_scale, tilingMatmul_E * tilingScamul_N);


    //biasGlobal = biasGlobal[offsetBias];

    //Init Queue内存
    //pipe->InitBuffer(step1MulreluOutQueue_, 1, tiling.baseM * tiling.baseN * sizeof(step1MulType)); // Init output buffer.
    //BaseM/BaseN
    //
    //
    //About tiling:我们后续也要计算 constexpr int32_t xPartial = ...
    //


    //此处必须计算每次部分矩阵乘最大规模==================DEPRECATED================E(max)?==

    //reference:
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> xInQueue_;       // Matmul A (int4)
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> wInQueue_;       // Matmul B (int4)
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> MatMulQueue_;       // Matmul B (int4)
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> GroupListsInQueue_;//GroupList直接用Global
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> xScaleQueue_;       // 直接用Global
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> wScaleQueue_;       // 直接用Global
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> ScaMulQueue_;


    //Option：我直接全部存进来一次性算，后面再tiling,--25.11.1, constexpr tiling = total lenth/ whatever
    pipe->InitBuffer(xInQueue_, 1, (tilingMatmul_M * tilingMatmul_K) * sizeof(int8_t)); 
    pipe->InitBuffer(wInQueue_, 1, (tilingMatmul_E * tilingMatmul_K * tilingMatmul_N) * sizeof(int8_t));
    pipe->InitBuffer(MatMulInQueue_, 1, tilingMatmul_M * tilingMatmul_N * sizeof(int32_t));
    pipe->InitBuffer(GroupListsInQueue_, 1, (tilingMatmul_E) * sizeof(int64_t));
    pipe->InitBuffer(xScaleInQueue_, 1, tilingScamul_M * sizeof(float));
    pipe->InitBuffer(wScaleInQueue_, 1, tilingMatmul_E * tilingScamul_N * sizeof(float));
    pipe->InitBuffer(ScaMulInQueue_, 1, (tilingScamul_M * tilingScamul_N) * sizeof(float));
    // //====================VECCALC
    // AscendC::TBuf<AscendC::TPosition::VECCALC> xCALC_Buf_;     
    // AscendC::TBuf<AscendC::TPosition::VECCALC> wCALC_Buf_;       
    // AscendC::TBuf<AscendC::TPosition::VECCALC> MatMulCALC_Buf_;//
    pipe->InitBuffer(xCALC_Buf_,tilingMatmul_K*tilingMatmul_M*sizeof(int8_t));
    pipe->InitBuffer(wCALC_Buf_,tilingMatmul_K*tilingMatmul_E*tilingMatmul_N*sizeof(int8_t));
    pipe->InitBuffer(MatMulCALC_Buf_,tilingMatmul_M*tilingMatmul_N*sizeof(int32_t));
    // //
    // AscendC::TBuf<AscendC::TPosition::VECCALC> MatMul_FLOAT_CALC_Buf_;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> y_CALC_Buf_;
    pipe->InitBuffer(MatMul_FLOAT_CALC_Buf_,tilingMatmul_M*tilingMatmul_N*sizeof(float));
    pipe->InitBuffer(y_CALC_Buf_,tilingMatmul_M*tilingMatmul_N*sizeof(float));
    //// AscendC::TBuf<AscendC::TPosition::VECCALC> GroupListsCALC_Buf_;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> xScaleCALC_Buf_;       
    // AscendC::TBuf<AscendC::TPosition::VECCALC> wScaleCALC_Buf_;       
    // AscendC::TBuf<AscendC::TPosition::VECCALC> ScaMulCALC_Buf_;//
    //pipe->InitBuffer(GroupListsCALC_Buf_,tilingMatmul_M*tilingMatmul_N*sizeof(float));
    pipe->InitBuffer(xScaleCALC_Buf_,tilingMatmul_M*sizeof(float));    
    pipe->InitBuffer(wScaleCALC_Buf_,tilingMatmul_E*tilingMatmul_N*sizeof(float));
    pipe->InitBuffer(ScaMulCALC_Buf_,tilingMatmul_M*tilingMatmul_N*sizeof(float));
    // //====================VECOUT
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> xOutQueue_;     
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> wOutQueue_;       
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> MatMulOutQueue_;//      
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> GroupListsOutQueue_;
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> xScaleOutQueue_;       
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> wScaleOutQueue_;       
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> ScaMulOutQueue_;//
    pipe->InitBuffer(xOutQueue_, 1, (tilingMatmul_M * tilingMatmul_K) * sizeof(int8_t)); 
    pipe->InitBuffer(wOutQueue_, 1, (tilingMatmul_E * tilingMatmul_K * tilingMatmul_N) * sizeof(int8_t));
    pipe->InitBuffer(MatMulOutQueue_, 1, (tilingMatmul_M * tilingMatmul_N) * sizeof(int32_t));
    //pipe->InitBuffer(GroupListsInQueue_, 1, (tilingMatmul_E) * sizeof(int32_t));
    pipe->InitBuffer(xScaleOutQueue_, 1, tilingScamul_M * sizeof(float));
    pipe->InitBuffer(wScaleOutQueue_, 1, tilingMatmul_E * tilingScamul_N * sizeof(float));
    pipe->InitBuffer(ScaMulOutQueue_, 1, (tilingScamul_M * tilingScamul_N) * sizeof(float));

    pipe->InitBuffer(yOutQueue_, 1, (tilingScamul_M * tilingScamul_N) * sizeof(half));
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> yOutQueue_;//

};
__aicore__ inline void W4A4GroupMatmul::CopyIn()
{
      printf("Testblock CopyIN=====================\n");
        AscendC::LocalTensor<int8_t> xLocal = xInQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> wLocal = wInQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x_scaleLocal = xScaleInQueue_.AllocTensor<float>();
        AscendC::LocalTensor<float> w_scaleLocal = wScaleInQueue_.AllocTensor<float>();
        AscendC::LocalTensor<int64_t> group_listLocal = GroupListsInQueue_.AllocTensor<int64_t>();

        //Attention:后续要注意tiling分片,则此处最后的参数需要前方再定义
        //GM->LOCAL(VECIN)
        AscendC::DataCopy(xLocal, xGlobal, tilingMatmul_M * tilingMatmul_K);
        AscendC::DataCopy(wLocal, wGlobal, tilingMatmul_E * tilingMatmul_K * tilingMatmul_N);
        AscendC::DataCopy(x_scaleLocal, xScaleGlobal, tilingScamul_M);
        AscendC::DataCopy(w_scaleLocal, wScaleGlobal, tilingMatmul_E * tilingScamul_N);
        AscendC::DataCopy(group_listLocal, groupListGlobal, tilingMatmul_E);

        //EnQue->Stage1
        xInQueue_.EnQue(xLocal);
        wInQueue_.EnQue(wLocal);
        xScaleInQueue_.EnQue(x_scaleLocal);
        wScaleInQueue_.EnQue(w_scaleLocal);
        GroupListsInQueue_.EnQue(group_listLocal);

        

        printf("Testblock CopyIN====END=================\n");
};
/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
__aicore__ inline void W4A4GroupMatmul::Process(GM_ADDR workspace)
{
  //CopyIn(GLOBAL->LOCAL,GM->VECIN)
  CopyIn();
  //LOCAL(VECIN) 
  //Deque(LOCALTENSOR)->Stage2
    AscendC::LocalTensor<int8_t> xLocal = xInQueue_.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> wLocal = wInQueue_.DeQue<int8_t>();
    AscendC::LocalTensor<float> xScaleLocal = xScaleInQueue_.DeQue<float>();
    AscendC::LocalTensor<float> wScaleLocal = wScaleInQueue_.DeQue<float>();
    AscendC::LocalTensor<int64_t> groupListLocal = GroupListsInQueue_.DeQue<int64_t>();

    AscendC::LocalTensor<half> yLocal = yOutQueue_.AllocTensor<half>();
    
    //
    //AscendC::TQue<AscendC::TPosition::VECIN, 1> MatMulInQueue_;//    
    AscendC::LocalTensor<int32_t> MatMulIn = MatMulInQueue_.AllocTensor<int32_t>();
    AscendC::LocalTensor<float> ScaMulIn = ScaMulInQueue_.AllocTensor<float>();

    AscendC::LocalTensor<int32_t> MatMul_CALC=MatMulCALC_Buf_.AllocTensor<int32_t>();
    AscendC::LocalTensor<float> ScaMul_CALC = ScaMulCALC_Buf_.AllocTensor<float>();

    //FOR CAST
    AscendC::LocalTensor<float> MatMul_FLOAT_CALC=MatMul_FLOAT_CALC_Buf_.AllocTensor<float>();
    //FOR RESULT
    AscendC::LocalTensor<float> y_CALC=y_CALC_Buf_.AllocTensor<float>();

    //======================================IMPORTANT========================
    //https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0102.html
    //GLOBAL->LOCAL / LOCAL->LOCAL /LOCAL->GLOBAL
    //GM->VECIN     VECIN->VECCALC  VECCALC->VECOUT (VECOUT(A,B)->COMPUTE->VECIN(C) ) VECOUT->GM
    //以上关系说明了，我们为什么需要进行上下方的搬运，因为不能一步到位
    //=========================================================

    //VECIN->CALC
    //注意参数以后也要修改
    AscendC::LocalTensor<int8_t> x_calc = xCALC_Buf_.AllocTensor<int8_t>();
    AscendC::LocalTensor<int8_t> w_calc = wCALC_Buf_.AllocTensor<int8_t>(); 
    AscendC::LocalTensor<float> x_scale_calc = xScaleCALC_Buf_.AllocTensor<float>();
    AscendC::LocalTensor<float> w_scale_calc = wScaleCALC_Buf_.AllocTensor<float>();

    AscendC::DataCopy(x_calc,xLocal,tilingMatmul_M*tilingMatmul_K);
    AscendC::DataCopy(w_calc,wLocal,tilingMatmul_E*tilingMatmul_K*tilingMatmul_N);
    AscendC::DataCopy(x_scale_calc,xScaleLocal,tilingScamul_M);
    AscendC::DataCopy(w_scale_calc,wScaleLocal,tilingScamul_E*tilingScamul_N);
    //CALC->VECOUT

    AscendC::LocalTensor<int8_t> x_out = xOutQueue_.AllocTensor<int8_t>();
    AscendC::LocalTensor<int8_t> w_out = wOutQueue_.AllocTensor<int8_t>(); 
    AscendC::LocalTensor<float> x_scale_out = xScaleOutQueue_.AllocTensor<float>();
    AscendC::LocalTensor<float> w_scale_out = wScaleOutQueue_.AllocTensor<float>();


    AscendC::DataCopy(x_out,x_calc,tilingMatmul_M*tilingMatmul_K);
    AscendC::DataCopy(w_out,w_calc,tilingMatmul_E*tilingMatmul_K*tilingMatmul_N);
    AscendC::DataCopy(x_scale_out,x_scale_calc,tilingScamul_M);
    AscendC::DataCopy(w_scale_out,w_scale_calc,tilingScamul_E*tilingScamul_N);


    //=============================COPY OVER============================
    
    //=============================MatMul Cal=================================
    int32_t current_E =0;
    int32_t sum_line = 0;

    current_E =0;
    sum_line = 0;
    printf("Testblock MatMul Cal Start=====================\n");
    for(int i = 0; i < tilingMatmul_E; i++)
    {
      //Step1 :计算Offset==========DEPRECATED&&ATTENTION:现在不用，后面分片+int4再用================
      //========================
      //从X中选取GroupList[E]行，矩阵乘，[第E个]W矩阵
      current_E = groupListLocal(i);
      //printf("Current E:%d \n",current_E);  
      //mmMatmul.SetOrgShape(current_E,tilingMatmul_N,tilingMatmul_K);
      mmMatmul.SetOrgShape(current_E,tilingMatmul_N,tilingMatmul_K);
      mmMatmul.SetTensorA(x_out[sum_line*tilingMatmul_K]);
      mmMatmul.SetTensorB(w_out[i*tilingMatmul_K*tilingMatmul_N]);
      

      //mmMatmul.IterateAll(MatMulIn[sum_line*tilingMatmul_N]);
      while(mmMatmul.Iterate()){ 
        mmMatmul.GetTensorC(MatMulIn[sum_line*tilingMatmul_N]);
         };
      sum_line +=current_E;

    }
    mmMatmul.End();
    //AscendC::TBuf<AscendC::TPosition::VECCALC> MatMulCALC_Buf_;// 
    //VECIN->VECCALC

    AscendC::DataCopy(MatMul_CALC,MatMulIn,tilingMatmul_M*tilingMatmul_N);
    printf("Testblock MatMul Cal OVER=====================\n");
    //=============================MatMul Cal OVER=================================     //


    //=============================Scamul Cal=================================
      //CopyScale计算所需到队列里
      //https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_10019.html
      //↑用矩阵向量乘优化这里？
      //
    current_E = 0;
    sum_line = 0;
    mmScamul.SetOrgShape(tilingScamul_M,tilingScamul_N,1);//Vector so 1
    //Test Print all
    //   for(int i=0;i<10;i++){
    //   printf("\n x_scale_out[%d]=%f ",i, x_scale_out(i));
    // }
    mmScamul.SetTensorA(x_scale_out);
    for(int i=0;i<tilingScamul_E;i++){
      current_E = groupListLocal(i);  
      mmScamul.SetTensorB(w_scale_out[i*tilingScamul_N]);
      while(mmScamul.Iterate()){mmScamul.GetTensorC(ScaMulIn);};
      AscendC::DataCopy(ScaMul_CALC[sum_line*tilingScamul_N],ScaMulIn[sum_line*tilingScamul_N],current_E*tilingScamul_N);
      sum_line+=current_E;

    }
    mmScamul.End();

    //=============================Scamul Cal OVER=================================

    //=============================Dequant&&FinalDotmul=================================
      // DotCompute();DequantCompute();
      //https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_10017.html
      //https://gitee.com/ascend/ascendc-api-adv/blob/v1.8-8.3.RC1.alpha002/examples/matrix/matmul_quant/op_kernel/matmul_quant_custom_kernel.cpp
      //CAST
      // https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0073.html
      //转换类型
      AscendC::Cast(MatMul_FLOAT_CALC,MatMul_CALC,AscendC::RoundMode::CAST_NONE,tilingMatmul_M*tilingMatmul_N);
      //https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0037.html
      //先不用Mul,可以直接乘法
      //AscendC::Mul()
      y_CALC = MatMul_FLOAT_CALC * ScaMul_CALC;

      //Y to half

      AscendC::Cast(yLocal,y_CALC,AscendC::RoundMode::CAST_NONE,tilingMatmul_M*tilingMatmul_N);
      yOutQueue_.EnQue<half>(yLocal);
     
    //=============================Dequant&&FinalDotmul END=================================
    //FREE TENSORS
    xInQueue_.FreeTensor(xLocal);
    wInQueue_.FreeTensor(wLocal);
    xScaleInQueue_.FreeTensor(xScaleLocal);
    wScaleInQueue_.FreeTensor(wScaleLocal);
    GroupListsInQueue_.FreeTensor(groupListLocal);
    //GroupListsInQueue_.FreeTensor(groupListGlobal);
    // AscendC::LocalTensor<int32_t> MatMulIn = MatMulInQueue_.AllocTensor<int32_t>();
    // AscendC::LocalTensor<float> ScaMulIn = ScaMulInQueue_.AllocTensor<float>();
    MatMulInQueue_.FreeTensor(MatMulIn);
    ScaMulInQueue_.FreeTensor(ScaMulIn);
    //
    CopyOut();

};

/**
  * @brief  Copy leakyRelu out result to GM.
  * @param  count: Iterate count(once Iterate, compute baseM * baseN).
  * @retval None
  */
__aicore__ inline void W4A4GroupMatmul::CopyOut()
{
    AscendC::LocalTensor<half> yLocal = yOutQueue_.DeQue<half>();
    AscendC::DataCopy(yGlobal, yLocal, tilingMatmul_M*tilingMatmul_N);
    yOutQueue_.FreeTensor(yLocal);

};
/**
  * @brief  Calculate the gm offset based on the blockidx.
  * @param  blockIdx: Current Core blockidx.
  * @param  tiling: Matmul tiling data.
  * @param  offsetA: Gm offset of A matrix.
  * @param  offsetB: Gm offset of B matrix.
  * @param  offsetC: Gm offset of C matrix.
  * @param  offsetBias: Gm offset of Bias matrix.
  * @retval None
  */
// template <typename aType, typename bType, typename cType, typename biasType>
// __aicore__ inline void
// W4A4GroupMatmul<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
//                                                              int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
//                                                              int32_t &offsetBias)
// {
//     auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
//     auto mCoreIndx = blockIdx % mSingleBlocks;
//     auto nCoreIndx = blockIdx / mSingleBlocks;

//     offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
//     offsetB = nCoreIndx * tiling.singleCoreN;
//     offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
//     offsetBias = nCoreIndx * tiling.singleCoreN;
// }


extern "C" __global__ __aicore__ void W4A4GroupMatmul_custom(GM_ADDR x, GM_ADDR w, GM_ADDR x_scale, GM_ADDR w_scale, GM_ADDR group_list, GM_ADDR y
, GM_ADDR workspace, GM_ADDR tilingGmMatmul, GM_ADDR tilingGmScamul)
{//op.Process(workspace, tilingGm1, tilingGm2);
    AscendC::TPipe pipe;
    TCubeTiling tilingMatmul;
    CopyTiling(&tilingMatmul, tilingGmMatmul);
    TCubeTiling tilingScamul;
    CopyTiling(&tilingScamul, tilingGmScamul);
    W4A4GroupMatmul op_kernel;
    //printf("Testblock 1 =====================\n");
    op_kernel.Init(x, w, x_scale, w_scale, group_list, y, tilingMatmul, tilingScamul , &pipe);
    // //https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0628.html
    // //↓
    //printf("Testblock 2 =====================\n");
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op_kernel.mmMatmul, &op_kernel.tilingMatmul, op_kernel.mmScamul, &op_kernel.tilingScamul); // Initialize the matmul object.
    //printf("Testblock 3 =====================\n");
    op_kernel.Process(workspace);
    //printf("Testblock 4 =====================\n");
}