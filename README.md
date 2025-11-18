Issue:
0.应该是在初始化Copytiling时
[ERROR][Block_0][AIC_0][/usr/local/Ascend/ascend-toolkit/8.3.RC1.alpha001/aarch64-linux/ascendc/include/highlevel_api/impl/matmul/param/matmul_shape_tiling.h:100][NumericalValidCheck][34664] tiling_.GetDepthA1() is 0 , which should be larger than 0
About DepthA1:↓
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/ascendcopapi/atlasascendc_api_07_0673.html
DepthA1 = baseM*baseK

0.a:当我删除所有有关ScaMul的代码，删除注册的mm2对象及各种定义包括REGIST_MATMUL_OBJ内的参数，也就是只进行X和W的matmal进行测试的时候，系统警告baseM*baseK / 或者baseK*baseN,超大导致溢出。尽管使用GetBaseM/K/N()的输出正常。

1.（此处前方被注释的测试块可以正常输出X和W矩阵）运行到while iterate（X MUL W getTensorC）时 [ERROR][AIC_0][pid 35283] error happened! ========= SIGABRT Signal (Abort Signal from abort) catched后面很长且定位到REGIST_MATMUL_OBJ所在行