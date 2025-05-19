# ComfyUI_Facefusion  
a [Facefusion](https://github.com/facefusion/facefusion)'s wrapper for ComfyUI custom node.

# 功能  
实现了facefusion的主要功能,包括:  
- 人脸替换
- 人脸增强
- 唇形同步
- 帧增强


# 安装  
1. cd ComfyUI/custom_nodes  
2. git clone <https://github.com/HJH-AILab/ComfyUI_Facefusion.git>  
3. cd ComfyUI_Facefusion  
4. git clone <https://github.com/facefusion/facefusion.git>  
5. 按照 <https://docs.facefusion.io/installation>步骤安装facefusion  
    (**注意**如果你需要在正在使用的comfy环境里安装,不建议按照facefusion的步骤安装, 建议缺啥补啥就行)
6. 配置 ComfyUI/extra_model_paths.yaml  
    ```yaml  
    facefusion: <your models path>/facefusion/.assets/models/
    ```

# 说明  
1. 本程序按照facefusion3.2版本进行编写,由于facefusion经常更新,可能会出现兼容问题(经常会有参数变动,调用方式变动之类无法预测的变更),尽量安装3.2版本进行使用.
2. 测试可在python3.12+, pytorch2.7.0+cuda128环境下正常运行.

