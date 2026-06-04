# PaddleSeg_ZL

基于 PaddlePaddle 框架的语义分割工具包，优化与改进了 PaddleSeg 原框架，使其更符合个人使用习惯。

## 📋 项目简介

PaddleSeg_ZL 是一个端到端的图像分割工具包，集成了多种先进的分割模型和工具，支持语义分割、实例分割等多种分割任务。该项目基于飞桨（PaddlePaddle）深度学习框架开发，提供了完整的训练、验证、推理和部署流程。

## 🎯 主要特性

- **丰富的分割模型**：支持 50+ 种分割模型架构，包括 DeepLabv3、PSPNet、BiSeNet、FCN、HRNet 等
- **多数据集支持**：支持 Cityscapes、VOC、ADE20k、COCO-Stuff 等多个标准数据集
- **完整的工作流**：包括数据处理、模型训练、性能验证、模型导出、推理预测等全套功能
- **ONNX 转换**：支持模型转换为 ONNX 格式，便于跨平台部署
- **灵活配置**：基于 YAML 配置文件的模块化设计，易于定制和扩展

## 📁 项目结构

```
├── paddleseg/              # 核心库文件
│   ├── models/            # 分割模型实现（50+ 种模型）
│   ├── datasets/          # 数据集加载和处理
│   ├── core/              # 训练、验证、推理核心逻辑
│   ├── transforms/        # 数据增强和预处理
│   ├── cvlibs/            # 配置管理、模型构建工具
│   ├── optimizers/        # 优化器实现
│   └── utils/             # 工具函数库
├── configs/               # 模型配置文件（按模型分类）
├── tools/                 # 工作流工具脚本
│   ├── 1_train.py        # 训练脚本
│   ├── 2_export.py       # 模型导出脚本
│   ├── 3_rename_onnx_model.py  # ONNX 模型重命名
│   ├── 4_onnx_simplify.py      # ONNX 模型简化
│   ├── 5_onnxrun.py     # ONNX 推理脚本
│   ├── predict.py        # 预测脚本
│   └── val.py           # 验证脚本
├── docs/                  # 文档
└── tests/                 # 测试文件
```

## 🚀 快速开始

### 环境要求

- Python 3.6+
- PaddlePaddle 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
pip install paddlepaddle-gpu  # 或 paddlepaddle 用于 CPU 版本
```

### 基本使用流程

#### 1. 训练模型
```bash
cd tools
python 1_train.py
```

#### 2. 验证模型
```bash
python val.py
```

#### 3. 导出模型
```bash
python 2_export.py          # 导出为 Paddle 格式
python 3_rename_onnx_model.py
python 4_onnx_simplify.py   # 简化 ONNX 模型
```

#### 4. 推理预测
```bash
python 5_onnxrun.py         # 使用 ONNX 进行推理
# 或
python predict.py           # 使用 Paddle 模型进行推理
```

## 📦 支持的模型

- **经典模型**：FCN、SegNet、UNet、PSPNet、DeepLabv3/v3+
- **轻量级模型**：BiSeNet、ENet、ESPNet、FastSCNN、PP-LiteSeg
- **高效模型**：PP-MobileSeg、MobileSeg、HardNet
- **Transformer 模型**：SETR、SegFormer、Topformer、RTFormer
- **其他高级模型**：OCRNet、GCNet、DMNet、DNLNet、EMANet 等

## 📊 支持的数据集

- Cityscapes
- PASCAL VOC
- ADE20k
- COCO-Stuff
- PP-HumanSeg14K
- 其他自定义数据集

## 🛠️ 主要工具

| 脚本 | 功能 |
|------|------|
| `1_train.py` | 模型训练 |
| `2_export.py` | 模型导出为推理格式 |
| `3_rename_onnx_model.py` | ONNX 模型重命名 |
| `4_onnx_simplify.py` | ONNX 模型简化优化 |
| `5_onnxrun.py` | ONNX 模型推理 |
| `predict.py` | Paddle 模型推理预测 |
| `val.py` | 模型验证和评估 |
| `test_seg.py` | 分割测试 |
| `analyse.py` | 模型分析 |

## 📝 配置说明

模型配置文件采用 YAML 格式，位于 `configs/` 目录。每个模型子目录对应一个模型架构，包含多个预配置的实验配置文件。

示例配置文件：
- `configs/deeplabv3/deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml`
- `configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml`

## 📄 许可证

Apache License 2.0

## 🙏 致谢

本项目基于 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 框架开发，感谢 PaddlePaddle 团队的优秀工作。
