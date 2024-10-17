## 项目简介

“脑电眼部伪影分析”项目旨在通过基于 PyTorch 框架构建的简易多层感知器（MLP）模型，分析脑电信号（EEG）与眨眼行为之间的关系。
该项目探索了如何利用机器学习技术从脑电数据中识别和预测眼部伪影，进而理解眨眼对脑电信号的影响。项目的创建目的是作为**课程作业**，
旨在实践和加深对脑电信号处理及机器学习模型应用的理解。若项目中存在不足之处，欢迎指出，感谢您的关注与支持！

## 环境配置

本项目在以下环境下开发和测试：

### Python

- **版本**: 3.10
- **建议**: 使用 [Conda 环境](https://blog.csdn.net/weixin_45242930/article/details/135356097) 来管理依赖和环境，以简化包管理和环境配置。

### PyTorch

- **版本**: 2.4.1
- **安装说明**: 使用 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 进行安装，以确保兼容性和性能优化。或者参考 [CSDN 安装教程](https://blog.csdn.net/weixin_46334272/article/details/135307663)。

## 环境搭建

```bash
pip install pandas numpy matplotlib
```
## 数据来源

本次项目使用的公共数据集为 **EEG Eye State**。该数据集可通过 [UCI 机器学习库的EEG Eye State](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State) 进行访问。

数据集简介：

- **来源**: UCI 机器学习库
- **特征数量**: 14个脑电图特征
- **目标变量**: 眼睛状态（0：闭眼，1：睁眼）

获取数据集的示例代码：

<div style="background-color: #000; color: #fff; padding: 10px; border-radius: 5px;">

```python
import pandas as pd

# 数据集的URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/EEG%20Eye%20State.csv"

# 读取数据集
data = pd.read_csv(url)
```

