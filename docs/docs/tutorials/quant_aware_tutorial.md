# 图像分类模型量化训练-快速开始

该教程以图像分类模型MobileNetV1为例，说明如何快速使用PaddleSlim的[量化训练接口](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/docs/api/quantization_api.md)。 该示例包含以下步骤：

1. 导入依赖
2. 构建模型
3. 量化
4. 训练量化后的模型



## 1. 导入依赖

PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
```
## 2. 构建网络

该章节构造一个用于对MNIST数据进行分类的分类模型，选用`MobileNetV1`，并将输入大小设置为`[1, 28, 28]`，输出类别数为10。
为了方便展示示例，我们在`paddleslim.models`下预定义了用于构建分类模型的方法，执行以下代码构建分类模型：

```
exe, train_program, val_program, inputs, outputs =
    slim.models.image_classification("MobileNet", [1, 28, 28], 10, use_gpu=False)
```

>注意：paddleslim.models下的API并非PaddleSlim常规API，是为了简化示例而封装预定义的一系列方法，比如：模型结构的定义、Program的构建等。

## 3. 量化



## 4. 训练量化后的模型

### 4.1 定义输入数据

为了快速执行该示例，我们选取简单的MNIST数据，Paddle框架的`paddle.dataset.mnist`包定义了MNIST数据的下载和读取。
代码如下：

```
import paddle.dataset.mnist as reader
train_reader = paddle.batch(
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())
```

### 4.2 执行训练
以下代码执行了一个`epoch`的训练：

```
for data in train_reader():
    acc1, acc5, loss = exe.run(pruned_program, feed=train_feeder.feed(data), fetch_list=outputs)
    print(acc1, acc5, loss)
```

