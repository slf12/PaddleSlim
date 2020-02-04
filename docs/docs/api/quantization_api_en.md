

## 量化配置
通过字典配置量化参数

```python
TENSORRT_OP_TYPES = [
    'mul', 'conv2d', 'pool2d', 'depthwise_conv2d', 'elementwise_add',
    'leaky_relu'
]
TRANSFORM_PASS_OP_TYPES = ['conv2d', 'depthwise_conv2d', 'mul']

QUANT_DEQUANT_PASS_OP_TYPES = [
        "pool2d", "elementwise_add", "concat", "softmax", "argmax", "transpose",
        "equal", "gather", "greater_equal", "greater_than", "less_equal",
        "less_than", "mean", "not_equal", "reshape", "reshape2",
        "bilinear_interp", "nearest_interp", "trilinear_interp", "slice",
        "squeeze", "elementwise_sub", "relu", "relu6", "leaky_relu", "tanh", "swish"
    ]

_quant_config_default = {
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # ops of name_scope in not_quant_pattern list, will not be quantized
    'not_quant_pattern': ['skip_quant'],
    # ops of type in quantize_op_types, will be quantized
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. defaulf is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # if True, 'quantize_op_types' will be TENSORRT_OP_TYPES
    'for_tensorrt': False,
    # if True, 'quantoze_op_types' will be TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
    'is_full_quantize': False
}
```

**参数：**

- **weight_quantize_type(str)** - 参数量化方式。可选``'abs_max'``,  ``'channel_wise_abs_max'``, ``'range_abs_max'``, ``'moving_average_abs_max'``。如果使用``TensorRT``加载量化后的模型来预测，请使用``'channel_wise_abs_max'``。 默认``'channel_wise_abs_max'``。
- **activation_quantize_type(str)** - 激活量化方式，可选``'abs_max'``, ``'range_abs_max'``, ``'moving_average_abs_max'``。如果使用``TensorRT``加载量化后的模型来预测，请使用``'range_abs_max', 'moving_average_abs_max'``。，默认``'moving_average_abs_max'``。
- **weight_bits(int)** - 参数量化bit数，默认8, 推荐设为8。
- **activation_bits(int)** -  激活量化bit数，默认8， 推荐设为8。
- **not_quant_pattern(str | list[str])** - 所有``name_scope``包含``'not_quant_pattern'``字符串的``op``，都不量化, 设置方式请参考[*fluid.name_scope*](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/name_scope_cn.html#name-scope)。
- **quantize_op_types(list[str])** -  需要进行量化的``op``类型，目前支持``'conv2d', 'depthwise_conv2d', 'mul' ``。
- **dtype(int8)** - 量化后的参数类型，默认 ``int8``, 目前仅支持``int8``。
- **window_size(int)** -  ``'range_abs_max'``量化方式的``window size``，默认10000。
- **moving_rate(int)** - ``'moving_average_abs_max'``量化方式的衰减系数，默认 0.9。
- **for_tensorrt(bool)** - 量化后的模型是否使用``TensorRT``进行预测。如果是的话，量化op类型为：``TENSORRT_OP_TYPES``。默认值为False.
- **is_full_quantize(bool)** - 是否量化所有可支持op类型。默认值为False.

!!! note "注意事项"

- 目前``Paddle-Lite``有int8 kernel来加速的op只有 ``['conv2d', 'depthwise_conv2d', 'mul']``, 其他op的int8 kernel将陆续支持。

## quant_aware
paddleslim.quant.quant_aware(program, place, config, scope=None, for_test=False)[[源代码]](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/quanter.py)
: 在``program``中加入量化和反量化``op``, 用于量化训练。


**参数：**

* **program (fluid.Program)** -  传入训练或测试``program``。
* **place(fluid.CPUPlace | fluid.CUDAPlace)** -  该参数表示``Executor``执行所在的设备。
* **config(dict)** -  量化配置表。
* **scope(fluid.Scope, optional)** -  传入用于存储``Variable``的``scope``，需要传入``program``所使用的``scope``，一般情况下，是[*fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html)。设置为``None``时将使用[*fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html)，默认值为``None``。
* **for_test(bool)** -  如果``program``参数是一个测试``program``，``for_test``应设为``True``，否则设为``False``。

**返回**

含有量化和反量化``operator``的``program``

**返回类型**

* 当``for_test=False``，返回类型为``fluid.CompiledProgram``， **注意，此返回值不能用于保存参数**。
* 当``for_test=True``，返回类型为``fluid.Program``。

!!! note "注意事项"

* 此接口会改变``program``结构，并且可能增加一些``persistable``的变量，所以加载模型参数时请注意和相应的``program``对应。
* 此接口底层经历了``fluid.Program``-> ``fluid.framework.IrGraph``->``fluid.Program``的转变，在``fluid.framework.IrGraph``中没有``Parameter``的概念，``Variable``只有``persistable``和``not persistable``的区别，所以在保存和加载参数时，请使用``fluid.io.save_persistables``和``fluid.io.load_persistables``接口。
* 由于此接口会根据``program``的结构和量化配置来对``program``添加op，所以``Paddle``中一些通过``fuse op``来加速训练的策略不能使用。已知以下策略在使用量化时必须设为``False``： ``fuse_all_reduce_ops, sync_batch_norm``。
* 如果传入的``program``中存在和任何op都没有连接的``Variable``，则会在量化的过程中被优化掉。



## convert
paddleslim.quant.convert(program, place, config, scope=None, save_int8=False)[[Source Code]](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/quanter.py)


: convert quantized and well-trained ``program`` to final  quantized ``program`` that can be used to  save ``inference model``.


**Parameters:**
- **program (fluid.Program)** - quantized and well-trained ``test program``.
- **place(fluid.CPUPlace | fluid.CUDAPlace)** - This parameter represents the executor run on which device.
- **config(dict)** - quantization config.
- **scope(fluid.Scope)** -  Scope records the mapping between variable names and variables, similar to brackets in programming languages. Usually users can use *fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html). When ``None`` will use [*fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html). Default is ``None``.
- **save_int8（bool)** - Whether to return ``program`` which model parameters' dtype is ``int8``. This parameter can only be used to get model size. Default is ``False``.


**Returns**
- **program (fluid.Program)** - freezed program which model parameters' dtype is ``float32`` but in ``int8`` range, can be used to save inference model. 
- **int8_program (fluid.Program)** - freezed program which model parameters' dtype is ``int8``, can be used to save inference model. When ``save_int8`` is set to False, will not return ``int8_program``.


!!! note "Note"
    Because this API will add or delete some ``Operators`` and ``Variables``, users must be this api after training is completed. If users want to convert middle model, they should use this api after loading parameters.

**Code Example**

```python hl_lines="27 28"
#encoding=utf8
import paddle.fluid as fluid
import paddleslim.quant as quant


train_program = fluid.Program()

with fluid.program_guard(train_program):
    image = fluid.data(name='x', shape=[None, 1, 28, 28])
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    conv = fluid.layers.conv2d(image, 32, 1)
    feat = fluid.layers.fc(conv, 10, act='softmax')
    cost = fluid.layers.cross_entropy(input=feat, label=label)
    avg_cost = fluid.layers.mean(x=cost)

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
eval_program = train_program.clone(for_test=True)
#config
config = {'weight_quantize_type': 'abs_max',
        'activation_quantize_type': 'moving_average_abs_max'}
build_strategy = fluid.BuildStrategy()
exec_strategy = fluid.ExecutionStrategy()
#use api
quant_train_program = quant.quant_aware(train_program, place, config, for_test=False)
quant_eval_program = quant.quant_aware(eval_program, place, config, for_test=True)
#disable some build strategy
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
quant_train_program = quant_train_program.with_data_parallel(
    loss_name=avg_cost.name,
    build_strategy=build_strategy,
    exec_strategy=exec_strategy)

inference_prog = quant.convert(quant_eval_program, place, config)
```

More details in <a href='https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_aware'>training-aware quantization demo</a>。

## quant_post
paddleslim.quant.quant_post(executor, model_dir, quantize_model_path,sample_generator, model_filename=None, params_filename=None, batch_size=16,batch_nums=None, scope=None, algo='KL', quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"], is_full_quantize=False, is_use_cache_file=False, cache_dir="./temp_post_training")[[Source Code]](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/quanter.py)

: Use data in ``sample_generator`` to calibrate quantized parameters for model saved in ``${model_dir}``.

**Parameters:**

- **executor (fluid.Executor)** - An Executor in Python, supports single/multiple-GPU running, and single/multiple-CPU running.
- **model_dir（str)** - Load the inference model from the directory to quantize.
- **quantize_model_path(str)** - The directory path to save the inference model that quantized.
- **sample_generator(python generator)** - Generator that generates one sample each time.
- **model_filename(str, optional)** - The name of file to load the inference program. If all parameters were saved in a single binary file, set it as file name. If parameters were saved in separate files, set it as None. Default is ``None``.
- **params_filename(str)** - The name of file to load all parameters. It is only used for the case that all parameters were saved in a single binary file. If parameters were saved in separate files, set it as ``None``. Default is ``None``.
- **batch_size(int)** - The number of samples in each mini-batch, default is 16.
- **batch_nums(int, optional)** - Iteration numbers。If ``None``，will stop training when ``sample_generator`` reaches last iteration， Otherwise ，will stop training when iteration numbers reaches ``batch_num``. Default is ``None``.
- **scope(fluid.Scope, optional)** - Scope records the mapping between variable names and variables, similar to brackets in programming languages. Usually users can use *fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html). When ``None`` will use [*fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html). Default is ``None``.
- **algo(str)** - Algorithm used in quantization, can be ``'KL'``or ``'direct'``. The parameter is for activation because weight quantization will use ``'channel_wise_abs_max'``. If algo is ``'KL'``, use KL-divergence method to get the more precise scale factor. If algo is 'direct', use ``abs_max`` method to get the scale factor. Default is ``'KL'``.
- **quantizable_op_type(list[str])** -  The list of op types that will be quantized. Default is ["conv2d","depthwise_conv2d", "mul"].
- **is_full_quantize(bool)** -  If True, apply quantization to all supported quantizable op type. If False, only apply quantization to the input ``quantizable_op_type``. Default is ``False``.
- **is_use_cache_file(bool)** - If False, all temp data will be saved in memory. If True, all temp data will be saved to disk. Defalut is ``False``.
- **cache_dir(str)** - When 'is_use_cache_file' is True, temp data will be save in 'cache_dir'. Default is './temp_post_training'.

**Returns**

None

!!! note "Note"

    1. This api will collect activations for all sample data. So if the number of samples to calibrate is too large, users should set ``is_use_cache_file`` as True.
    - Now, operators in ``Paddle-Lite`` which have int8 kernel are ``['conv2d', 'depthwise_conv2d', 'mul']``, other operators' will have int8 kernel in the future。

**Code Example**

> Note: This code can't run directly. The api will load inference model under ``${model_path}``.

```python hl_lines="9"
import paddle.fluid as fluid
import paddle.dataset.mnist as reader
from paddleslim.quant import quant_post
val_reader = reader.train()
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

exe = fluid.Executor(place)
quant_post(
        executor=exe,
        model_dir='./model_path',
        quantize_model_path='./save_path',
        sample_generator=val_reader,
        model_filename='__model__',
        params_filename='__params__',
        batch_size=16,
        batch_nums=10)
```
More details in <a href='https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post'>post training quantization demo</a>。

## quant_embedding
paddleslim.quant.quant_embedding(program, place, config, scope=None)[[Source Code]](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/quant_embedding.py)
: quantize ``Embedding`` parameters.

**Parameters:**

- **program(fluid.Program)** - infer program needed to quantize ``Embedding`` parameters.
- **scope(fluid.Scope, optional)** - Scope records the mapping between variable names and variables, similar to brackets in programming languages. Usually users can use *fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html). When ``None`` will use [*fluid.global_scope()*](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html). Default is ``None``.
- **place(fluid.CPUPlace | fluid.CUDAPlace)** - This parameter represents the executor run on which device.
- **config(dict)** - quantization config. Users can set these keys in dict：
    - ``'params_name'`` (str, required): Parameter's name needed to quantize.
    - ``'quantize_type'`` (str, optional): quantize type, supported types are ['abs_max'], ``log`` and ``product_quantization`` will support in the future. default is "abs_max".
    - ``'quantize_bits'``（int, optional): quantize bits, supported bits are [8].  default is 8.
    - ``'dtype'``(str, optional): quantize dtype, supported dtype are ['int8']. default is 'int8'.
    - ``'threshold'``(float, optional): threshold to clip tensor before quantize. When threshold is not set, tensor will not be clipped.

**Returns**

quantized program

**Return Type**

``fluid.Program``

**Code Example**
```python hl_lines="22"
import paddle.fluid as fluid
import paddleslim.quant as quant

train_program = fluid.Program()
with fluid.program_guard(train_program):
    input_word = fluid.data(name="input_word", shape=[None, 1], dtype='int64')
    input_emb = fluid.embedding(
        input=input_word,
        is_sparse=False,
        size=[100, 128],
        param_attr=fluid.ParamAttr(name='emb',
        initializer=fluid.initializer.Uniform(-0.005, 0.005)))

infer_program = train_program.clone(for_test=True)

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

config = {'params_name': 'emb', 'quantize_type': 'abs_max'}
quant_program = quant.quant_embedding(infer_program, place, config)
```

More details in  <a href='https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_embedding'>Embedding quantization demo</a>。
