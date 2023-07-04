from .convolution_layers import convert_conv, convert_convtranspose
from .activation_layers import convert_relu, convert_elu, convert_lrelu, convert_selu, \
    convert_sigmoid, convert_tanh, convert_softmax, convert_prelu, convert_hard_sigmoid, convert_erf
from .ltsm_layers import convert_lstm
from .operation_layers import convert_clip, convert_exp, convert_neg, convert_reduce_sum, convert_reduce_mean, \
    convert_log, convert_pow, convert_sqrt, convert_split, convert_cast, convert_floor, convert_identity, \
    convert_argmax, convert_reduce_l2, convert_reduce_max, convert_reciprocal, convert_abs, convert_not, convert_cosine, \
    convert_less, convert_less_equal, convert_and, convert_greater, convert_greater_equal, convert_xor, convert_or
from .elementwise_layers import convert_elementwise_div, convert_elementwise_add, convert_elementwise_mul, \
    convert_elementwise_sub, convert_max, convert_min, convert_mean, convert_equal, convert_where, convert_scatter_nd
from .linear_layers import convert_gemm
from .reshape_layers import convert_transpose, convert_shape, convert_gather, convert_unsqueeze, \
    convert_concat, convert_reshape, convert_flatten, convert_slice, convert_squeeze, convert_expand, convert_resize, \
    convert_tile
from .constant_layers import convert_constant, convert_constant_of_shape
from .normalization_layers import convert_batchnorm, convert_instancenorm, convert_dropout, convert_lrn
from .pooling_layers import convert_avgpool, convert_maxpool, convert_global_avg_pool
from .padding_layers import convert_padding
from .upsampling_layers import convert_upsample
from .caffe2_layers import convert_alias_with_name, convert_resize_nearest
from .sampling_layers import convert_gridsample, convert_range

AVAILABLE_CONVERTERS = {
    'Abs': convert_abs,
    'AliasWithName': convert_alias_with_name,
    'Conv': convert_conv,
    'ConvTranspose': convert_convtranspose,
    'Relu': convert_relu,
    'Resize': convert_resize,
    'Elu': convert_elu,
    'LeakyRelu': convert_lrelu,
    'Sigmoid': convert_sigmoid,
    'HardSigmoid': convert_hard_sigmoid,
    'Tanh': convert_tanh,
    'Selu': convert_selu,
    'Clip': convert_clip,
    'Exp': convert_exp,
    'Neg': convert_neg,
    'Log': convert_log,
    'Softmax': convert_softmax,
    "ScatterND": convert_scatter_nd,
    'PRelu': convert_prelu,
    'ReduceMax': convert_reduce_max,
    'ReduceSum': convert_reduce_sum,
    'ReduceMean': convert_reduce_mean,
    'Pow': convert_pow,
    'Slice': convert_slice,
    'Squeeze': convert_squeeze,
    'Expand': convert_expand,
    'Sqrt': convert_sqrt,
    'Split': convert_split,
    'Cast': convert_cast,
    'Floor': convert_floor,
    'Identity': convert_identity,
    'ArgMax': convert_argmax,
    'ReduceL2': convert_reduce_l2,
    'Max': convert_max,
    'Min': convert_min,
    'Mean': convert_mean,
    'Div': convert_elementwise_div,
    'Add': convert_elementwise_add,
    'Sum': convert_elementwise_add,
    'Mul': convert_elementwise_mul,
    'Sub': convert_elementwise_sub,
    'Gemm': convert_gemm,
    'MatMul': convert_gemm,
    'Transpose': convert_transpose,
    'Constant': convert_constant,
    'BatchNormalization': convert_batchnorm,
    'InstanceNormalization': convert_instancenorm,
    'Dropout': convert_dropout,
    'LRN': convert_lrn,
    'MaxPool': convert_maxpool,
    'AveragePool': convert_avgpool,
    'GlobalAveragePool': convert_global_avg_pool,
    'Shape': convert_shape,
    'Gather': convert_gather,
    'Unsqueeze': convert_unsqueeze,
    'Concat': convert_concat,
    'Reshape': convert_reshape,
    'ResizeNearest': convert_resize_nearest,
    'Pad': convert_padding,
    'Flatten': convert_flatten,
    'Upsample': convert_upsample,
    'Erf': convert_erf,
    'Reciprocal': convert_reciprocal,
    'ConstantOfShape': convert_constant_of_shape,
    'Equal': convert_equal,
    'Where': convert_where,
    'LSTM': convert_lstm,
    'Tile': convert_tile,
    'GridSample': convert_gridsample,
    'Range': convert_range,
    'Not': convert_not,
    'Less': convert_less,
    'LessOrEqual': convert_less_equal,
    "And": convert_and,
    "Greater": convert_greater,
    "GreaterOrEqual": convert_greater_equal,
    "Xor": convert_xor,
    "Or": convert_or,
    'Cos': convert_cosine,
}
