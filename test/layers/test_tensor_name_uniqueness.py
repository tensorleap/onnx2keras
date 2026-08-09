"""A converted graph must never give two distinct tensors the same name.

Consumers differ in how they wire a layer's call arguments back to producers.
The h5 path wires by LAYER name and tolerates duplicate tensor names;
leap_model_parser wires by TENSOR name and cannot. When two distinct tensors
share a name the parser collapses them to one key and feeds the surviving
tensor into every slot that referenced that name -- silently wrong values, or a
downstream shape error far from the real cause.

The concrete instance was GatherElements: tf.where already yields int64, so
casting its coordinate slices to int64 emitted no TF op, and each resulting
KerasTensor fell back to the generic name "Placeholder:0". At rank >= 3 there
are two or more such coordinates, so they all serialized under that one name.
reshapes/test_gather_elements_static_shape.py pins that specific op; this file
generalizes the invariant across converter families, so the next converter that
produces a nameless tensor is caught by construction rather than by a failing
model months later.

Note what the invariant is NOT: it is not "a layer's inputs are pairwise
distinct". Concat(t, t) and ordinary fan-out legitimately reference one tensor
more than once, and both appear below as guards against tightening this into a
check that rejects correct graphs.
"""
import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras


def _vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def _model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, 'g', inputs, outputs, list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def _gather_elements(data_shape, indices_shape, axis):
    data = np.random.default_rng(0).standard_normal(data_shape).astype(np.float32)
    nodes = [helper.make_node('Cast', ['indices'], ['i'], to=TensorProto.INT64),
             helper.make_node('GatherElements', ['data', 'i'], ['y'], axis=axis)]
    return _model(nodes, [_vi('indices', indices_shape)], [_vi('y', indices_shape)],
                  [numpy_helper.from_array(data, 'data')]), ['indices']


def _scatter_nd():
    # updates must be indices.shape[:-1] + data.shape[index_depth:]
    nodes = [helper.make_node('Cast', ['idx'], ['i'], to=TensorProto.INT64),
             helper.make_node('ScatterND', ['data', 'i', 'upd'], ['y'])]
    return _model(nodes,
                  [_vi('data', (1, 8, 4)), _vi('idx', (1, 3, 2)), _vi('upd', (1, 3, 4))],
                  [_vi('y', (1, 8, 4))]), ['data', 'idx', 'upd']


def _conv_relu():
    w = np.random.default_rng(1).standard_normal((4, 3, 3, 3)).astype(np.float32)
    nodes = [helper.make_node('Conv', ['x', 'w'], ['c'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             helper.make_node('Relu', ['c'], ['y'])]
    return _model(nodes, [_vi('x', (1, 3, 8, 8))], [_vi('y', (1, 4, 8, 8))],
                  [numpy_helper.from_array(w, 'w')]), ['x']


def _elementwise_chain():
    nodes = [helper.make_node('Add', ['a', 'b'], ['s']),
             helper.make_node('Mul', ['s', 'a'], ['m']),
             helper.make_node('Sub', ['m', 'b'], ['y'])]
    return _model(nodes, [_vi('a', (1, 6)), _vi('b', (1, 6))], [_vi('y', (1, 6))]), ['a', 'b']


def _split_softmax_concat():
    nodes = [helper.make_node('Softmax', ['x'], ['s'], axis=-1),
             helper.make_node('Split', ['s'], ['p', 'q'], axis=1, num_outputs=2),
             helper.make_node('Concat', ['p', 'q'], ['y'], axis=1)]
    return _model(nodes, [_vi('x', (1, 8))], [_vi('y', (1, 8))]), ['x']


def _concat_same_tensor_twice():
    # legitimate duplicate reference: one tensor into both Concat slots
    shp = numpy_helper.from_array(np.array([1, 4, 3], np.int64), 'shp')
    nodes = [helper.make_node('Reshape', ['x', 'shp'], ['r']),
             helper.make_node('Transpose', ['r'], ['t'], perm=[0, 2, 1]),
             helper.make_node('Concat', ['t', 't'], ['y'], axis=1)]
    return _model(nodes, [_vi('x', (1, 12))], [_vi('y', (1, 6, 4))], [shp]), ['x']


def _fanout_to_two_layers():
    # legitimate duplicate reference: one tensor consumed by two layers
    nodes = [helper.make_node('Relu', ['x'], ['r']),
             helper.make_node('Add', ['r', 'r'], ['a']),
             helper.make_node('Mul', ['r', 'a'], ['y'])]
    return _model(nodes, [_vi('x', (1, 5))], [_vi('y', (1, 5))]), ['x']


GRAPHS = {
    'gather_elements_rank3': lambda: _gather_elements((1, 32, 2), (1, 4, 2), 1),
    'gather_elements_rank4': lambda: _gather_elements((1, 6, 4, 2), (1, 3, 4, 2), 1),
    'scatter_nd': _scatter_nd,
    'conv_relu': _conv_relu,
    'elementwise_chain': _elementwise_chain,
    'split_softmax_concat': _split_softmax_concat,
    'concat_same_tensor_twice': _concat_same_tensor_twice,
    'fanout_to_two_layers': _fanout_to_two_layers,
}


def _layer_input_tensors(layer):
    inp = layer.input
    return inp if isinstance(inp, (list, tuple)) else [inp]


def _tensor_names_by_identity(keras_model):
    """name -> set of distinct tensor identities referenced under that name."""
    by_name = {}
    for layer in keras_model.layers:
        try:
            tensors = _layer_input_tensors(layer)
        except (AttributeError, ValueError):
            continue
        for tensor in tensors:
            try:
                name = tensor.name
            except (AttributeError, ValueError):
                continue  # eager/constant operand, carries no graph name
            by_name.setdefault(name, set()).add(id(tensor))
    return by_name


@pytest.mark.parametrize('graph_name', sorted(GRAPHS))
def test_distinct_tensors_do_not_share_a_name(graph_name):
    onnx_model, input_names = GRAPHS[graph_name]()
    keras_model = onnx_to_keras(onnx_model, input_names=input_names,
                                name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model

    by_name = _tensor_names_by_identity(keras_model)
    collisions = {name: ids for name, ids in by_name.items() if len(ids) > 1}
    assert not collisions, (
        "distinct tensors share a name in '%s': %s -- consumers that wire call "
        "args by tensor name will collapse these into one operand"
        % (graph_name, {n: len(ids) for n, ids in collisions.items()}))


@pytest.mark.parametrize('graph_name', sorted(GRAPHS))
def test_no_operand_falls_back_to_the_generic_placeholder_name(graph_name):
    # A tensor named "Placeholder:0" carries no identity of its own, so any
    # second one collides by construction. Catching the name directly flags the
    # hazard even in a graph that happens to produce only one of them today.
    onnx_model, input_names = GRAPHS[graph_name]()
    keras_model = onnx_to_keras(onnx_model, input_names=input_names,
                                name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model

    generic = sorted(n for n in _tensor_names_by_identity(keras_model)
                     if n.startswith('Placeholder'))
    assert not generic, (
        "operands in '%s' fell back to the generic placeholder name: %s -- a "
        "no-op tensor conversion produces no TF op and so no qualified name"
        % (graph_name, generic))
