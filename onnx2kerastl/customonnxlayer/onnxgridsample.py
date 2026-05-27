import numpy as np
import tensorflow as tf
from keras.layers import Layer


class OnnxGridSampleLayer(Layer):
    """Bilinear grid sampling with zero padding, equivalent to ONNX GridSample
    (mode=bilinear, padding_mode=zeros). Inputs are NCHW (channels_first).

    If sample_grid is not a tracked Keras tensor (e.g. an EagerTensor or numpy
    constant), pass it via constant_grid so the layer stores it internally and
    only img is wired as a Keras input.
    """

    def __init__(self, align_corners: int = 1, constant_grid=None, **kwargs):
        super().__init__(**kwargs)
        self.align_corners = align_corners
        if constant_grid is not None:
            if hasattr(constant_grid, 'numpy'):
                constant_grid = constant_grid.numpy()
            if not isinstance(constant_grid, np.ndarray):
                constant_grid = np.array(constant_grid)
        self.constant_grid = constant_grid

    def call(self, inputs):
        if self.constant_grid is not None:
            img = inputs
            sample_grid = tf.cast(tf.constant(self.constant_grid), tf.float32)
        else:
            img, sample_grid = inputs  # img: NCHW, sample_grid: N,H_out,W_out,2

        torch_shape = tf.shape(img)  # [N, C, H, W]
        max_xy = tf.cast(
            tf.expand_dims(
                tf.expand_dims(
                    tf.expand_dims(
                        tf.stack([torch_shape[3] - 1, torch_shape[2] - 1]),
                        0),
                    0),
                0),
            tf.float32)  # (1, 1, 1, 2): [W-1, H-1]

        if self.align_corners == 1:
            grid_index_coords = 0.5 * (sample_grid + 1.) * max_xy + 1
        else:
            grid_index_coords = 0.5 * (sample_grid + 1.) * (max_xy + 1) + 0.5

        orig_query_shape = tf.shape(grid_index_coords)
        query_points = tf.reshape(grid_index_coords, [orig_query_shape[0], -1, 2])

        # Pad image (channels_first) by 1 on each spatial side, then go NHWC
        padded_img = tf.pad(img, [[0, 0], [0, 0], [1, 1], [1, 1]])
        grid = tf.transpose(padded_img, [0, 2, 3, 1])

        grid_shape = tf.shape(grid)    # [N, H+2, W+2, C]
        query_shape = tf.shape(query_points)
        batch_size = grid_shape[0]
        height = grid_shape[1]
        width = grid_shape[2]
        channels = grid_shape[3]
        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
        batch_offsets = tf.reshape(tf.range(batch_size) * height * width, [batch_size, 1])

        alphas = []
        floors = []
        ceils = []

        for i, dim in enumerate([1, 0]):  # 'ji' indexing: first y then x
            queries = query_points[:, :, dim]
            max_floor = tf.cast(grid_shape[i + 1] - 2, query_type)
            floor = tf.minimum(
                tf.maximum(tf.constant(0.0, dtype=query_type), tf.math.floor(queries)),
                max_floor,
            )
            int_floor = tf.cast(floor, tf.int32)
            floors.append(int_floor)
            ceils.append(int_floor + 1)

            alpha = tf.minimum(
                tf.maximum(
                    tf.constant(0.0, dtype=grid_type),
                    tf.cast(queries - floor, grid_type),
                ),
                tf.constant(1.0, dtype=grid_type),
            )
            alphas.append(tf.expand_dims(alpha, 2))  # [N, Q, 1]

        def gather(y_coords, x_coords):
            linear_coordinates = batch_offsets + y_coords * width + x_coords
            gathered = tf.gather(flattened_grid, linear_coordinates)
            return tf.reshape(gathered, [batch_size, num_queries, channels])

        top_left = gather(floors[0], floors[1])
        top_right = gather(floors[0], ceils[1])
        bottom_left = gather(ceils[0], floors[1])
        bottom_right = gather(ceils[0], ceils[1])

        interp_top = alphas[1] * (top_right - top_left) + top_left
        interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
        interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        # N,Q,C -> N,H_out,W_out,C -> N,C,H_out,W_out
        reshaped = tf.reshape(interp, tf.concat([orig_query_shape[:-1], torch_shape[1:2]], axis=0))
        return tf.transpose(reshaped, [0, 3, 1, 2])

    def get_config(self):
        config = super().get_config()
        config.update({
            'align_corners': self.align_corners,
            'constant_grid': self.constant_grid.tolist() if self.constant_grid is not None else None,
        })
        return config
