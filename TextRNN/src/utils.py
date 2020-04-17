import tensorflow as tf
import os


def listdir(path):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    files = []
    for f in os.listdir(path):
        abs_path = os.path.join(path, f)
        if os.path.isfile(abs_path):
            files.append(abs_path)
    return files


def get_shape_list(tensor):
    """Returns a list of the shape of tensor, preferring static dimensions.
    """
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
