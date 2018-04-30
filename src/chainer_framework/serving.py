import json

import numpy as np
import chainer

# Try to import cupy (for GPU inference)
try:
    import cupy as cp
except ImportError:
    None

from chainer_framework.serialization import npy, csv
from container_support.app import ServingEngine
from container_support.serving import JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, NPY_CONTENT_TYPE, \
    UnsupportedContentTypeError, UnsupportedAcceptTypeError

engine = ServingEngine()


@engine.model_fn()
def model_fn(model_dir):
    """
    Loads a model
    Args:
        model_dir:

    Returns: A Chainer model.

    """
    raise NotImplementedError('Please provide a model_fn implementation. '
                              'See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk')


@engine.input_fn()
def input_fn(serialized_input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    Args:
        serialized_input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: deserialized input_data
    """

    if content_type == NPY_CONTENT_TYPE:
        return npy.loads(serialized_input_data)

    if content_type == JSON_CONTENT_TYPE:
        data = json.loads(serialized_input_data)
        return np.array(data, dtype=np.float32)

    if content_type == CSV_CONTENT_TYPE:
        return csv.loads(serialized_input_data)

    raise UnsupportedContentTypeError(content_type)


@engine.predict_fn()
def predict_fn(input_data, model):
    """A default predict_fn for Chainer. Calls a model on data deserialized in input_fn.

    Args:
        input_data: input data for prediction deserialized by input_fn
        model: model loaded in memory by model_fn

    Returns: a prediction
    """
    chainer.config.train = False
    if chainer.cuda.available:
        input_data = cp.array(input_data)
        model.to_gpu()

    predicted_data = model(input_data)
    return predicted_data.data


@engine.output_fn()
def output_fn(prediction_output, accept):
    """A default output_fn for Chainer. Serializes predictions from predict_fn.

    Args:
        prediction_output: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns
        output data serialized
    """
    prediction_output = prediction_output.tolist() if hasattr(prediction_output, 'tolist') else prediction_output

    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), JSON_CONTENT_TYPE

    if accept == NPY_CONTENT_TYPE:
        return npy.dumps(prediction_output), NPY_CONTENT_TYPE

    if accept == CSV_CONTENT_TYPE:
        return csv.dumps(prediction_output), CSV_CONTENT_TYPE

    raise UnsupportedAcceptTypeError(accept)


@engine.transform_fn()
def transform_fn(model, data, content_type, accept):
    input_data = input_fn(data, content_type)
    prediction = predict_fn(input_data, model)
    output_data, accept = output_fn(prediction, accept)
    return output_data, accept
