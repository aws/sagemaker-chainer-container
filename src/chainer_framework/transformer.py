import chainer

# Try to import cupy (for GPU inference)
try:
    import cupy as cp
except ImportError:
    None

from sagemaker_containers import transformer


def default_predict_fn(input_data, model):
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


def default_output_fn(prediction_output, accept):
    """A default output_fn for Chainer. Serializes predictions from predict_fn.

    Args:
        prediction_output: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns
        output data serialized
    """
    prediction_output = prediction_output.tolist() if hasattr(prediction_output, 'tolist') else prediction_output

    return transformer.default_output_fn(prediction_output, accept)


def create(model_fn=transformer.default_model_fn,
           input_fn=transformer.default_input_fn,
           predict_fn=default_predict_fn,
           output_fn=default_output_fn):

    return transformer.create(model_fn, input_fn, predict_fn, output_fn)
