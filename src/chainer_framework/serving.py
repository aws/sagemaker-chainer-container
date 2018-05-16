import logging

import chainer
from sagemaker_containers import encoders, env, modules, transformer, worker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_input_fn(input_data, content_type):
    return encoders.decode(input_data, content_type)


def default_predict_fn(data, model):
    """A default predict_fn for Chainer. Calls a model on data deserialized in input_fn.

    Args:
        input_data: input data for prediction deserialized by input_fn
        model: model loaded in memory by model_fn

    Returns: a prediction
    """
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        predicted_data = model(data)
        return predicted_data.data


def default_output_fn(prediction, accept):
    """A default output_fn for Chainer. Serializes predictions from predict_fn.

    Args:
        prediction_output: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns
        output data serialized
    """
    prediction = prediction.tolist() if hasattr(prediction, 'tolist') else prediction

    return transformer.default_output_fn(prediction, accept)


def _user_module_transformer(user_module):
    fns = {'model_fn': None,
           'input_fn': default_input_fn,
           'predict_fn': default_predict_fn,
           'output_fn': default_output_fn}
    for f in fns.keys():
        if hasattr(user_module, f):
            fns[f] = getattr(user_module, f)
    logger.debug('fns: {}'.format(fns))
    return transformer.Transformer(**fns)


def main(environ, start_response):
    serving_env = env.ServingEnv()
    user_module = modules.download_and_import(serving_env.module_dir, serving_env.module_name)

    trans = _user_module_transformer(user_module)

    trans.initialize()

    return worker.Worker(transform_fn=trans.transform, module_name='fake_ml_model')(environ, start_response)
