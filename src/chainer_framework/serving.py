import chainer

from sagemaker_containers import transformer, modules, env, worker


def default_predict_fn(input_data, model):
    """A default predict_fn for Chainer. Calls a model on data deserialized in input_fn.

    Args:
        input_data: input data for prediction deserialized by input_fn
        model: model loaded in memory by model_fn

    Returns: a prediction
    """
    with chainer.using_config('train', False), chainer.no_backprop_mode():
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


class ChainerTransformer(transformer.Transformer):
    def __init__(self, model_fn, input_fn, predict_fn=default_predict_fn, output_fn=default_output_fn):
        super().__init__(model_fn, input_fn, predict_fn, output_fn)


def main():
    serving_env = env.ServingEnv()
    user_module = modules.download_and_import(serving_env.module_dir, serving_env.module_name)

    # THIS LINE WILL FAIL
    transformer = ChainerTransformer(output_fn=user_module.output_fn, predict_fn=user_module.predict_fn)

    # I removed the initilize from the worker by mistake I will add it again.
    transformer.initialize()

    return worker.Worker(transform_fn=transformer.transform, module_name='fake_ml_model')
