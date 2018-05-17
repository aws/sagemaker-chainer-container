# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import logging

import chainer
import numpy as np
from sagemaker_containers import encoders, env, modules, transformer, worker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_input_fn(input_data, content_type):
    return encoders.decode(input_data, content_type).astype(np.float32)


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
    model_fn = getattr(user_module, 'model_fn', transformer.default_model_fn)
    input_fn = getattr(user_module, 'input_fn', default_input_fn)
    predict_fn = getattr(user_module, 'predict_fn', default_predict_fn)
    output_fn = getattr(user_module, 'output_fn', default_output_fn)

    return transformer.Transformer(model_fn=model_fn,
                                   input_fn=input_fn,
                                   predict_fn=predict_fn,
                                   output_fn=output_fn)


def main(environ, start_response):
    serving_env = env.ServingEnv()
    user_module = modules.download_and_import(serving_env.module_dir, serving_env.module_name)

    trans = _user_module_transformer(user_module)

    trans.initialize()

    return worker.Worker(
        transform_fn=trans.transform, module_name='fake_ml_model')(environ, start_response)
