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
from sagemaker_containers.beta.framework import (content_types, encoders, env, modules, transformer,
                                                 worker)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_input_fn(input_data, content_type):
    """Takes request data and de-serializes the data into an object for prediction.

        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:

            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.

        The input_fn is responsible to take the request data and pre-process it before prediction.

    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.

    Returns:
        (obj): data ready for prediction.
    """
    np_array = encoders.decode(input_data, content_type)
    return np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array


def default_predict_fn(data, model):
    """A default predict_fn for Chainer. Calls a model on data deserialized in input_fn.

    Args:
        data: input data for prediction deserialized by input_fn
        model: model loaded in memory by model_fn

    Returns: a prediction
    """
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        predicted_data = model(data)
        return predicted_data.data


def default_output_fn(prediction, accept):
    """Function responsible to serialize the prediction for the response.

    Args:
        prediction (obj): prediction returned by predict_fn .
        accept (str): accept content-type expected by the client.

    Returns:
        (worker.Response): a Flask response object with the following args:

            * Args:
                response: the serialized data to return
                accept: the content-type that the data was transformed to.
    """
    return worker.Response(encoders.encode(prediction, accept), accept)


def default_model_fn(model_dir):
    """Function responsible to load the model.
        For more information about model loading https://github.com/aws/sagemaker-python-sdk#model-loading.

    Args:
        model_dir (str): The directory where model files are stored.

    Returns:
        (obj) the loaded model.
    """
    return transformer.default_model_fn(model_dir)


def _user_module_transformer(user_module):
    model_fn = getattr(user_module, 'model_fn', default_model_fn)
    input_fn = getattr(user_module, 'input_fn', default_input_fn)
    predict_fn = getattr(user_module, 'predict_fn', default_predict_fn)
    output_fn = getattr(user_module, 'output_fn', default_output_fn)

    return transformer.Transformer(model_fn=model_fn, input_fn=input_fn, predict_fn=predict_fn,
                                   output_fn=output_fn)


def main(environ, start_response):
    serving_env = env.ServingEnv()
    user_module = modules.import_module_from_s3(serving_env.module_dir, serving_env.module_name)

    user_module_transformer = _user_module_transformer(user_module)

    user_module_transformer.initialize()

    app = worker.Worker(transform_fn=user_module_transformer.transform,
                        module_name=serving_env.module_name)
    return app(environ, start_response)
