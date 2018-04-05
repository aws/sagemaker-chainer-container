import pytest
import json
import numpy as np

from chainer import Variable

from container_support.serving import JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, \
    UnsupportedContentTypeError, UnsupportedAcceptTypeError
from test.utils import csv_parser
from chainer_framework.serving import model_fn, input_fn, predict_fn, output_fn, transform_fn


@pytest.fixture()
def np_array():
    return np.ones((2, 2))


def fake_predict(x):
    return x * 2


class FakeModel:
    def __init__(self):
        pass

    def __call__(self, x):
        return Variable(fake_predict(x))


def test_model_fn():
    with pytest.raises(NotImplementedError):
        model_fn('model_dir')


def test_input_fn_json(np_array):
    json_data = json.dumps(np_array.tolist())
    deserialized_np_array = input_fn(json_data, JSON_CONTENT_TYPE)

    assert np.array_equal(np_array, deserialized_np_array)


def test_input_fn_csv(np_array):
    flattened_np_array = np.ndarray.flatten(np_array)
    csv_data = csv_parser.dumps(np.ndarray.flatten(np_array))

    deserialized_np_array = input_fn(csv_data, CSV_CONTENT_TYPE)

    assert np.array_equal(flattened_np_array, deserialized_np_array)


def test_input_fn_bad_content_type():
    with pytest.raises(UnsupportedContentTypeError):
        input_fn('', 'application/not_supported')


def test_predict_fn(np_array):
    predicted_data = predict_fn(np_array, FakeModel())
    assert np.array_equal(fake_predict(np_array), predicted_data)


def test_output_fn_json(np_array):

    output = output_fn(np_array, JSON_CONTENT_TYPE)

    assert json.dumps(np_array.tolist()) in output
    assert JSON_CONTENT_TYPE in output


def test_output_fn_csv(np_array):

    output = output_fn(np_array, CSV_CONTENT_TYPE)

    assert '1.0,1.0\n1.0,1.0\n' in output
    assert CSV_CONTENT_TYPE in output


def test_input_fn_bad_accept():
    with pytest.raises(UnsupportedAcceptTypeError):
        output_fn('', 'application/not_supported')


def test_transform_fn_json(np_array):

    transformed_data = transform_fn(FakeModel(), json.dumps(np_array.tolist()), JSON_CONTENT_TYPE, JSON_CONTENT_TYPE)

    assert '[[2.0, 2.0], [2.0, 2.0]]' in transformed_data
    assert JSON_CONTENT_TYPE in transformed_data


def test_transform_fn_csv(np_array):

    transformed_data = transform_fn(FakeModel(), csv_parser.dumps(np_array.tolist()), CSV_CONTENT_TYPE, CSV_CONTENT_TYPE)

    assert '2.0,2.0\n2.0,2.0\n' in transformed_data
    assert CSV_CONTENT_TYPE in transformed_data