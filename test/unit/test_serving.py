import json

from chainer import Variable
import numpy as np
import pytest

from sagemaker_containers import content_types, encoders

from chainer_framework import serving


@pytest.fixture(scope='module', name='np_array')
def fixture_np_array():
    return np.ones((2, 2))


def fake_predict(x):
    return x * 2


class FakeModel:
    def __init__(self):
        pass

    def __call__(self, x):
        return Variable(fake_predict(x))


def test_input_fn_json(np_array):
    json_data = json.dumps(np_array.tolist())
    deserialized_np_array = serving.default_input_fn(json_data, content_types.JSON)

    assert np.array_equal(np_array, deserialized_np_array)


def test_input_fn_npz(np_array):
    input_data = encoders.array_to_npy(np_array)
    deserialized_np_array = serving.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(np_array, deserialized_np_array)


def test_input_fn_csv(np_array):
    flattened_np_array = np.ndarray.flatten(np_array)
    csv_data = encoders.array_to_csv(np.ndarray.flatten(np_array))

    deserialized_np_array = serving.default_input_fn(csv_data, content_types.CSV)

    assert np.array_equal(flattened_np_array, deserialized_np_array)


def test_input_fn_bad_content_type():
    with pytest.raises(encoders.UnsupportedFormatError):
        serving.default_input_fn('', 'application/not_supported')


def test_predict_fn(np_array):
    predicted_data = serving.default_predict_fn(np_array, FakeModel())
    assert np.array_equal(fake_predict(np_array), predicted_data)


def test_output_fn_json(np_array):
    response = serving.default_output_fn(np_array, content_types.JSON)

    assert response.get_data(as_text=True) == encoders.array_to_json(np_array.tolist())
    assert response.content_type == content_types.JSON


def test_output_fn_csv(np_array):
    response = serving.default_output_fn(np_array, content_types.CSV)

    assert response.get_data(as_text=True) == '1.0,1.0\n1.0,1.0\n'
    assert response.content_type == content_types.CSV


def test_output_fn_npz(np_array):
    response = serving.default_output_fn(np_array, content_types.NPY)

    assert response.get_data() == encoders.array_to_npy(np_array)
    assert response.content_type == content_types.NPY


def test_input_fn_bad_accept():
    with pytest.raises(encoders.UnsupportedFormatError):
        serving.default_output_fn('', 'application/not_supported')
