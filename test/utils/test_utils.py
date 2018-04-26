from local_mode import file_exists, request


def files_exist(opt_ml, files):
    for f in files:
        assert file_exists(opt_ml, f), 'file {} was not created'.format(f)


def predict_and_assert_response_length(data, content_type):
    predict_response = request(data, request_type=content_type)
    assert len(predict_response) == len(data)
