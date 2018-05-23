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
from __future__ import absolute_import

import test.utils.local_mode as localmode


def files_exist(opt_ml, files):
    for f in files:
        assert localmode.file_exists(opt_ml, f), 'file {} was not created'.format(f)


def predict_and_assert_response_length(data, content_type):
    predict_response = localmode.request(data, content_type=content_type)
    assert len(predict_response) == len(data)
