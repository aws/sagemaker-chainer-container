import numpy as np
from six import BytesIO


def loads(data):
    stream = BytesIO(data)
    return np.load(stream)


def dumps(data):
    buffer = BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    return buffer.getvalue()

