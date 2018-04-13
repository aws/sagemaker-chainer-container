import numpy as np
from six import StringIO


def loads(data):
    stream = StringIO(data)
    return np.load(stream)


def dumps(data):
    buffer = StringIO()
    np.save(buffer, data)
    buffer.seek(0)
    return buffer.getvalue()

