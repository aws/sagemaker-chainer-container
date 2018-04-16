import numpy as np
from six import StringIO


def loads(data):
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    stream = StringIO(data)
    return np.genfromtxt(stream, dtype=np.float32, delimiter=',')


def dumps(data):
    stream = StringIO()
    np.savetxt(stream, data, delimiter=',', fmt='%s')
    return stream.getvalue()
