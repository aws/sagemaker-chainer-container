from container_support import ContainerSupport
from chainer_framework import training
from chainer_framework import serving

cs = ContainerSupport()
cs.register_engine(training.engine)
cs.register_engine(serving.engine)

if __name__ == '__main__':
    cs.run()