from sagemaker.chainer.estimator import Chainer


class ChainerTestEstimator(Chainer):
    def __init__(self, docker_image_uri, **kwargs):
        super(ChainerTestEstimator, self).__init__(**kwargs)
        self.docker_image_uri = docker_image_uri

    def train_image(self):
        return self.docker_image_uri

    def create_model(self, model_server_workers=None):
        model = super(ChainerTestEstimator, self).create_model()
        model.image = self.docker_image_uri
        return model
