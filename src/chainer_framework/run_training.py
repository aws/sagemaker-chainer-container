import os
import logging

from chainer import serializers

from container_support.environment import TrainingEnvironment

logger = logging.getLogger(__name__)

def train():
    env = TrainingEnvironment()
    user_module = env.import_user_module()
    training_parameters = env.matching_parameters(user_module.train)
    model = user_module.train(**training_parameters)

    if model and env.current_host == 'algo-1':
        if hasattr(user_module, 'save'):
            user_module.save(model, env.model_dir)
        else:
            serializers.save_npz(os.path.join(env.model_dir, 'model.npz'), model)
    if not model and env.current_host == 'algo-1':
        logger.warn("Model object is empty. No model was saved!")


if __name__=="__main__":
    train()
