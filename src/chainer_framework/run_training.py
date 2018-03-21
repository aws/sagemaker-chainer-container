from chainer import serializers

import os

from container_support.environment import TrainingEnvironment

def train(user_module, env):
    '''
    Runs the user's training script.
    '''
    training_parameters = env.matching_parameters(user_module.train)
    model = user_module.train(**training_parameters)

    if model:
        if hasattr(user_module, 'save'):
            user_module.save(model, env.model_dir)
        else:
            serializers.save_npz(os.path.join(env.model_dir, 'model.npz'), model)

if __name__=="__main__":
    env = TrainingEnvironment()
    user_module = env.import_user_module()
    train(user_module, env)
