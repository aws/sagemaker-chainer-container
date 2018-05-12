import chainer

from sagemaker_containers import transformers


class ChainerTransformer(transformers.BaseTransformer):
    def predict_fn(self, input_data, model):
        """A default predict_fn for Chainer. Calls a model on data deserialized in input_fn.

        Args:
            input_data: input data for prediction deserialized by input_fn
            model: model loaded in memory by model_fn

        Returns: a prediction
        """

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            predicted_data = model(input_data)
            return predicted_data.data

    def output_fn(self, prediction_output, accept):
        """A default output_fn for Chainer. Serializes predictions from predict_fn.

        Args:
            prediction_output: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns
            output data serialized
        """
        prediction_output = prediction_output.tolist() if hasattr(prediction_output, 'tolist') else prediction_output

        return super(ChainerTransformer, self).output_fn(prediction_output, accept)
