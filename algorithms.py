import abc


class IAlgorithm(abc.ABC):
    @abc.abstractclassmethod
    def predict(cls, test_sample, validation_sample):
        """
        Makes predictions based on the transmitted data.
        User must override this method.

        :param test_sample: array-like, sparse matrix
            Samples for training.

        :param validation_sample: array-like, sparse matrix
            Samples for prediction.

        :return: array
            Returns predicted values.
        """
        raise NotImplementedError("Called abstract class method!")
