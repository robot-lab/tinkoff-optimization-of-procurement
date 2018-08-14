import abc


class IAlgorithm(abc.ABC):
    @abc.abstractclassmethod
    def predict(cls, sample):
        """
        Makes predictions based on the transmitted data.
        User must override this method.

        :param sample: array-like, sparse matrix
            Samples.
        :return: array
            Returns predicted values.
        """
        raise NotImplementedError("Called abstract class method!")
