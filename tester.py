import abc
import json
from sklearn import metrics


class Tester:

    def __init__(self, config_filename, border=0.5):
        """
        Initializing object of main class with testing algorithm.

        :param config_filename: string
            Name of the json file with configuration.

        :param border: float
            The accuracy boundary at which the algorithm is considered to be exact.
        """
        with open(config_filename, "r") as f:
            self.__parsed_json = json.loads(f.read())

        self.__bad_metric_alarm = bool(self.__parsed_json["bad_metric_alarm"])

        if self.__parsed_json["metric_name"] == "Jaccard":
            self.__tester = Jaccard(border)
        elif self.__parsed_json["metric_name"] == "MeanSquaredError":
            self.__tester = MeanSquaredError(border)
        elif self.__parsed_json["metric_name"] == "MeanF1Score":
            self.__tester = MeanF1Score(border)
        else:
            raise ValueError("No method with given name!")

    def test(self, validation_labels, predictions):
        """
        Main testing function.

        :param predictions: list of lists
            Predicted data.

        :param validation_labels: list of lists
            Known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """
        return self.__tester.test(validation_labels, predictions)

    def quality_control(self, validation_labels, predictions):
        """
        Function to get threshold estimation of the accuracy of the algorithm.

        :param predictions: list of lists
            Predicted data.

        :param validation_labels: list of lists
            Known data.

        :return: float
            Bool value which define quality of the algorithm.
        """
        return self.__tester.quality_control(validation_labels, predictions)


class Metric(abc.ABC):

    def __init__(self, border=0.5):
        """
        Initializing object of testing algorithm's class.

        :param border: float
            The accuracy boundary at which the algorithm is considered to be exact.
        """
        self.border = border

    @abc.abstractmethod
    def test(self, validation_labels, predictions):
        """
        Main testing function.

        :param predictions: list of lists
            Predicted data.

        :param validation_labels: list of lists
            Known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """
        raise NotImplementedError("Called abstract class method!")

    def quality_control(self, validation_labels, predictions):
        """
        Function to get threshold estimation of the accuracy of the algorithm.

        :param predictions: list of lists
            Predicted data.

        :param validation_labels: list of lists
            Known data.

        :return: bool
            Bool value which define quality of the algorithm.
        """
        return self.test(validation_labels, predictions) < self.border


class Jaccard(Metric):

    @staticmethod
    def test_check(validation_labels, predictions):
        """
        Main testing function for one list of data.

        :param predictions: list
            Predicted data.

        :param validation_labels: list
            Known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """

        num_dishes = len(validation_labels)
        out_min = [min(validation_labels[i], predictions[i]) for i in range(num_dishes)]
        out_max = [max(validation_labels[i], predictions[i]) for i in range(num_dishes)]

        numerator, denominator = 0, 0

        for k in range(num_dishes):
            numerator += out_min[k]
            denominator += out_max[k]

        return numerator / denominator

    def test(self, validation_labels, predictions):
        num_checks = len(validation_labels)
        result = [self.test_check(validation_labels[i], predictions[i]) for i in range(num_checks)]
        return sum(result) / num_checks


class MeanSquaredError(Metric):

    def test(self, validation_labels, predictions):
        return metrics.mean_squared_error(validation_labels, predictions)


class MeanF1Score(Jaccard):

    @staticmethod
    def test_check(validation_labels, predictions):
        conj = len(set(validation_labels) and set(predictions))
        p = conj / len(predictions)
        r = conj / len(validation_labels)
        return 2 * p * r / (p + r)

    def quality_control(self, validation_labels, predictions):
        return self.test(validation_labels, predictions) > self.border


def tester_testing():
    tester = Tester("ml_config.json", 0.5)

    sv = [[0, 1, 2, 0, 1, 0, 0]]
    predict = [[0, 2, 1, 1, 1, 0, 0]]

    print(tester.test(sv, predict))


if __name__ == "__main__":
    tester_testing()
