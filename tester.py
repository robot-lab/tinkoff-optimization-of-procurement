import abc
import json
from sklearn import metrics


class Tester:

    def __init__(self, file_name, border=0.5):
        """
        Initializing object of main class with testing algorithm.

        :param file_name: string
            Name of the .json file with configuration.

        :param border: float
            The accuracy boundary at which the algorithm is considered to be exact.
        """
        with open(file_name, "r") as f:
            self.__parsed_json = json.loads(f.read())

        self.__bad_metric_alarm = bool(self.__parsed_json["badMetricAlarm"])

        if self.__parsed_json["metricName"] == "Jaccard":
            self.__tester = Jaccard(border)
        elif self.__parsed_json["metricName"] == "MeanSquaredError":
            self.__tester = MeanSquaredError(border)
        elif self.__parsed_json["metricName"] == "MeanF1Score":
            self.__tester = MeanF1Score(border)
        else:
            raise ValueError("No method with given name")

    def test(self, test_sample, validation_samples):
        """
        Main testing function.

        :param test_sample: list of lists
            Predicted data.

        :param validation_samples: list of lists
            Known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """
        return self.__tester.test(test_sample, validation_samples)

    def quality_control(self, test_sample, validation_samples):
        """
        Function to get threshold estimation of the accuracy of the algorithm.

        :param test_sample: list of lists
            Predicted data.

        :param validation_samples: list of lists
            Known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """
        return self.__tester.quality_control(test_sample, validation_samples)


class Metric(abc.ABC):

    def __init__(self, border=0.5):
        """
        Initializing object of testing algorithm's class.

        :param border: float
            The accuracy boundary at which the algorithm is considered to be exact.
        """
        self.border = border

    @abc.abstractmethod
    def test(self, validation_samples, test_sample):
        """
        Main testing function.

        :param test_sample: list
            List of lists with predicted data.

        :param validation_samples: list
            List of lists with known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """
        raise NotImplementedError("Called abstract class method!")

    def quality_control(self, validation_samples, test_sample):
        """
        Function to get threshold estimation of the accuracy of the algorithm.

        :param test_sample: list
            List of lists with predicted data.

        :param validation_samples: list
            List of lists with known data.

        :return: bool
            A logic estimate of the accuracy of the algorithm.
        """
        return self.test(test_sample, validation_samples) < self.border


class Jaccard(Metric):

    @staticmethod
    def test_check(validation_samples, test_sample):
        """
        Main testing function for one list of data.

        :param test_sample: list
            List with predicted data.

        :param validation_samples: list
            List with known data.

        :return: float
            A numerical estimate of the accuracy of the algorithm.
        """

        num_dishes = len(validation_samples)
        out_min = [min(validation_samples[i], test_sample[i]) for i in range(num_dishes)]
        out_max = [max(validation_samples[i], test_sample[i]) for i in range(num_dishes)]

        numerator, denominator = 0, 0

        for k in range(num_dishes):
            numerator += out_min[k]
            denominator += out_max[k]

        return numerator / denominator

    def test(self, validation_samples, test_sample):
        num_checks = len(validation_samples)
        result = [self.test_check(validation_samples[i], test_sample[i]) for i in range(num_checks)]
        return sum(result) / num_checks


class MeanSquaredError(Metric):

    def test(self, validation_samples, test_sample):
        return metrics.mean_squared_error(validation_samples, test_sample)


class MeanF1Score(Jaccard):

    @staticmethod
    def test_check(validation_samples, test_sample):
        conj = len(set(validation_samples) and set(test_sample))
        p = conj / len(test_sample)
        r = conj / len(validation_samples)
        return 2 * p * r / (p + r)

    def quality_control(self, validation_samples, test_sample):
        return self.test(test_sample, validation_samples) > self.border


if __name__ == "__main__":
    tester = Tester("ml_config.json", 0.5)

    sv = [[0, 1, 2, 0, 1, 0, 0]]
    predict = [[0, 2, 1, 1, 1, 0, 0]]

    print(tester.test(predict, sv))
