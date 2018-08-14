import json
from sklearn import metrics


class Tester:
    def __init__(self, file_name: str, border=0.5, relevance=None):
        with open(file_name, 'r') as f:
            self.__parsed_json = json.loads(f.read())

        self.__bad_metric_alarm = bool(self.__parsed_json["badMetricAlarm"])

        if self.__parsed_json["metricName"] == "Jaccard":
            self.__tester = Jaccard(border, relevance)
        elif self.__parsed_json["metricName"] == "MeanSquaredError":
            self.__tester = MeanSquaredError(border, relevance)
        else:
            raise ValueError("No method with given name")

    def test(self, test_sample, validation_samples):
        return self.__tester.test(test_sample, validation_samples)

    def quality_control(self, test_sample, validation_samples):
        return self.__tester.quality_control(test_sample, validation_samples)


class Jaccard:
    def __init__(self, border=0.5, relevance=None):
        self.__border = border
        self.__relevance = relevance
        self.__num_dishes = len(self.__relevance)

    def test_check(self, validation_samples, test_sample):
        out_min = [min(validation_samples[i], test_sample[i]) for i in range(self.__num_dishes)]
        out_max = [max(validation_samples[i], test_sample[i]) for i in range(self.__num_dishes)]

        numerator, denominator = 0, 0

        for k in range(self.__num_dishes):
            numerator += self.__relevance[k] * out_min[k]
            denominator += self.__relevance[k] * out_max[k]

        return numerator / denominator

    def test(self, validation_samples, test_sample):
        num_checks = len(validation_samples)
        result = [self.test_check(validation_samples[i], test_sample[i]) for i in range(num_checks)]
        return sum(result) / num_checks

    def quality_control(self, validation_samples, test_sample):
        return self.test(test_sample, validation_samples) > self.__border


class MeanSquaredError:
    def __init__(self, border=0.5, relevance=None):
        self.__border = border
        self.__relevance = relevance

    def test(self, validation_samples, test_sample):
        rel = [self.__relevance for i in range(len(validation_samples))]
        return metrics.mean_squared_error(validation_samples, test_sample, rel)

    def quality_control(self, validation_samples, test_sample):
        return self.test(validation_samples, test_sample) > self.__border


if __name__ == '__main__':
    relevance_table = [1, 5, 4, 3, 1, 2, 1]
    tester = Tester('config.json', 0.5, relevance_table)

    sv = [[0, 1, 2, 0, 1, 0, 0]]
    predict = [[0, 2, 1, 1, 1, 0, 0]]

    print(tester.test(predict, sv))
