import json

from sklearn.metrics import mean_squared_error, r2_score


# TODO(Oleg): need to write docs for your methods.
class Tester:

    def __init__(self, config_filename="ml_config.json", border=0.5,
                 relevance=None):
        with open(config_filename, 'r') as f:
            self.__parsed_json = json.loads(f.read())

        self.__bad_metric_alarm = bool(self.__parsed_json["bad_metric_alarm"])

        if self.__parsed_json["metric_name"] == "Jaccard":
            self.__tester = Jaccard(border, relevance)
        elif self.__parsed_json["metric_name"] == "MeanSquaredError":
            self.__tester = MeanSquaredError(border, relevance)
        else:
            raise ValueError("No method with given name!")

    def test(self, validation_labels, predictions):
        return self.__tester.test(validation_labels, predictions)

    def quality_control(self, validation_labels, predictions):
        return self.__tester.quality_control(validation_labels, predictions)


# TODO(Oleg, Timur, Vasily): can we separate some code in abstract class?
class Jaccard:

    def __init__(self, border=0.5, relevance=None):
        self.__border = border
        self.__relevance = relevance
        self.__num_dishes = len(self.__relevance)

    def test_check(self, validation_labels, predictions):
        out_min = [min(validation_labels[i],
                       predictions[i]) for i in range(self.__num_dishes)]
        out_max = [max(validation_labels[i],
                       predictions[i]) for i in range(self.__num_dishes)]

        numerator, denominator = 0, 0

        for k in range(self.__num_dishes):
            numerator += self.__relevance[k] * out_min[k]
            denominator += self.__relevance[k] * out_max[k]

        return numerator / denominator

    def test(self, validation_labels, predictions):
        num_checks = len(validation_labels)
        result = [self.test_check(validation_labels[i],
                                  predictions[i]) for i in range(num_checks)]
        return sum(result) / num_checks

    def quality_control(self, validation_labels, predictions):
        return self.test(validation_labels, predictions) < self.__border


class MeanSquaredError:

    def __init__(self, border=0.5, relevance=None):
        self.__border = border
        self.__relevance = relevance

    def test(self, validation_labels, predictions, r2=False):
        # TODO(Oleg): rel coefficient doesn't work!
        # rel = [self.__relevance for _ in range(len(validation_labels))]

        # # The mean squared error: 0 is perfect prediction.
        mse = mean_squared_error(validation_labels, predictions)

        # Explained variance score (r2_score): 1 is perfect prediction.
        if r2:
            return mse, r2_score(validation_labels, predictions)
        return mse

    def quality_control(self, validation_labels, predictions):
        return self.test(validation_labels, predictions) < self.__border


# TODO(Oleg): put this code in separate function for testing this classes.
if __name__ == "__main__":
    relevance_table = [1, 5, 4, 3, 1, 2, 1]
    tester = Tester("ml_config.json", 0.5, relevance_table)

    sv = [[0, 1, 2, 0, 1, 0, 0]]
    predict = [[0, 2, 1, 1, 1, 0, 0]]

    print(tester.test(predict, sv))
