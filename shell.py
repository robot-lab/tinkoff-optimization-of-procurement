import pandas as pd
import pickle

import logger
import tester

import models.linear_model as lm
import parsers.linear_model_parser as lmp


@logger.decor_class_logging_error_and_time(
    "__init__", "__input", "output", "predict", "test"
)
class Shell:

    def __init__(self, parser_instance, algorithm_instance):
        """
        Constructor which initialize class fields.
        """
        self.__validation_labels = None
        self.__prediction = None
        self.__parser = parser_instance
        self.__algorithm = algorithm_instance
        self.__tester = tester.Tester()

    def __input(self, filepath_or_buffer, **kwargs):
        """
        An additional method that loads data and divides it into test and
        validation samples.

        :param filepath_or_buffer: same as Parser.parse or self.predict

        :param kwargs: dict
            Passes additional arguments to the parser.parse method.
        """
        self.__parser.parse(filepath_or_buffer, to_list=True, **kwargs)

    def output(self, output_filename="result"):
        """
        Output current prediction to filename.

        :param output_filename: str
            Filename to output.
        """
        out = pd.DataFrame(self.__prediction)
        out.to_csv(f"{output_filename}.csv", index=False, header=False)

    def predict(self, filepath_or_buffer, **kwargs):
        """
        Make prediction for input dataset.

        :param filepath_or_buffer: str, pathlib.Path, py._path.local.LocalPath
            or any object with a read() method (such as a file handle or
            StringIO)
            The string could be a URL. Valid URL schemes include http, ftp, s3,
            and file. For file URLs, a host is expected. For instance, a local
            file could be file://localhost/path/to/table.csv.

        :param kwargs: dict
            Passes additional arguments to the parser.parse method.
        """
        self.__input(filepath_or_buffer, **kwargs)
        self.__algorithm.train(*self.__parser.get_train_data())

        validation_samples, self.__validation_labels = \
            self.__parser.get_validation_data()

        self.__prediction = self.__algorithm.predict(validation_samples,
                                                     self.__validation_labels)

    def test(self):
        """
        Test prediction quality of algorithm.
        """
        test_result = self.__tester.test(self.__validation_labels,
                                         self.__prediction)

        quality = self.__tester.quality_control(self.__validation_labels,
                                                self.__prediction)

        print(f"Metrics: {test_result}")
        print(f"Quality satisfaction: {quality}")

    def save_model(self, filename="model"):
        """
        Save trained model with all parameters to file.

        :param filename: str
            Filename of model.
        """
        with open(f"models/{filename}.mdl", "wb") as output_stream:
            output_stream.write(pickle.dumps(self.__algorithm.model))


def test_linear():
    lin_model = lm.LinearModel()
    lin_parser = lmp.LinearModelParser()
    sh = Shell(lin_parser, lin_model)
    sh.predict("data/tinkoff/train.csv")
    sh.test()
    # sh.save_model()
    # sh.output()


def main():
    logger.setup_logging()
    test_linear()

    # Example of execution:
    # sh = Shell()
    # sh.predict()
    # sh.test()
    # sh.save_model()
    # sh.output()


if __name__ == "__main__":
    main()
