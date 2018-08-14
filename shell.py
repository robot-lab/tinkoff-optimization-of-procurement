import logging

import pandas as pd


class Logger:
    pass


class Parser:
    pass


class Tester:
    pass


class Algorithm:
    pass


class Shell:
    def __init__(self):
        """
        Constructor which initialize class fields.
        """
        self.__PROPORTION = 0.7

        self.__df = None
        self.__test_samples = None
        self.__validation_samples = None
        self.__train_samples_num = 0
        self.__prediction = None
        self.__parser = Parser()
        self.__algorithm = Algorithm()
        self.__tester = Tester()

    def __input(self, filepath_or_buffer):
        """
        An additional method that loads data and divides it into test and
        validation samples.

        :param filepath_or_buffer: same as Parser.parse or self.predict
        """
        self.__df = self.__parser.parse(filepath_or_buffer, to_list=True)
        self.__train_samples_num = self.__PROPORTION * len(self.__df)
        self.__test_samples = self.__df[:self.__train_samples_num]
        self.__validation_samples = self.__df[self.__train_samples_num:]

    def output(self, filename="result"):
        """

        :param filename:
        """
        out = pd.DataFrame(self.__prediction)
        out.to_csv(f"{filename}.csv", index=False, header=False)

    def predict(self, filepath_or_buffer):
        """
        Make prediction for input dataset.

        :param filepath_or_buffer: str, pathlib.Path, py._path.local.LocalPath
            or any object with a read() method (such as a file handle or
            StringIO)
            The string could be a URL. Valid URL schemes include http, ftp, s3,
            and file. For file URLs, a host is expected. For instance, a local
            file could be file://localhost/path/to/table.csv.
        """
        self.__input(filepath_or_buffer)
        self.__prediction = self.__algorithm.predict(self.__test_samples,
                                                     self.__validation_samples)

    def test(self):
        """
        Test prediction quality of algorithm.
        """
        self.__tester.test(self.__validation_samples, self.__prediction)
        self.__tester.quality_control(self.__validation_samples,
                                      self.__prediction)


def main():
    shell = Shell()
    shell.predict("data/food.csv")
    shell.test()
    shell.output()


if __name__ == '__main__':
    main()
