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
        self.PROPORTION = 0.7

        self.df = None
        self.test_samples = None
        self.validation_samples = None
        self.train_samples_num = 0
        self.prediction = None
        self.parser = Parser()
        self.algorithm = Algorithm()
        self.tester = Tester()

    def __input(self, filepath_or_buffer):
        self.df = self.parser.parse(filepath_or_buffer, to_list=True)
        self.train_samples_num = self.PROPORTION * len(self.df)
        self.test_samples = self.df[:self.train_samples_num]
        self.validation_samples = self.df[self.train_samples_num:]

    def output(self, filename="result"):
        out = pd.DataFrame(self.prediction)
        out.to_csv(f"{filename}.csv", index=False, header=False)

    def predict(self, filepath_or_buffer):
        self.__input(filepath_or_buffer)
        self.prediction = self.algorithm.predict(self.test_samples,
                                                 self.validation_samples)

    def test(self):
        self.tester.test(self.validation_samples, self.prediction)
        self.tester.check_quality(self.validation_samples, self.prediction)


def main():
    shell = Shell()
    shell.predict("data/food.csv")
    shell.test()
    shell.output()


if __name__ == '__main__':
    main()
