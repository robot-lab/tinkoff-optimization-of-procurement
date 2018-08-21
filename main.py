from mlalgorithms import shell


def test():
    sh = shell.Shell()
    sh.train("data/tinkoff/train.csv")
    sh.test()
    sh.output()


def main():
    test()

    # Example of execution:
    # sh = Shell()
    # sh.predict()
    # sh.test()
    # sh.save_model()
    # sh.output()


if __name__ == "__main__":
    main()
