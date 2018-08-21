from mlalgorithms import shell


def test():
    sh = shell.Shell()
    sh.train("data/tinkoff/train.csv")
    test_result, quality = sh.test()
    print(f"Metrics: {test_result}")
    print(f"Quality satisfaction: {quality}")
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
