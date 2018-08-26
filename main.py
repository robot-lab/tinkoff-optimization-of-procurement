from mlalgorithms import shell


def test():
    sh = shell.Shell()
    sh.train("data/tinkoff/train.csv")
    test_result, quality = sh.test()
    print(f"Metric: {test_result}")
    print(f"Quality satisfaction: {quality}")
    sh.predict("data/tinkoff/test.csv", "data/tinkoff/menu.csv")
    sh.output()


def main():
    test()

    # Example of execution:
    # sh = shell.Shell()
    # sh.train(train_set_filename)
    # test_result, quality = sh.test()
    # sh.save_model(output_model_name)
    # sh.predict(test_set_filename, menu_filename)
    # sh.output(output_filename)


if __name__ == "__main__":
    main()
