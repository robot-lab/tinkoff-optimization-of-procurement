# mlalgorithms library

Data science project with regression task for machine learning.

At this time our team has been developing [web-service](https://github.com/robot-lab/tinkoff-web-service) for this library.

Documentation is in [the Wiki](https://github.com/robot-lab/tinkoff-optimization-of-procurement/wiki).

## Installation

You'll need Python 3.6 or newer.

```git
git clone https://github.com/robot-lab/tinkoff-optimization-of-procurement.git
```

or you can download library directly from main page of repository.

### Example

After installing the library, you can easily run simply script, for example:

```python
from mlalgorithms import shell


sh = shell.Shell()
sh.train("data/tinkoff/train.csv")
test_result, quality = sh.test()
print(f"Metric: {test_result}")
print(f"Quality satisfaction: {quality}")
sh.predict("data/tinkoff/test.csv", "data/tinkoff/menu.csv")
sh.output()

```

You can find more examples in [the Wiki](https://github.com/robot-lab/tinkoff-optimization-of-procurement/wiki).

## Help and support

You have questions but don't want to create an issue? Questions about this repository can be sent to email `vasar007@yandex.ru`.

## Bug reports, feature requests and ideas

If you have any issues, ideas or feedback, please create [a new issue](https://github.com/robot-lab/tinkoff-optimization-of-procurement/issues/new/choose). Pull requests are also welcome!

## Contributing & style guidelines

Git commit messages use [imperative-style messages](https://stackoverflow.com/a/3580764/2867076), start with capital letter and do not have trailing commas.
