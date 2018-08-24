import numpy as np

from . import model


class TestModel(model.IModel):

    def train(self, train_samples, train_labels, **kwargs):
        assert len(train_samples) == len(train_labels), \
            "Samples and labels have different sizes."

    def predict(self, samples, **kwargs):
        predictions = []
        for _, label in zip(samples, kwargs["labels"]):
            prediction = np.array(label)
            predictions.append(prediction)
        return predictions


class EatMostPopular(model.IModel):

    def __init__(self, **kwargs):
        self.data = dict()

    def train(self, train_samples, train_labels, **kwargs):
        assert len(train_samples) == len(train_labels), \
            "Samples and labels have different sizes."
        self.data = kwargs["most_popular_goods"]

    def predict(self, samples, **kwargs):
        predictions = []
        for _ in samples:
            prediction = np.array(self.data)
            predictions.append(prediction)
        return predictions


class EatSameAsBefore(model.IModel):

    def __init__(self, **kwargs):
        self.data = dict()

    def train(self, train_samples, train_labels, **kwargs):
        assert len(train_samples) == len(train_labels), \
            "Samples and labels have different sizes."
        persons_ids = [person_data[0] for person_data in train_samples]

        for persons_id, label in zip(persons_ids, train_labels):
            if self.data.get(persons_id) is None:
                self.data[persons_id] = label
            else:
                self.data[persons_id].extend(label)

    def predict(self, samples, **kwargs):
        predictions = []
        for sample in samples:
            predictions.append(self.data[sample[0]])
        return predictions
