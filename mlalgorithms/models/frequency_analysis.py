from . import model


class EatSameAsBefore(model.IModel):

    def __init__(self, **kwargs):
        self.data = dict()

    def train(self, train_samples, train_labels, **kwargs):
        assert(len(train_samples) == len(train_labels),
               "Samples and labels have different sizes.")
        persons_ids = [person_data[0] for person_data in train_samples]
        self.data = dict(zip(persons_ids, train_labels))
        # TODO(Vasily): process multiple persons checks with goods.

    def predict(self, samples, **kwargs):
        predictions = []
        for sample in samples:
            predictions.append(self.data[sample[0]])
        return predictions
