from sklearn.neighbors import KNeighborsRegressor

from . import model


class KNearestNeighborsModel(model.IModel):

    def __init__(self, **kwargs):
        super().__init__(KNeighborsRegressor(**kwargs))
