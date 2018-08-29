from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.multioutput import MultiOutputRegressor

from . import model


class RandomForestModel(model.IModel):

    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs))


class ExtraTreesModel(model.IModel):

    def __init__(self, **kwargs):
        super().__init__(ExtraTreesRegressor(**kwargs))


class GradientBoostingModel(model.IModel):

    def __init__(self, **kwargs):
        super().__init__(MultiOutputRegressor(
            GradientBoostingRegressor(**kwargs))
        )
