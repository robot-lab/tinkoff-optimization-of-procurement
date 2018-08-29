from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from . import model


class DecisionTreeModel(model.IModel):

    def __init__(self, **kwargs):
        super().__init__(DecisionTreeRegressor(**kwargs))


class ExtraTreeModel(model.IModel):

    def __init__(self, **kwargs):
        super().__init__(ExtraTreeRegressor(**kwargs))
