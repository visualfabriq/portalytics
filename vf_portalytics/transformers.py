import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin

POTENTIAL_TRANSFORMER = {
    'OneHotEncoder': ce.OneHotEncoder,
    'OrdinalEncoder': ce.OrdinalEncoder,
    'TargetEncoder': ce.TargetEncoder,
    'JamesSteinEncoder': ce.JamesSteinEncoder
}

def get_transformer(name):
    try:
        output = POTENTIAL_TRANSFORMER[name]
        return output()
    except KeyError:
        print('KeyError: The "%s" is not a potential transformer. TargetEncoder is being returned' % str(name))
        return ce.TargetEncoder()

class PhasingTransformer(BaseEstimator, TransformerMixin):
    """ Tansformer for multiple targets that uses only the first one to fit a transformer"""
    def __init__(self, transformer='TargetEncoder'):
        self.transformer = get_transformer(transformer)

    def fit(self, X, y=None):
        if y is None:
            self.transformer.fit(X)
        else:
            self.transformer.fit(X, y.iloc[:, 0])
        return self

    def transform(self, X, y=None):
        if y is None:
            return self.transformer.transform(X)
        else:
            return self.transformer.transform(X, y.iloc[:, 0])