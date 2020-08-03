import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from vf_portalytics.multi_model import MultiModel


def make_regression_dataset(n_samples, n_features, n_informative, **kwargs):
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.5,
        n_informative=n_informative,
        random_state=0
    )
    x = pd.DataFrame(x)

    x.columns = ['feature_' + str(i) for i in range(n_features)]
    x = x.assign(**kwargs)
    return x, pd.Series(y, name='target')


def make_dataset():
    # Generate data for 4 different categories
    # different #samples for each category but the same #features since they belong to the same dataset
    n_features = 5
    x1, y1 = make_regression_dataset(n_samples=100, n_features=n_features, n_informative=3, category='A')
    x2, y2 = make_regression_dataset(n_samples=150, n_features=n_features, n_informative=4, category='B')
    x3, y3 = make_regression_dataset(n_samples=80, n_features=n_features, n_informative=5, category='C')
    x4, y4 = make_regression_dataset(n_samples=120, n_features=n_features, n_informative=1, category='D')

    # combine into one dataset
    total_x = pd.concat([x1, x2, x3, x4], axis=0, ignore_index=True).reset_index(drop=True)
    total_y = pd.concat([y1, y2, y3, y4], axis=0, ignore_index=True).reset_index(drop=True)

    # make two random features categorical
    labels = ['g1', 'g2', 'g3']
    bins = [[], []]
    for i in range(2):
        bins[i] = [-np.inf,
                   total_x['feature_' + str(i)].mean() - total_x['feature_' + str(i)].std(),
                   total_x['feature_' + str(i)].mean() + total_x['feature_' + str(i)].std(),
                   total_x['feature_' + str(i)].max()]
    total_x['feature_0'] = pd.cut(total_x['feature_0'], bins=bins[0], labels=labels).astype('object')
    total_x['feature_1'] = pd.cut(total_x['feature_1'], bins=bins[1], labels=labels).astype('object')

    return total_x, total_y


def test_multi_model():
    total_x, total_y = make_dataset()

    # Declare basic parameters
    target = 'target'
    cat_feature = 'category'
    feature_col_list = total_x.columns.drop(cat_feature)
    clusters = total_x[cat_feature].unique()

    # Split into train and test
    train_index, test_index = train_test_split(total_x.index, test_size=0.33, random_state=5)
    train_x, train_y = total_x.loc[train_index, :], total_y.loc[train_index]
    test_x, test_y = total_x.loc[test_index, :], total_y.loc[test_index]

    # keep all the features
    selected_features = {}
    for gp_key in clusters:
        selected_features[gp_key] = feature_col_list
    nominal_features = ['feature_0']
    ordinal_features = ['feature_1']

    # imitate params given from hyper optimization tuning
    params = {
        'A': {'transformer_nominal': 'TargetEncoder',
              'transformer_ordinal': 'OrdinalEncoder'},
        'B': {'transformer_nominal': 'TargetEncoder',
              'transformer_ordinal': 'OrdinalEncoder'},
        'C': {'transformer_nominal': 'TargetEncoder',
              'transformer_ordinal': 'OrdinalEncoder'},
        'D': {'transformer_nominal': 'TargetEncoder',
              'transformer_ordinal': 'OrdinalEncoder'},
    }

    # Initiliaze model
    model = MultiModel(group_col=cat_feature, clusters=clusters, params=params,
                       selected_features=selected_features, nominals=nominal_features, ordinals=ordinal_features)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)

    # check if produces 1-1 results
    assert (test_y.index == pred_test_y.index).all()

    # check if we predict a sample from an other category
    test_sample_0 = pd.DataFrame(test_x.loc[0].copy()).T
    test_sample_0['category'] = 'New_Category'
    pred_test_sample_0 = model.predict(test_sample_0)
    assert (pred_test_sample_0.values == 0).all()


def test_multi_model_to_single_model():
    n_features = 5
    total_x, total_y = make_regression_dataset(n_samples=100, n_features=n_features, n_informative=3)

    # Split into train and test
    train_index, test_index = train_test_split(total_x.index, test_size=0.33, random_state=5)
    train_x, train_y = total_x.loc[train_index, :], total_y.loc[train_index]
    test_x, test_y = total_x.loc[test_index, :], total_y.loc[test_index]

    cat_feature = 'category'
    train_x[cat_feature] = 'group_0'
    clusters = train_x[cat_feature].unique()
    feature_col_list = train_x.columns.drop(cat_feature)


    print('Clusters: {}'.format(clusters))

    # keep all the features
    selected_features = {}
    for gp_key in clusters:
        selected_features[gp_key] = feature_col_list
    nominal_features = ['feature_0']
    ordinal_features = ['feature_1']

    # imitate params given from hyper optimization tuning
    params = {
        'group_0': {'transformer_nominal': 'TargetEncoder',
              'transformer_ordinal': 'OrdinalEncoder'},
    }


    # Initiliaze model
    model = MultiModel(group_col=cat_feature, clusters=clusters, params=params,
                       selected_features=selected_features, nominals=nominal_features, ordinals=ordinal_features)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)
    # check if we predicted with a model different than DummyClassifier
    assert not (pred_test_y == 0).all()
