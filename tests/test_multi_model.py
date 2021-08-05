import pandas as pd

from sklearn.model_selection import train_test_split

from tests.helpers import make_dataset, make_regression_dataset
from vf_portalytics.multi_model import MultiModel


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
        'A': {
            'model_name': 'XGBRegressor',
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'B': {
            'model_name': 'XGBRegressor',
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'C': {
            'model_name': 'XGBRegressor',
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'D': {
            'model_name': 'XGBRegressor',
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
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
        'group_0': {
            'model_name': 'XGBRegressor',
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'},
    }


    # Initiliaze model
    model = MultiModel(group_col=cat_feature, clusters=clusters, params=params,
                       selected_features=selected_features, nominals=nominal_features, ordinals=ordinal_features)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)
    # check if we predicted with a model different than DummyClassifier
    assert not (pred_test_y == 0).all()[0]


def test_multi_model_with_double_target():

    n_features = 5
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

    # make the target double
    train_y = pd.DataFrame({'target_1': train_y, 'target_2': 2*train_y})
    test_y = pd.DataFrame({'target_1': test_y, 'target_2': 2*test_y})

    # keep all the features
    selected_features = {}
    for gp_key in clusters:
        selected_features[gp_key] = feature_col_list
    nominal_features = ['feature_0']
    ordinal_features = ['feature_1']

    # imitate params given from hyper optimization tuning
    params = {
        'A': {
            'model_name': 'ExtraTreesRegressor',
            'max_depth': 2,
            'min_samples_leaf': 400,
            'min_samples_split': 400,
            'n_estimators': 100,
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'B': {
            'model_name': 'ExtraTreesRegressor',
            'max_depth': 2,
            'min_samples_leaf': 400,
            'min_samples_split': 400,
            'n_estimators': 100,
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'C': {
            'model_name': 'ExtraTreesRegressor',
            'max_depth': 2,
            'min_samples_leaf': 400,
            'min_samples_split': 400,
            'n_estimators': 100,
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'D': {
            'model_name': 'XGBRegressorChain',
            'order': [0, 1],
            'max_depth': 2,
            'min_samples_leaf': 400,
            'min_samples_split': 400,
            'n_estimators': 200,
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
    }

    # Initiliaze model
    model = MultiModel(group_col=cat_feature, clusters=clusters, params=params,
                       selected_features=selected_features, nominals=nominal_features, ordinals=ordinal_features)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)

    assert pred_test_y.shape[1] == 2
    assert model.sub_models['C'].n_estimators == 100
    assert model.sub_models['D'].base_estimator.n_estimators == 200
