import pandas as pd
from sklearn.model_selection import train_test_split

from tests.helpers import make_dataset
from vf_portalytics.multi_model import MultiTransformer



def test_multi_transformer_categorical_features():
    total_x, total_y = make_dataset()

    # make a copy of feature_0 to test if ordinals and nominals are being handled differently
    total_x['feature_0_copy'] = total_x['feature_0']

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
    ordinal_features = ['feature_1', 'feature_0_copy']

    # imitate params given from hyper optimization tuning
    params = {
        'A': {
            'transformer_nominal': 'TargetEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
        'B': {
            'transformer_nominal': 'OneHotEncoder',
            'transformer_ordinal': 'TargetEncoder'
        },
        'C': {
            'transformer_nominal': 'OneHotEncoder',
            'transformer_ordinal': 'JamesSteinEncoder'
        },
        'D': {
            'transformer_nominal': 'JamesSteinEncoder',
            'transformer_ordinal': 'OrdinalEncoder'
        },
    }

    # Initiliaze transformer
    transformer = MultiTransformer(
        cat_feature,
        clusters,
        selected_features,
        nominal_features,
        ordinal_features,
        params
    )
    transformer.fit(train_x, train_y)
    transformed_test_x = transformer.transform(test_x)

    # check shapes
    assert test_x.shape[0] == transformed_test_x.shape[0]
    assert transformed_test_x.shape[1] == 9

    # check if OneHotEncoder did what expected
    transformed_test_x = transformed_test_x.reindex(test_x.index)
    assert (transformed_test_x['feature_0_1'].isna() == test_x[cat_feature].isin(['A', 'D'])).all()

    # test if ordinals and nominals are being handled differently
    assert transformed_test_x[test_x[cat_feature] == 'B']['feature_0'].isnull().values.all()  #  OneHotEncoder used
    assert not transformed_test_x[test_x[cat_feature] == 'B']['feature_0_copy'].isnull().values.all()  #  TargetEncoder used

    # check that non categoricals are not changed
    assert transformed_test_x['feature_3'].equals(test_x['feature_3'])

def test_multi_transformer_non_categorical_features():
    cat_feature = 'cat'
    clusters = ['A']
    selected_features = {'A': ['col_1', 'col_2']}
    total_x = pd.DataFrame({'col_1': [50, 100, 200], 'col_2': [1., 2., 3.], 'cat_feature': clusters*3})

    params = {
        'A': {
            'transformer_non_categorical': 'MinMaxScaler',
            'potential_cat_feat': []
        }
    }

    # Since MultiTransformer is built for having a transformer for each group of the input, these groups need to exist
    total_x[cat_feature] = clusters[0]
    # Initiliaze transformer
    transformer = MultiTransformer(
        cat_feature,
        clusters,
        selected_features,
        nominals=[],
        ordinals=[],
        params=params
    )
    transformer.fit(total_x)
    transformed_x = transformer.transform(total_x)

    expected = pd.DataFrame({'col_1': [0., 0.333333, 1.], 'col_2': [0.0, 0.5, 1]})
    pd.testing.assert_frame_equal(transformed_x, expected)
