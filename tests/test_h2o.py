import pandas as pd
from h2o.automl import H2OAutoML
import h2o
from sklearn.model_selection import train_test_split

from tests.helpers import make_dataset
from vf_portalytics.multi_model import MultiModel

h2o.init()


class TestVfH2O:

    def test_simple_h2o(self):
        total_x, total_y = make_dataset()

        train = pd.concat([total_x, total_y], axis=1)
        train = h2o.H2OFrame(train)

        aml = H2OAutoML(max_runtime_secs=1, seed=1)
        aml.train(y=total_y.name, x=total_x.columns.tolist(), training_frame=train)
        predicted = aml.predict(train)
        assert predicted.shape == (450, 1)

        # Test recreating the trained model


    def test_multimodel_h2o(self):

        total_x, total_y = make_dataset()
        # keep only two groups
        total_x['category'] = total_x['category'].replace('D', 'B')
        total_x['category'] = total_x['category'].replace('C', 'A')

        # Declare basic parameters
        cat_feature = 'category'
        feature_col_list = total_x.columns.drop(cat_feature)
        clusters = total_x[cat_feature].unique()

        # keep all the features
        selected_features = {}
        for gp_key in clusters:
            selected_features[gp_key] = feature_col_list
        nominal_features = ['feature_0']
        ordinal_features = ['feature_1']

        # Split into train and test
        train_index, test_index = train_test_split(total_x.index, test_size=0.33, random_state=5)
        train_x, train_y = total_x.loc[train_index, :], total_y.loc[train_index]
        test_x, test_y = total_x.loc[test_index, :], total_y.loc[test_index]

        # imitate params given from hyper optimization tuning
        params = {
            'A': {
                'model_name': 'ExtraTreesRegressor',
                'max_depth': 2,
                'min_samples_leaf': 400,
                'min_samples_split': 400,
                'n_estimators': 10,
                'transformer_nominal': 'TargetEncoder',
                'transformer_ordinal': 'OrdinalEncoder'
            },
            'B': {
                'model_name': 'AutoML',
                'max_runtime_secs': 2,
                'transformer_nominal': 'TargetEncoder',
                'transformer_ordinal': 'OrdinalEncoder'
            }
        }

        # Initialize model
        model = MultiModel(group_col=cat_feature, clusters=clusters, params=params,
                           selected_features=selected_features, nominals=nominal_features, ordinals=ordinal_features)

        assert isinstance(model.sub_models['B'], H2OAutoML)
        assert model.sub_models['B'].max_runtime_secs == 2

        model.fit(train_x, train_y)
        pred_test_y = model.predict(test_x)
        assert pred_test_y.shape == (149, 1)
        assert not pred_test_y.isna().any().all()
