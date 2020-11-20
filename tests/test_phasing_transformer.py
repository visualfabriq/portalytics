import pandas as pd
import random

from vf_portalytics.ml_helpers import CustomTransformer


def test_prediction_model_categorical_features():
    train_x = {
        'Var_1': [random.choice(['A', 'B', 'C', 'D']) for i in range(100)],
        'Var_2': [random.choice([22, 21, 19, 18]) for i in range(100)]
    }
    train_y = {
        'Target_2': [random.uniform(1.7, 2) for i in range(100)],
        'Target_1': [random.uniform(1.7, 2) for i in range(100)]
    }
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)

    transformer = CustomTransformer()
    transformed_x = transformer.fit_transform(train_x, train_y)

    assert all(train_x['Var_2'] == transformed_x['Var_2'])

    unique_1 = train_x['Var_1'] == train_x['Var_1'].unique()[0]
    unique_2 = transformed_x['Var_1'] == transformed_x['Var_1'].unique()[0]
    assert all(unique_1 == unique_2)