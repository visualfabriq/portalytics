# portalytics
Portable Jupyter Setup for Machine Learning.

A consistent interface for creating Machine Learning Models compatible with VisualFabriq environment.

Build models using our portalytics module.
The module is available as [pip package](https://pypi.org/project/vf-portalytics/), install simply by:
```
pip install vf-portalytics
```
Pay attention to the requirements because it is important for the model to be built with the ones that we support. 

There are [examples](https://github.com/visualfabriq/portalytics/blob/master/example_notebooks/feature_subset_example.ipynb) of how you can use portalytics. Examples for a simple model or more complex models like MultiModel.

Make sure that after saving the model using portalyctis, its possible that the model can be loaded and still contains all the important information (eg. the loaded model is able to perform a prediction?)


## [MultiModel and MultiTransformer](./vf_portalytics/multi_model.py) 
MultiModel is a custom sklearn model that contains one model for each group of training data. 
It is valuable in cases that our dataset vary a lot, but we still need to manage one model because the problem is the same.
    
* Define the groups using input parameter `clusters` which is a list of all possible groups 
  and `group_col` which is a string that indicates in which feature the groups can be found.
      
* `selected_features` give the ability of using different features for each group.

* `params` give the ability of using different model and categorical-feature transformer for each group.
    
The Jupyter notebook [multimodel_example.ipynb](example_notebooks/multimodel_example.ipynb) contains an 
end-to-end example of how MultiModel can be trained and saved using vf_portalytics Model wrapper.

MultiModel can support every sklearn based model, the only thing that is need to be done is to extend 
[`POTENTIAL_MODELS`](./vf_portalytics/ml_helpers.py) dictionary. Feel free to raise a PR. 

MultiTransformer is the transformer that is being used inside MultiModel to transform categorical features into numbers.
It is a custom sklearn transformer that contains one transformer for each group of training data.

* Can be used also separately, in the same way as MultiModel. Check [example](./tests/test_multi_model.py)

MultiTransformer can support every sklearn based transformer, the only thing that is need to be done is to extend 
[`POTENTIAL_TRANSFORMER`](./vf_portalytics/ml_helpers.py) dictionary. Feel free to raise a PR. 



## [Model](./vf_portalytics/model.py)
Model is a wrapper for ML models to make the model more portable and easier to use inside Visualfabriq environment.
```
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from vf_portalytics.model import PredictionModel

model_name = 'test_model'
prediction_model = PredictionModel(model_name, '.')

train_df = pd.DataFrame({
    'baseline_units': [800, 700],
    'promotion_technical_id': ['promotion_id_1', 'promotion_id_1'],
    'promotion_type': [1, 2],
    'promotion_ext_id': [1, 1],
    'account_id': ['pa_1', 'pa_1'],
    'pid': ['pid_1', 'pid_2']
})
dummy_regression = DummyRegressor(strategy="mean")
dummy_regression.fit(train_df, np.array([1800, 1700]))

prediction_model.features = {
    'baseline_units': [],
    'total_baseline_units': [],
    'total_nr_products': [],
    'base_price': [],
    'discount_perc': [],
    'discount_amt': [],
    'account_id': [],
}
prediction_model.model = dummy_regression

prediction_model.save()
```
The save function will generate 2 files: test_model.pkl and test_model.meta in the current directory. 
These are the files needed to load the model and make predictions inside Visualfabriq environment.

For more details check the [example](./example_notebooks/example.ipynb) of how to use the model wrapper.
