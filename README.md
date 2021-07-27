# portalytics
Portable Jupyter Setup for Machine Learning.


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
