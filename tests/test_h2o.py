import pandas as pd
from h2o.automl import H2OAutoML
import h2o

from tests.helpers import make_dataset

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