from vf_portalytics.model import PredictionModel


def test_prediction_model(tmpdir):
    model_dir = str(tmpdir.mkdir('model_data'))
    model_wrapper = PredictionModel('test_model', path=model_dir)
    assert model_wrapper is not None
