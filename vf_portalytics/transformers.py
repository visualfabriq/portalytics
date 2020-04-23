import category_encoders as ce


def get_transformer(name):
    potential_transformers = {
        'OneHotEncoder': ce.OneHotEncoder,
        'OrdinalEncoder': ce.OrdinalEncoder,
        'TargetEncoder': ce.TargetEncoder,
        'JamesSteinEncoder': ce.JamesSteinEncoder
    }

    try:
        output = potential_transformers[name]
        return output()
    except KeyError:
        print('KeyError: The "%s" is not a potential transformer. TargetEncoder is being returned' % str(name))
        return ce.TargetEncoder()
