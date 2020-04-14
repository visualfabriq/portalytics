import category_encoders as ce

potential_transformers = {
        'OneHotEncoder': ce.OneHotEncoder,
        'OrdinalEncoder': ce.OrdinalEncoder,
        'TargetEncoder': ce.TargetEncoder
}

def get_transformer(name):
    try:
        output = potential_transformers[name]
        return output()
    except KeyError:
        print('KeyError: The "%s" is not a potential transformer' % str(name))
