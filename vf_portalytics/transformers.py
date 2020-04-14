import category_encoders as ce

types = {
        'OneHotEncoder': ce.OneHotEncoder,
        'OrdinalEncoder': ce.OrdinalEncoder,
        'TargetEncoder': ce.TargetEncoder
}

def potential_transformers(name):
    try:
        output = types[name]
        return output()
    except KeyError:
        print('KeyError: The "%s" is not a potential transformer' % str(name))
