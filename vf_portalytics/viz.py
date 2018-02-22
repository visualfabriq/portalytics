import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# standard make nice big charts
sns.set(rc={'figure.figsize': (11.7, 8.27)})


def plot_histogram(df, col, top_def=20, low_limit_def=0.0, high_limit_def=100.0, debug=False):
    if df[col].dtype in [np.int64, np.float64]:
        if low_limit_def != 0.0 or high_limit_def != 100.0:
            low_limit = np.percentile(df[col].fillna(0.0), low_limit_def)
            high_limit = np.percentile(df[col].fillna(0.0), high_limit_def)
            if debug:
                print('Showing ' + col + ' between ' + str(low_limit) + ' and ' + str(high_limit))
            mask = (df[col] >= low_limit) & (df[col] <= high_limit)
            use_ser = df.loc[mask, col]
        else:
            use_ser = df[col]
        plt.figure()
        sns.distplot(use_ser)
        sns.despine()
        plt.tight_layout()
        plt.show()
    else:
        if debug:
            print('Showing ' + col + ' top ' + str(top_def))
        plt.figure()
        plt.xticks(rotation=90)
        sns.countplot(x=col, data=df, order=df[col].value_counts().iloc[:top_def].index)
        sns.despine()
        plt.tight_layout()
        plt.show()


def plot_bar(x, y, df, top_def=20, debug=False):
    plt.figure()
    plt.xticks(rotation=90)
    order_df = df.sort_values(y, ascending=False)
    sns.barplot(x=x, y=y, data=order_df[:top_def])
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_relationship(x, y, df, top_def=10, debug=False):
    if df[y].dtype not in [np.int64, np.float64]:
        raise TypeError('Column ' + y + ' is not a numerical column')

    if df[x].dtype == np.float64:
        plt.figure()
        sns.jointplot(x, y, data=df)
        sns.despine()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.xticks(rotation=90)
        sns.boxplot(x=x, y=y, data=df, order=df[x].value_counts().iloc[:top_def].index)
        sns.despine()
        plt.tight_layout()
        plt.show()


def plot_prediction(x, y, df, debug=False):
    if df[x].dtype not in [np.int64, np.float64]:
        raise TypeError('Column ' + x + ' is not a numerical column')
    if df[y].dtype not in [np.int64, np.float64]:
        raise TypeError('Column ' + y + ' is not a numerical column')
    plt.figure()
    sns.regplot(x, y, data=df)
    sns.despine()
    plt.tight_layout()
    plt.show()
