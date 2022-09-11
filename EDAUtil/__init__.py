#%%
from inspect import stack
from opcode import stack_effect
import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns

def tableEDA(train, test, max_columns=50, max_rows=50):
    display(Markdown('# データ数'))
    display(Markdown('## train'))
    display(train.shape)
    display(Markdown('## test'))
    display(test.shape)

    display(Markdown('# データの中身'))
    pd.set_option('display.max_columns',max_columns)
    pd.set_option('display.max_rows',max_rows)
    display(Markdown('## train'))
    display(train.head())
    display(Markdown('## test'))
    display(test.head())

    display(Markdown('# データ型'))
    display(Markdown('## train'))
    display(train.dtypes)

    display(Markdown('# データ統計量'))
    display(Markdown('## train'))
    display(train.describe())
    display(Markdown('## test'))
    display(test.describe())

    display(Markdown('# ユニークデータ数確認'))
    display(Markdown('## train'))
    display(train.nunique())
    display(Markdown('## test'))
    display(test.nunique())

    display(Markdown('# 欠損値'))
    display(Markdown('## train'))
    display(train.isnull().sum())
    display(Markdown('## test'))
    display(test.isnull().sum())

def categoryEDA(df:pd.DataFrame, target:str, unique:str, style:str='ggplot', figsize=(10,10)):
    plt.style.use(style)
    X = df.drop([target,unique],axis=1)
    fig, axes = plt.subplots(nrows=len(X.columns), ncols=1, sharex=True, figsize=figsize)
    for i, category in enumerate(X.columns):
        df[[category, target, unique]].dropna().groupby([category, target]).count() \
            .unstack().plot.barh(ax=axes[i],stacked=True)
    plt.show()

def numericalEDA(df:pd.DataFrame, target:str, style:str='ggplot'):
    plt.style.use(style)
    X = df.drop(target,axis=1)
    for column in X.columns:
        for kind in df[target].unique():
            df[df[target]==kind][column].dropna().hist(label=str(kind), alpha=0.5)
        plt.xlabel(column)
        plt.legend()
        plt.show()

def corrEDA(df:pd.DataFrame,figsize=(10,10)):
    sns.set(rc={'figure.figsize':figsize})
    df_one_hot = pd.get_dummies(df, columns=df.select_dtypes(object).columns, drop_first=True)
    sns.heatmap(df_one_hot.dropna().corr(),vmax=1,vmin=-1,center=0,annot=True)
    plt.show()