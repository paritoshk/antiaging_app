from collections import Counter
import seaborn as sns
import pandas as pd

def make_barplot(pdframe,column):
    pdframe = pd.DataFrame.from_dict(
        Counter(pdframe[column]), orient='index').reset_index()
    pdframe = pdframe.rename(columns = {'index': column,0:'Frequency'})
    pdframe = pdframe.dropna(axis=0)
    pdframe= pdframe.sort_values(by = ['Frequency'],ascending=False)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Frequency", y=column, data=pdframe.head(10))
    return ax
if __name__ == "__main__":
    df = pd.read_csv('data/df.csv', index_col=0)
    make_barplot(df,"author_name")