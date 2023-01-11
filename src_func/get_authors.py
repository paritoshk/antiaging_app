import pandas as pd
from collections import Counter
import seaborn as sns

def make_authors_column(df):
    df_copy = df.copy(deep=True)
    authors_list = []
    for i in range(len(df_copy['authors'])):
        try:
            test_string = (df_copy['authors'][i])
            df = pd.DataFrame.from_dict((eval(test_string.strip(']['))))
            df['company_name'] = df_copy['company_name'][i]
            df['pubmed_id'] = df_copy['article_id'][i]
            df['title'] = df_copy['title'][i]
            authors_list.append(df)
        except:
            test_string = (df_copy['authors'][i])
            df = pd.DataFrame.from_dict(eval(test_string.strip('][')), orient='index').T
            df['company_name'] = df_copy['company_name'][i]
            df['pubmed_id'] = df_copy['article_id'][i]
            df['title'] = df_copy['title'][i]
            authors_list.append(df)
            
    df_final_authors= pd.concat(authors_list).reset_index(drop=True)
    return df_final_authors

""" function that combiens firstname and lastname into one column called complete name"""

def combine_first_last_name(df):
    df_copy = df.copy(deep=True)
    df_copy['author_name'] = df_copy['firstname'] + ' ' + df_copy['lastname']
    return df_copy

def make_barplot(pdframe,column):
    pdframe = pd.DataFrame.from_dict(
        Counter(pdframe[column]), orient='index').reset_index()
    pdframe = pdframe.rename(columns = {'index': column,0:'Frequency'})
    pdframe = pdframe.dropna(axis=0)
    pdframe= pdframe.sort_values(by = ['Frequency'],ascending=False)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Frequency", y=column, data=pdframe.head(10))
    return ax

if __name__=="__main__":
    df = pd.read_csv('data/df_papers.csv')
    df_authors = make_authors_column(df)
    df_authors_combined = combine_first_last_name(df_authors)
    df_authors_combined.to_csv('data/df_authors_combined.csv', index=False)
    make_barplot(df_authors_combined,'author_name')