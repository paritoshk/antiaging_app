import re
import pandas as pd
from sklearn.model_selection import train_test_split




### Read datasets and split 


def split_datasets(path):
    """Split the dataset into train, valid, test sets.
    Args:
        data: A list of titles, abstracts, dates, sorted by dates.
    Returns:
        Train, valid, test sets.
    """
    df= pd.read_excel(path,index_col=0)
    train_test_ratio = 0.9
    train_valid_ratio = 7/9
    df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)
    df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)
    return df_train, df_valid, df_test

def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    summaries = df['abstract'].tolist()
    for summary in summaries:
        summary = str(summary).strip()
        summary = re.sub(r"\s", " ", summary)
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        data += bos_token + ' ' + summary + ' ' + eos_token + '\n'
        
    f.write(data)
    


if __name__ == "__main__":
    path = 'data/publications/final_database_of_papers.xlsx'
    df_train, df_valid, df_test = split_datasets(path)
    build_dataset(df_train, 'train.txt')
    build_dataset(df_valid, 'valid.txt')
    build_dataset(df_test, 'test.txt')
