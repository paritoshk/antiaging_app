#get papers for all companies using pymed library - and then rank order them to collect top 10 papers fro each company
import pandas as pd
import regex as re
#%pip install pymed
import numpy as np

# define helper funcitons 

def remove_newline(x):
    return x.replace('\n','')

def remove_all_characters_afternon_alphabet(x):
    return re.sub(r'[^a-zA-Z]+', ' ', x)

def find_non_empty_string(x):
    list_of_strings = []
    for i in range(len(x)):
        if x[i] != '':
            list_of_strings.append([x[i],i])
        else:
            list_of_strings.append(['None',i])
    return list_of_strings

def find_none_stringindoublelist(x):
    list_of_strings = []
    for i in range(len(x)):
        if x[i][0] =='None':
            list_of_strings.append(x[i])
    return list_of_strings


def replace_nan_with_none(x):
    """" function to replce NaN in the list with None string"""
    for i in range(len(x)):
        if x[i] != x[i]:
            x[i] = ''
    return x


def combine_two_lists_into_tuple(x,y):
    """ function to combine two lists in to a list of tuples"""

    list_of_tuples = []
    for i in range(len(x)):
        list_of_tuples.append((x[i],y[i]))
    return list_of_tuples

# read csv file

df_comapnies = pd.read_csv('aging_companies/Aging Companies - Companies.csv')

#get only running companies, rename columns

# clean the diseases column and then add the data to S3

df_comapnies_running = (df_comapnies[df_comapnies['operating status'] =='operating']).reset_index(drop=True)
df_comapnies_running['company_name'] = df_comapnies_running['[HOW TO USE THIS TABLE]\ncompany']
df_comapnies_running['diseases'] =df_comapnies_running['diseases / indications']
df_comapnies_running['company_name'] = df_comapnies_running['company_name'].apply(remove_newline)
df_comapnies_running['company_name'] = df_comapnies_running['company_name'].apply(remove_all_characters_afternon_alphabet)
comapny_list = list(set(df_comapnies_running['company_name'].tolist()))