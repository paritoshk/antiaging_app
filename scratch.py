
@st.cache(allow_output_mutation=True)
def change_empty_lists_to_string(df,column):
    """  change empty double lists in a column to 'Nothing here' """ 
    df_copy = df.copy(deep=True)
    for i in range(len(df_copy[column])):
        if df_copy[column][i] == ['[]']:
            df_copy[column][i] = 'Nothing here'
    return df_copy

@st.cache(allow_output_mutation=True)
def random_list(list,n):
    """ function that creates a random list of n elements from a long list"""
    random.seed(42) 
    random_list = []
    for i in range(n):
        random_list.append(random.choice(list))
    return random_list

@st.cache(allow_output_mutation=True)
def combine_list_column(df,column):
    """ function that combines a column of lists in dataframe to a large list """
    df_copy = df.copy(deep=True)
    list_of_lists = df_copy[column].tolist()
    list_of_lists = [item for sublist in list_of_lists for item in sublist]
    return list(set(list_of_lists)) #drops duplicates


@st.cache(allow_output_mutation=True)
def convert_list_of_strings_to_list(list_words:list)->list:
    """ function that converts a list of strings sperated by comma into a list each strings"""
    list_output = []
    for text in list_words:
        text = text.split(',')
        for word in text:
            word = word.strip()
            list_output.append(word)
    return list_output