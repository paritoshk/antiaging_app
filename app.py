import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import regex as re
import random
from collections import Counter
import seaborn as sns
from ast import literal_eval
st.set_page_config(layout="wide",page_title='Longevity AI', page_icon = '🤖', initial_sidebar_state = 'auto')
#author: @parikul
#display authors from paper database
# DOI Is removed from the display - because of errors in the data
#add wordcloud barplot and gpt2 sumamry of what each company is working on and in what topic model
#analysis of main Karls data

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    color:orange;
}
.medium-font {
    font-size:22px;
    color:skyblue;
}
.small-font {
    font-size:14px;
    color:grey;
}
.keyword-font {
    font-size:18px;
    color:lightgreen;
}
</style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def change_empty_lists_to_string(df,column):
    """  change empty double lists in a column to 'Nothing here' """ 
    df_copy = df.copy(deep=True)
    for i in range(len(df_copy[column])):
        if df_copy[column][i] == ['[]']:
            df_copy[column][i] = 'Nothing here'
    return df_copy

@st.cache(allow_output_mutation=True)
def read_data(data="data/publications/final_database_of_papers.xlsx"):
    """Read and process the data from local to avoid errors 
    ['company_name', 'article_id', 'title', 'keywords', 'publication_date',
       'abstract', 'journal', 'doi'] are columns that are MUST in order to run the app"""
    try:
        data = pd.read_excel(data,index_col=0)
        #for some reason keyword lists are getting converted into strings - I have to stop storing data as csv 
        data['keywords'] = data['keywords'].apply(lambda x: literal_eval(x) if "[" in x else x)
        data = change_empty_lists_to_string(data,'keywords') #sorry next data version update will fix this
        # datetime conversation for display
        data['publication_date'] = pd.to_datetime(data['publication_date'])
        data['publication_date'] = data['publication_date'].dt.date
        # to capitalize each row in the company_name column.
        data['company_name'] = data['company_name'].str.strip()
        data['company_name'] = data['company_name'].str.capitalize()
        return data
    except Exception as e:
        print(e)
        return None
    
@st.cache(allow_output_mutation=True)
def find_indexes_of_matching_keywords(list_of_keywords,df,column):
    """" function that takes in a input list of strings, a dataframe and a column containing 
    lists as input and returns indexes of rows that match elements in the input list with 
    elements in lists in the column
    match a string.
    :parameter
        :param column: string - name of column containing lists of text to match
        :param list_of_keywords: list - list of keywords to match
        :param df: dataframe - dataframe containing column
    :return
        index list"""

    indexes = []
    matched_words = []
    df_copy = df.copy(deep=True)
    df_copy[column] = df_copy[column].apply(lambda x: [item.lower() for item in x])
    list_of_keywords = [item.lower() for item in list_of_keywords]
    for i in range(len(df_copy[column])):
        for p in list_of_keywords:
            for j in range(len(df_copy[column][i])):
                    if p in df[column][i][j]:
                         indexes.append(i)
                         matched_words.append((p,df[column][i][j]))
    return indexes, matched_words


@st.cache(allow_output_mutation=True)
def display_dataframe_withindex(df,indexes):
    """Display dataframe."""
    df_display = df.iloc[indexes]
    return df_display



@st.cache(allow_output_mutation=True)
def authors_data(data="data/publications/authors_dataframe.csv"):
    """Read the data from local."""
    data = pd.read_csv(data,index_col=0)
    return data


@st.cache(allow_output_mutation=True)
def get_author_affiliation(df:pd.DataFrame,company_name=None,id_the=31740545):
    #to avoid overflow id is given as sample value do not use it as is.
    """ function that prints author_name, affilaition given a pubmed_id"""
    df_copy = df.copy(deep=True)
    if id_the:
        df_copy = df_copy[df_copy['pubmed_id']==id_the]
    if company_name:
        df_copy_list = []
        for i in company_name:
            df_copy_ele = df_copy[df_copy['company_name']==i]
            df_copy_list.append(df_copy_ele)
        df_copy = pd.concat(df_copy_list)
    df_copy = df_copy.reset_index(drop=True)
    print_list = []
    for i in range(len(df_copy['author_name'])):
        print_list.append('{0} is the author and {1} is the affiliation'.format(df_copy['author_name'][i], df_copy['affiliation'][i]))
    return df_copy[['author_name', 'affiliation']], print_list
    
        
    
        

@st.cache(allow_output_mutation=True)
def load_bert_model():
    """Instantiate a sentence-level model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache(allow_output_mutation=True)
def load_keywords(path_to_keywords="pickle_files/keyword_final.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_keywords, "rb") as h:
        keywords = pickle.load(h)
    return keywords

@st.cache(allow_output_mutation=True)
def load_embeddings(path_to_embeddings="pickle_files/embeddings.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_embeddings, "rb") as h:
        embeddings = pickle.load(h)
    return embeddings 

@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="pickle_files/fiass_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)

@st.cache(allow_output_mutation=True)
def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using #https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    and finds similar vectors using FAISS.
    
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I

@st.cache(allow_output_mutation=True)
def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx][column]) for idx in I[0]]

@st.cache(allow_output_mutation=True)
def frame_builder(data,filter_company,keyword_list):
    #must contain ['company_name', 'article_id', 'title', 'keywords', 'publication_date'] as column names
    frame = data[data['company_name'].isin(filter_company)]
    index, matched_words = find_indexes_of_matching_keywords(keyword_list,data,'keywords')
    frame = display_dataframe_withindex(data,index)
    return frame, index, matched_words

def main():
    try:
    
        # Load data and models
        data = read_data()
        model = load_bert_model()
        faiss_index = load_faiss_index()
        list_combined_keywords = load_keywords()
        authors_data_df = authors_data()
        instructions = """
        \n 1. Enter a search term in the search box. Leaving the box empty and pressing (CTRL/CMD + ENTER) will show all publications. This is a free text semantic search and will return results based on context, meaning and concept.
        \n 2. Select keywords from the dropdown. The keyword search is a soft match. Free text has precedence over keywords.
        \n 3. Select a company or companies from the dropdown. This will filter the results to strictly show results from the selected company.
        \n 4. Select the number of results you want to see. The default is 5."""
        newline= '\n'
        # important columns - company_name, article_id, title, keywords, publication_date, abstract, journal, doi, authors
        # variables - user_input, filter_company, num_results
        comapny_list = list(set(data['company_name'].to_list()))
        
        # duplicate keywords are found - use set to remove duplicates - like blood,
        #"""This application attempts to automate searching thousands of abstracts focused on 97 selected for-profit published by companies focusing on anti-aging and longevity. These are funded over $10B."""
        st.title("🧬 Longevity-AI 🧪")
        st.header("👨‍🔬: Hi! I am  your due diligence associate.")
        st.write("🤖: Ask me about the :blue[anti-aging] industry. Please try to be specific and elaborate. I am still learning.")
        st.write(" 💊 This does a semantic search over abstracts within :blue[anti-aging] & longevity industry and research*")
        user_input = st.text_area("Type below. I will try to be accurate. Pardon me if I reply nothing.", "Tell me about research in methylation using stem cells in mouse models")
        st.caption('Try these - 1) tell me how red meat affects cancer 2) show me research about how lung fibrosis occurs. etc')
        
        # Sidebar 
        # Filters
        st.sidebar.markdown("**Filters**")
        # User search
        
        # Keyword search
        keyword_list = st.sidebar.multiselect('Select Keywords (**beta feature, choose multiple)',list_combined_keywords, ['x-ray crystallography','gfat2','haploid mouse embryonic stem cells']) #get dropdown of keywords     
        #filter by keywords, company and seed terms (stem cell, aging, etc) within th abstract and title
        #display the number of results, authors, companies, journals, keywords
        filter_company = st.sidebar.multiselect('Select a Company or Companies',comapny_list, "Altos labs") #get dropdown of companies
        num_results = st.sidebar.slider("Number of search results", 5,20, 5)
        # Instructions
        
        st.sidebar.markdown("**Instructions**")
        st.sidebar.markdown('<p class="small-font">{0}</p>'.format(instructions), unsafe_allow_html=True)
        st.sidebar.markdown("*If no results appear - try broadening your criteria, keywords or deslecting your filters*")
        st.sidebar.write("**Stats**")
        st.sidebar.write("Total number of publications: ", len(data))
        st.sidebar.write("Total number of companies: ", len(comapny_list))
        st.sidebar.write("Total number of keywords: ", len(list_combined_keywords))
        st.sidebar.write("Total number of journals: ", len(set(data['journal'].to_list())))
        
        
        
        # Fetch results
        if user_input or keyword_list:
            # Get paper IDs
            D, I = vector_search([user_input], model, faiss_index, num_results)
            # Slice data on comapny name
            try:
                if filter_company:
                    frame = data[data['company_name'].isin(filter_company)]
                    author_frame_comps= get_author_affiliation(authors_data_df,filter_company)
                    
                    if keyword_list:
                        index, matched_words = find_indexes_of_matching_keywords(keyword_list,frame,'keywords')
                        frame = display_dataframe_withindex(frame,index)
                    else:
                        pass
                else:
                    frame = data

            except:
                pass #see if this works if you have multiple companies
            # Get individual results
            # summary barplots
            st.header("Summary plots, followed by results - top 10")
            company_namedf = frame['company_name'].value_counts().rename_axis('unique_values')
            journal_df = frame['journal'].value_counts().rename_axis('unique_values')
            st.subheader("1. Number of papers per company")
            st.bar_chart(company_namedf.T.head(10))
            st.subheader("2. Number of papers per jounral")
            st.bar_chart(journal_df.T.head(10))
            with st.expander("Show all authors"):
                st.write(author_frame_comps)
            
            for id_ in I.flatten().tolist():
                if id_ in set(frame.article_id):
                    f = frame[(frame.article_id == id_)]
                    author_frame = get_author_affiliation(authors_data_df,id_the= f.iloc[0].article_id)[0]
                else:
                    continue
                
                title_str = f.iloc[0].title
                st.markdown('<p class="big-font">{0}</p>'.format(title_str), unsafe_allow_html=True)
                st.markdown('<p class="medium-font">Affiliate Anti-Aging Company Name: {0}</p>'.format(f.iloc[0].company_name.capitalize()), unsafe_allow_html=True)
                
                st.write(
                        f"""
                    {newline}**Journal**: {f.iloc[0].journal}  
                    {newline}**Publication Date**: {f.iloc[0].publication_date}  
                    {newline}**Abstract**: {f.iloc[0].abstract}
                    """
                    )
                for i in f.iloc[0].keywords:
                        st.markdown('<p class="keyword-font">Top Keywords: {0}</p>'.format(i), unsafe_allow_html=True)
                with st.expander("Show information about authros and affiliations"):
                        st.write(author_frame)
                


        else:
            try:
                if filter_company:
                    frame = data[data['company_name'].isin(filter_company)]
                else:
                    index, matched_words = find_indexes_of_matching_keywords(keyword_list,data,'keywords')
                    frame = display_dataframe_withindex(data,index)


                for id_ in set(frame.article_id):
                    f = frame[(frame.article_id == id_)]
                    author_frame = get_author_affiliation(authors_data_df,id_the= f.iloc[0].article_id)[0]
                    title_str = f.iloc[0].title
                    st.markdown('<p class="big-font">{0}</p>'.format(title_str), unsafe_allow_html=True)
                    st.markdown('<p class="medium-font">Affiliate Anti-Aging Company Name: {0}</p>'.format(f.iloc[0].company_name.capitalize()), unsafe_allow_html=True)
                

                    st.write(
                        f"""
                    {newline}**Journal**: {f.iloc[0].journal}  
                    {newline}**Publication Date**: {f.iloc[0].publication_date} 
                    {newline}**Abstract**: {f.iloc[0].abstract}
                    """
                    )
                    for i in f.iloc[0].keywords:
                        st.markdown('<p class="keyword-font">{0}</p>'.format(i), unsafe_allow_html=True)
                    with st.expander("Show information about authros and affiliations"):
                        st.write(author_frame)


            
            except:
                st.write('**Sorry! We are working on replying to that query**') #see if this works if you have multiple companies
        st.caption("""There are over _3000_ abstracts in the database taken from PubMed. All publications are from _2018_ onwards.
                   These are compiled from 97 operating companies in the :blue[anti-aging] space, funded over $10B, including Altos Labs, Unity Biotechnology, Insilico Medicine, and many more.""")
        st.caption("""The data about the companies is taken from [this](https://agingbiotech.info/companies) maintained by investor and creator Karl Pfleger.""")

        st.caption("**Keyword feature may contain duplicates and is in beta mode. *Search results may not reflect all information available in the PubMed database, please search the title or DOI to get more information.")
    except Exception as e:
        st.write(e)
        st.write('**Sorry! We are working on imrpoving the app.**')

if __name__ == "__main__":
    main()
    
    
    
