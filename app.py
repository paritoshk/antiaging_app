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
import openpyxl

st.set_page_config(layout="wide", page_title='Longevity AI', page_icon='🤖', initial_sidebar_state='auto')

# CSS styles
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
def change_empty_lists_to_string(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Change empty double lists in a column to 'Nothing here'."""
    df_copy = df.copy(deep=True)
    df_copy[column] = df_copy[column].apply(lambda x: 'Nothing here' if x == ['[]'] else x)
    return df_copy

@st.cache(allow_output_mutation=True)
def read_data(data: str = "data/publications/final_database_of_papers.xlsx") -> pd.DataFrame:
    """Read and process the data from local to avoid errors."""
    try:
        data = pd.read_excel(data, index_col=0)
        data['keywords'] = data['keywords'].apply(lambda x: literal_eval(x) if "[" in x else x)
        data = change_empty_lists_to_string(data, 'keywords')
        data['publication_date'] = pd.to_datetime(data['publication_date']).dt.date
        data['company_name'] = data['company_name'].str.strip().str.capitalize()
        return data
    except Exception as e:
        print(f"Error reading data: {e}")
        return pd.DataFrame()

@st.cache(allow_output_mutation=True)
def find_indexes_of_matching_keywords(list_of_keywords: list, df: pd.DataFrame, column: str) -> tuple:
    """Find indexes of rows that match elements in the input list with elements in lists in the column."""
    indexes = []
    matched_words = []
    df_copy = df.copy(deep=True)
    df_copy[column] = df_copy[column].apply(lambda x: [item.lower() for item in x])
    list_of_keywords = [item.lower() for item in list_of_keywords]
    for i, row in enumerate(df_copy[column]):
        for p in list_of_keywords:
            for j, item in enumerate(row):
                if p in item:
                    indexes.append(i)
                    matched_words.append((p, df[column][i][j]))
    return indexes, matched_words

@st.cache(allow_output_mutation=True)
def display_dataframe_withindex(df: pd.DataFrame, indexes: list) -> pd.DataFrame:
    """Display dataframe with specified indexes."""
    return df.iloc[indexes]

@st.cache(allow_output_mutation=True)
def authors_data(data: str = "data/publications/authors_dataframe.csv") -> pd.DataFrame:
    """Read the authors data from local."""
    return pd.read_csv(data, index_col=0)

@st.cache(allow_output_mutation=True)
def get_author_affiliation(df: pd.DataFrame, company_name: str = None, id_the: int = 31740545) -> tuple:
    """Get author name and affiliation given a PubMed ID."""
    df_copy = df.copy(deep=True)
    if id_the:
        df_copy = df_copy[df_copy['pubmed_id'] == id_the]
    if company_name:
        df_copy_list = [df_copy[df_copy['company_name'] == i] for i in company_name]
        df_copy = pd.concat(df_copy_list)
    df_copy = df_copy.reset_index(drop=True)
    print_list = [f"{row['author_name']} is the author and {row['affiliation']} is the affiliation"
                  for _, row in df_copy.iterrows()]
    return df_copy[['author_name', 'affiliation']], print_list

@st.cache(allow_output_mutation=True)
def highlight_company_auths(match_string: str, case_df: pd.DataFrame) -> list:
    """Highlight a string in the 'affiliation' column of a dataframe."""
    match_string = match_string.lower()
    match_index = []
    for i, affiliation in enumerate(case_df['affiliation']):
        affiliation_string = affiliation.lower()
        if any(j in affiliation_string for j in match_string.split()):
            match_index.append(i)
    return match_index

@st.cache(allow_output_mutation=True)
def load_bert_model() -> SentenceTransformer:
    """Instantiate a sentence-level model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache(allow_output_mutation=True)
def load_keywords(path_to_keywords: str = "pickle_files/keyword_final.pickle") -> list:
    """Load and deserialize the keywords."""
    with open(path_to_keywords, "rb") as h:
        return pickle.load(h)

@st.cache(allow_output_mutation=True)
def load_embeddings(path_to_embeddings: str = "pickle_files/embeddings.pickle") -> np.ndarray:
    """Load and deserialize the embeddings."""
    with open(path_to_embeddings, "rb") as h:
        return pickle.load(h)

@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss: str = "pickle_files/fiass_index.pickle") -> faiss.IndexFlatL2:
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        return faiss.deserialize_index(pickle.load(h))

@st.cache(allow_output_mutation=True)
def vector_search(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, num_results: int = 10) -> tuple:
    """Transform query to vector using the model and find similar vectors using FAISS."""
    vector = model.encode([query])
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I

@st.cache(allow_output_mutation=True)
def id2details(df: pd.DataFrame, I: np.ndarray, column: str) -> list:
    """Return the paper titles based on the paper index."""
    return [list(df[df.article_id == idx][column]) for idx in I[0]]

@st.cache(allow_output_mutation=True)
def frame_builder(data: pd.DataFrame, filter_company: list, keyword_list: list) -> tuple:
    """Build a dataframe based on the selected company and keywords."""
    frame = data[data['company_name'].isin(filter_company)]
    index, matched_words = find_indexes_of_matching_keywords(keyword_list, data, 'keywords')
    frame = display_dataframe_withindex(data, index)
    return frame, index, matched_words

def render_search_results(frame: pd.DataFrame, keyword_list: list, authors_data_df: pd.DataFrame):
    """Render the search results."""
    for id_ in set(frame.article_id):
        paper_row = frame[frame.article_id == id_]
        author_frame = get_author_affiliation(authors_data_df, id_the=paper_row.iloc[0].article_id)[0]
        title_str = paper_row.iloc[0].title
        st.markdown(f'<p class="big-font">{title_str}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="medium-font">Affiliate Anti-Aging Company Name: {paper_row.iloc[0].company_name.capitalize()}</p>',
                    unsafe_allow_html=True)
        st.write(f"""
            **Journal**: {paper_row.iloc[0].journal}  
            **Publication Date**: {paper_row.iloc[0].publication_date} 
            **Abstract**: {paper_row.iloc[0].abstract}
            """)
        st.markdown('<p class="medium-font">Top Keywords:</p>', unsafe_allow_html=True)
        if paper_row.iloc[0].keywords != 'Nothing here':
            if len(paper_row.iloc[0].keywords) > 1:
                keyword_string = ', '.join(paper_row.iloc[0].keywords)
                for keyword in keyword_list:
                    keyword_string = keyword_string.replace(keyword, f'**{keyword}**')
                st.markdown(f'<p class="keyword-font">{keyword_string}</p>', unsafe_allow_html=True)
            else:
                for keyword in paper_row.iloc[0].keywords:
                    st.markdown(f'<p class="keyword-font">{keyword}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="keyword-font">{paper_row.iloc[0].keywords}</p>', unsafe_allow_html=True)
        st.markdown('<p class="medium-font">Authors:</p>', unsafe_allow_html=True)
        with st.expander("Show affiliations working in the company"):
            index_temp = highlight_company_auths(paper_row.iloc[0].company_name, author_frame)
            frame_sample = author_frame.iloc[index_temp].reset_index(drop=True)
            st.table(frame_sample.style.apply(lambda x: ["background: darkred"]))
        with st.expander("Show All authors"):
            st.table(author_frame)

def main():
    try:
        # Load data and models
        data = read_data()
        model = load_bert_model()
        faiss_index = load_faiss_index()
        list_combined_keywords = load_keywords()
        authors_data_df = authors_data()
        
        # Instructions
        instructions = """
        1. Enter a search term in the search box. Leaving the box empty and pressing (CTRL/CMD + ENTER) will show all publications. This is a free text semantic search and will return results based on context, meaning and concept.
        2. Select keywords from the dropdown. The keyword search is a soft match. Free text has precedence over keywords.
        3. Select a company or companies from the dropdown. This will filter the results to strictly show results from the selected company.
        4. Select the number of results you want to see. The default is 5."""
        
        # Company list
        company_list = list(set(data['company_name'].tolist()))
        
        # Streamlit app
        st.title("🧬 Longevity-AI 🧪")
        st.header("👨‍🔬: Hi! I am your due diligence associate.")
        st.write("🤖: Ask me about the :blue[anti-aging] industry. Please try to be specific and elaborate. I am still learning.")
        st.write("💊: This does a semantic search over abstracts within :blue[anti-aging] & longevity industry and research*")
        user_input = st.text_area("Type below. I will try to be accurate. Pardon me if I reply nothing.",
                                  "Tell me about research in methylation using stem cells in mouse models")
        st.caption('Try these - 1) tell me how red meat affects cancer 2) show me research about how lung fibrosis occurs. etc')
        
        # Sidebar 
        st.sidebar.markdown("**Filters**")
        keyword_list = st.sidebar.multiselect('Select Keywords (**beta feature, choose multiple)', list_combined_keywords,
                                              ['x-ray crystallography', 'gfat2', 'haploid mouse embryonic stem cells'])
        filter_company = st.sidebar.multiselect('Select a Company or Companies', company_list, "Altos labs")
        num_results = st.sidebar.slider("Number of search results", 5, 20, 5)
        
        # Instructions
        st.sidebar.markdown("**Instructions**")
        st.sidebar.markdown(f'<p class="small-font">{instructions}</p>', unsafe_allow_html=True)
        st.sidebar.markdown("*If no results appear - try broadening your criteria, keywords or deselecting your filters*")
        
        # Fetch results
        if user_input or keyword_list:
            D, I = vector_search([user_input], model, faiss_index, num_results)
            
            if filter_company:
                frame = data[data['company_name'].isin(filter_company)]
                if keyword_list:
                    index, matched_words = find_indexes_of_matching_keywords(keyword_list, frame, 'keywords')
                    frame = display_dataframe_withindex(frame, index)
            else:
                frame = data
            
            render_search_results(frame[frame.article_id.isin(I.flatten().tolist())], keyword_list, authors_data_df)
            
            # Summary plots
            st.header("Summary plots, followed by results - top 10")
            company_namedf = frame['company_name'].value_counts().rename_axis('unique_values')
            journal_df = frame['journal'].value_counts().rename_axis('unique_values')
            st.subheader("1. Number of papers per company")
            st.bar_chart(company_namedf.head(10))
            st.subheader("2. Number of papers per journal")
            st.bar_chart(journal_df.head(10))
        
        else:
            if filter_company:
                frame = data[data['company_name'].isin(filter_company)]
            else:
                index, matched_words = find_indexes_of_matching_keywords(keyword_list, data, 'keywords')
                frame = display_dataframe_withindex(data, index)
            
            render_search_results(frame, keyword_list, authors_data_df)
            
            # Summary plots
            st.header("Summary plots, followed by results - top 10")
            company_namedf = frame['company_name'].value_counts().rename_axis('unique_values')
            journal_df = frame['journal'].value_counts().rename_axis('unique_values')
            st.subheader("1. Number of papers per company")
            st.bar_chart(company_namedf.head(10))
            st.subheader("2. Number of papers per journal")
            st.bar_chart(journal_df.head(10))
        
        # App information
        st.caption("""There are over _3000_ abstracts in the database taken from PubMed. All publications are from _2018_ onwards.
                   These are compiled from 97 operating companies in the :blue[anti-aging] space, funded over $10B, including Altos Labs, Unity Biotechnology, Insilico Medicine, and many more.""")
        st.caption("""The data about the companies is taken from [this](https://agingbiotech.info/companies) maintained by investor and creator Karl Pfleger.""")
        st.caption("**Keyword feature may contain duplicates and is in beta mode. *Search results may not reflect all information available in the PubMed database, please search the title or DOI to get more information.")
        
        # Stats
        st.subheader("**Stats**")
        st.write("Total number of Publications:", len(data))
        st.write("Total number of Companies: ", len(company_list))
        st.write("Total number of Keywords: ", len(list_combined_keywords))
        st.write("Total number of Journals: ", len(set(data['journal'].to_list())))
    except Exception as e:
        st.write(e)
        st.write('**Sorry! We are working on imrpoving the app.**')

if __name__ == "__main__":
    main()
    
    
    
