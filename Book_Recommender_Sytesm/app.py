
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings


warnings.filterwarnings('ignore')

book = pd.read_excel("book_pivot.xlsx")

book = book.set_index("title")

filename = 'final_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))

st.markdown('<style>body{background-color: Orange;text-align: center;text-color:black}</style>', unsafe_allow_html=True)
st.title("Book Recommendation System")


def recommend_book(book_name):

    book_id = np.where(book.index == book_name)[0][0]

    distances, suggestions = loaded_model.kneighbors(book.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    st.title("Recommended Books: \n")

    st.write(pd.DataFrame({
        'Book Sr.No.': [1, 2, 3, 4, 5],
        'Book Name': book.index[suggestions[0][1:]],
    }, index=None))


with st.form("my_form"):

    user_input = st.text_input("BOOK NAME", "")
    submitted = st.form_submit_button("Submit")

if submitted:
    st.write('You have submitted!')
    recommend_book(user_input)
else:
    st.write('Please submit!')