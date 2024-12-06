# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import datetime

# Set page configuration
st.set_page_config(page_title="📚 Book Recommendation System", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            color: #4B0082;
            text-align: center;
            margin-bottom: 20px;
        }
        .header {
            font-size: 1.5em;
            color: #333;
            margin-top: 20px;
        }
        .subheader {
            font-size: 1.2em;
            color: #555;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 0.8em;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)


# Title of the app
st.markdown('<div class="title">Welcome to the Book Recommendation System!</div>', unsafe_allow_html=True)
st.markdown("""
    Explore popular books, get personalized recommendations based on ratings, checkouts, or collaborative filtering. 📖
""")

# Sidebar for navigation and options
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select an option:", ["Popular Books", "Cluster Recommendations", "Collaborative Filtering"])

# Load the data
@st.cache_data(ttl=600)
def load_data():
    # Load data from the Excel file
    book_df = pd.read_excel('Bookshop.xlsx', sheet_name="Book")
    author_df = pd.read_excel('Bookshop.xlsx', sheet_name="Author")
    author_df['Author'] = author_df["First Name"] + " " + author_df["Last Name"]
    author_df.drop(["First Name", "Last Name", "Birthday", "Country of Residence", "Hrs Writing per Day"], axis=1, inplace=True)
    
    info_df = pd.read_excel('Bookshop.xlsx', sheet_name='Info')
    info_df['BookID'] = info_df['BookID1'] + info_df['BookID2'].astype(str)
    info_df.drop(['BookID1', 'BookID2', 'SeriesID', 'Volume Number', 'Staff Comment'], axis=1, inplace=True)
    
    checkout_df = pd.read_excel('Bookshop.xlsx', sheet_name='Checkouts')
    edition_df = pd.read_excel('Bookshop.xlsx', sheet_name='Edition')
    rating_df = pd.read_excel('Bookshop.xlsx', sheet_name='Ratings')
    rating_df.rename(columns={'ReviewerID': 'UserID'}, inplace=True)
    rating_df.drop('ReviewID', axis=1, inplace=True)

    # Merge datasets to create a unified dataset
    df = book_df.merge(author_df, on='AuthID').merge(info_df, on='BookID').merge(checkout_df, on='BookID', how='left')
    df['CheckoutMonth'].fillna(df['CheckoutMonth'].mode()[0], inplace=True)
    df['Number of Checkouts'].fillna(df['Number of Checkouts'].median(), inplace=True)
    
    df = df.merge(edition_df, on='BookID').merge(rating_df, on='BookID', how='left')
    df.dropna(inplace=True)
    
    return df[['BookID', 'Title', 'Author', 'Genre', 'CheckoutMonth',
                'Number of Checkouts', 'ISBN', 'Format', 'Pages',
                'Price', 'Rating', 'UserID']]

df = load_data()

if options == "Popular Books":
    st.header('Popular Books')
    
    top_n = st.slider('Number of books to display:', min_value=5, max_value=20, value=10)
    method = st.radio(' Select method:', ['Checkouts', 'Ratings'])

    if method == 'Checkouts':
        st.subheader("🔝 Popular Books by Checkout")
        popular_books = df.groupby('Title')['Number of Checkouts'].sum().sort_values(ascending=False).head(top_n)
        st.write(popular_books.reset_index())
        
    else:
        st.subheader("🌟 Popular Books by Ratings")
        popular_books = df.groupby('Title')['Rating'].mean().sort_values(ascending=False).head(top_n)
        st.write(popular_books.reset_index())

elif options == "Cluster Recommendations":
    st.header("🔍 Recommend Books Based on a Cluster")
    
    book_title = st.text_input('Enter a book title for recommendations:')
    
    if book_title:
        scaler = StandardScaler()
        le = LabelEncoder()
        
        df[['Number of Checkouts', 'Pages', 'Price']] = scaler.fit_transform(df[['Number of Checkouts', 'Pages', 'Price']])
        df['Genre'] = le.fit_transform(df['Genre'])
        df['Format'] = le.fit_transform(df['Format'])

        features = df[["Genre", "Number of Checkouts", "Format", "Pages", "Price", "Rating"]]
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(features)

        kmeans = KMeans(n_clusters=4)
        df["Cluster"] = kmeans.fit_predict(pca_result)

        if book_title in df["Title"].values:
            cluster_label = df[df["Title"] == book_title]["Cluster"].values[0]
            recommendations = df[(df["Cluster"] == cluster_label) & (df["Title"] != book_title)][["Title", "Author"]].drop_duplicates().reset_index(drop=True)

            if not recommendations.empty:
                st.write(recommendations)
            else:
                st.write("No similar books found.")
        else:
            st.write("Book not found.")

elif options == "Collaborative Filtering":
    st.header("🤖 Recommend Books Based on Collaborative Filtering")
    
    method = st.radio('Choose recommendation method:', ['Based on Users', 'Based on Books'])
    
    if method == "Based on Users":
        user_id = st.number_input("Enter User ID:", step=1)

        if user_id:
            user_item_matrix = df.pivot_table(index="UserID", columns="Title", values="Rating").fillna(0)
            user_similarity_matrix = cosine_similarity(user_item_matrix)
            user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

            if user_id in user_similarity_df.index:
                similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:11]
                recommendations = user_item_matrix.loc[similar_users.index].mean(axis=0).sort_values(ascending=False).index.tolist()
                st.write(recommendations[:10])
            else:
                st.write("User  not found.")

    elif method == "Based on Books":
        book_title = st.text_input("Enter Book Title:")
        
        if book_title:
            book_user_matrix = df.pivot_table(index="Title", columns="UserID", values="Rating").fillna(0)
            book_similarity_matrix = cosine_similarity(book_user_matrix)
            book_similarity_df = pd.DataFrame(book_similarity_matrix, index=book_user_matrix.index, columns=book_user_matrix.index)

            if book_title in book_similarity_df.index:
                similar_books = book_similarity_df[book_title].sort_values(ascending=False).iloc[1:11]
                st.write(similar_books.index.tolist())
            else:
                st.write("Book not found.")

# Real-time Update Feature
st.sidebar.header("Real-time Updates")
if st.sidebar.button("Add New Book"):
    new_title = st.sidebar.text_input("New Book Title:")
    new_author = st.sidebar.text_input("New Author:")
    
    if new_title and new_author:
        st.sidebar.success(f"Added '{new_title}' by {new_author}!")

# Footer with additional styling or information
st.markdown("""
---
<div class="footer">Made with ❤️ using <a href="https://streamlit.io/">Streamlit</a>. Enjoy exploring books!</div>
""", unsafe_allow_html=True)
