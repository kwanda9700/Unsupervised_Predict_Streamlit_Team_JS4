"""
    Streamlit webserver-based Recommender Engine.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.
    NB: !! Do not remove/modify the code delimited by dashes !!
    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st

# Import seaborn library
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# To create interactive plots
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.express as px


# Data handling dependencies
import pandas as pd
import numpy as np
import codecs
from wordcloud import WordCloud, STOPWORDS

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# Importing data
movies = pd.read_csv('resources/data/movies.csv')
train = pd.read_csv('resources/data/ratings.csv')
df_imdb = pd.read_csv('resources/data/imdb_data.csv')

# Merging the train and the movies
df_merge1 = train.merge(movies, on = 'movieId')

from datetime import datetime
# Convert timestamp to year column representing the year the rating was made on merged dataframe
df_merge1['rating_year'] = df_merge1['timestamp'].apply(lambda timestamp: datetime.fromtimestamp(timestamp).year)
df_merge1.drop('timestamp', axis=1, inplace=True)

# -------------- Create a Figure that shows us that shows us how the Ratigs are distriuted. ----------------#
# Get the data
data = df_merge1['rating'].value_counts().sort_index(ascending=False)

ratings_df = pd.DataFrame()
ratings_df['Mean_Rating'] = df_merge1.groupby('title')['rating'].mean().values
ratings_df['Num_Ratings'] = df_merge1.groupby('title')['rating'].count().values

genre_df = pd.DataFrame(df_merge1['genres'].str.split('|').tolist(), index=df_merge1['movieId']).stack()
genre_df = genre_df.reset_index([0, 'movieId'])
genre_df.columns = ['movieId', 'Genre']

def make_bar_chart(dataset, attribute, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if sort_index == False:
        xs = dataset[attribute].value_counts().index
        ys = dataset[attribute].value_counts().values
    else:
        xs = dataset[attribute].value_counts().sort_index().index
        ys = dataset[attribute].value_counts().sort_index().values


    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)

    plt.bar(x=xs, height=ys, color=bar_color, edgecolor=edge_color, linewidth=2)
    plt.xticks(rotation=45)

# Merging the merge data earlier on with the df_imbd
df_merge3 = df_merge1.merge(df_imdb, on = "movieId" )

num_ratings = pd.DataFrame(df_merge3.groupby('movieId').count()['rating']).reset_index()
df_merge3 = pd.merge(left=df_merge3, right=num_ratings, on='movieId')
df_merge3.rename(columns={'rating_x': 'rating', 'rating_y': 'numRatings'}, inplace=True)

# pre_process the budget column

# remove commas
df_merge3['budget'] = df_merge3['budget'].str.replace(',', '')
# remove currency signs like "$" and "GBP"
df_merge3['budget'] = df_merge3['budget'].str.extract('(\d+)', expand=False)
#convert the feature into a float
df_merge3['budget'] = df_merge3['budget'].astype(float)
#remove nan values and replacing with 0
df_merge3['budget'] = df_merge3['budget'].replace(np.nan,0)
#convert the feature into an integer
df_merge3['budget'] = df_merge3['budget'].astype(int)

df_merge3['release_year'] = df_merge3.title.str.extract('(\(\d\d\d\d\))', expand=False)
df_merge3['release_year'] = df_merge3.release_year.str.extract('(\d\d\d\d)', expand=False)

data_1= df_merge3.drop_duplicates('movieId')

num_ratings = pd.DataFrame(df_merge3.groupby('movieId').count()['rating']).reset_index()
df_merge4 = pd.merge(left=df_merge3, right=num_ratings, on='movieId')
df_merge4.rename(columns={'rating_x': 'rating', 'rating_y': 'NumberRatings'}, inplace=True)

# Dropping the duplicates in the movies
Remove_duplicates = df_merge4.drop_duplicates('movieId')

# Movies published by year:

years = []

for title in df_merge3['title']:
    year_subset = title[-5:-1]
    try: years.append(int(year_subset))
    except: years.append(9999)

df_merge3['moviePubYear'] = years
print('The Number of Movies Published each year:',len(df_merge3[df_merge3['moviePubYear'] == 9999]))

def make_histogram(dataset, attribute, bins=25, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if attribute == 'moviePubYear':
        dataset = dataset[dataset['moviePubYear'] != 9999]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    #ax.set_yticklabels([yticklabels(item, 'M') for item in ax.get_yticks()])
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)

    plt.hist(dataset[attribute], bins=bins, color=bar_color, ec=edge_color, linewidth=2)

    plt.xticks(rotation=45)
    
def get_genre_list(df,column) :
    genres_list = []
    for genre in df[column].unique():
        genres_list = genres_list + genre.split("|")
        genres_list = list(set(genres_list))
    return genres_list
    
# Make a census of the genre keywords
def get_genre_labels(df,column) :
    genre_labels = set()
    for s in movies['genres'].str.split('|').values:
        genre_labels = genre_labels.union(set(s))
    return genre_labels

genre_labels = get_genre_list(movies, 'genres')
    
# Function that counts the number of times each of the genre keywords appear
def count_word(dataset, ref_col, census):
   
    keyword_count = dict()
    for s in census: 
        keyword_count[s] = 0
    for census_keywords in dataset[ref_col].str.split('|'):        
        if type(census_keywords) == float and pd.isnull(census_keywords): 
            continue        
        for s in [s for s in census_keywords if s in census]: 
            if pd.notnull(s): 
                keyword_count[s] += 1
                
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

keyword_occurences, dum = count_word(movies, 'genres', genre_labels)

avg_genre_ratings = df_merge3.groupby(['genres'], as_index=False)['rating'].mean()
avg_genre_ratings = avg_genre_ratings.sort_values(by=['rating'], ascending=False)


# ------------------------------ CODE FOR THE FIGURES ENDS HERE ------------------------------------#

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Exploratory Data Analysis", "Project Overview", "About Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

# ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Exploratory Data Analysis":
        st.header("Movie Recommender System Datasets")
#         st.sidebar.header("Configuration")
        st.sidebar.subheader("The EDA sections below:")
        all_cols = df_merge1.columns.values
        numeric_cols = df_merge1.columns.values
        obj_cols = df_merge1.columns.values

        if st.sidebar.checkbox("Visuals on Ratings"):
            if st.checkbox("Distribution of Movie ratings"):
                trace = go.Bar(x = data.index, y = data.values, marker = dict(color='#0080ff'))
                layout = dict(title='Distribution of Movie ratings'.format(df_merge1.shape[0]), xaxis = dict(title='rating'), yaxis = dict(title = 'Count'))
                fig = go.Figure(data=[trace], layout=layout)
                st.write(fig)
            
            if st.checkbox("Number of rated movies"):
                user_ratings = df_merge3.groupby(by='userId')
                d = user_ratings['rating'].count()
                limit = 200
                fig, ax = plt.subplots(figsize=(14, 7))
                plt.hist(d[d<=limit], bins='fd')
                plt.xlabel('number of rated movies')
                plt.ylabel('number of users')
                print(f'Only users with less than {limit} ratings are displayed ({len(user_ratings) - len(d[d<=limit]):,} users omitted).')
                plt.show()
                st.write(fig)
                
            if st.checkbox("Average rating for a movie / by a user"):
                users_average = df_merge1.groupby('userId')['rating'].mean()
                items_average = df_merge1.groupby('movieId')['rating'].mean()
                fig, ax = plt.subplots(figsize=(14, 7))
                plt.hist([users_average, items_average], histtype='step', density=True)     
                plt.xlabel('average rating for a movie / by a user')
                plt.ylabel('number of movies / users')
                plt.legend(['average rating given by a user', 'average rating of a movie'], loc=2)
                plt.show()
                st.write(fig)
        
        if st.sidebar.checkbox("Visuals on Genres"):
            if st.checkbox("Genre wordcloud"):
                genres = dict()
                trunc_occurences = keyword_occurences[0:18]
                for s in trunc_occurences:
                    genres[s[0]] = s[1]
                genre_wordcloud = WordCloud(width=1000,height=400, background_color='black')
                genre_wordcloud.generate_from_frequencies(genres)
                fig, ax = plt.subplots(figsize=(16, 8))
                plt.imshow(genre_wordcloud, interpolation="bilinear")
                plt.axis('off')
                plt.show()
                st.write(fig)
                
            if st.checkbox("The number of movie per genre"):
                st.info("The number of movie per genre")
                fig=make_bar_chart(genre_df, 'Genre', title='Most Popular Movie Genres', xlab='Genre', ylab='Counts')
                st.pyplot(fig)

        if st.sidebar.checkbox("Visuals on Movies"):
            if st.checkbox("wordcloud of the movie titles"):
                movies['title'] = movies['title'].fillna("").astype('str')
                title_corpus = ' '.join(movies['title'])
                fig = plt.figure(figsize=(16,8))
                title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)     
#                 plt.figure(figsize=(16,8))
                plt.imshow(title_wordcloud)
                plt.axis('off')
                plt.show()
                st.write(fig)
            
            if st.checkbox("Number of ratings per director"):
                Director_ratings = pd.DataFrame(Remove_duplicates.groupby('director').sum()['NumberRatings'].sort_values(ascending=False)).reset_index()
                fig = plt.figure(figsize = (14, 9.5))
                sns.barplot(data = Director_ratings.head(50), y = 'director', x = 'NumberRatings', color = 'Blue')     
                plt.ylabel('Directors')
                plt.xlabel('Number of ratings')
                plt.title('Number of ratings per director\n')
                plt.show()
                st.write(fig)
                
            if st.checkbox("Number of Movies released per director"):
                director_movies = pd.DataFrame(Remove_duplicates.groupby('director').count()['title'].sort_values(ascending=False)).reset_index()
                fig = plt.figure(figsize = (14, 9.5))
                sns.barplot(data = director_movies.head(50), y = 'director', x = 'title', color = 'Blue')     
                plt.ylabel('Directors')
                plt.xlabel('Number of movies released')
                plt.title('Number of Movies released per director\n')
                plt.xlim(0, 27)
                plt.show()
                st.write(fig)

#     st.sidebar.header("About")
#     st.sidebar.text("Team name : Unsupervised_Team_JS4")
#     st.sidebar.text(
#         "Code : https://github.com/kwanda9700/Unsupervised_Predict_Streamlit_Team_JS4"
#     )




    if page_selection == "Project Overview":
        st.title("Project Overview")
#         st.write("Describe your winning approach on this page")

#         st.markdown("""A `Recommender System (RS)` is no doubt one of the most obvious ways in which companies are enhancing the user experience
#         in the platform that they provide their customers services. Companies Like Facebook, Netflix, Amazon, and Youtube are using RS to do so.
#         More likely, these companies and other companies that are implementing the RS are doing so in introducing machine learning into these
#         companies. It is therefore important for aspiring Data Scientists to develop skills in such areas. At `Explore Data Science Academy (EDSA)`,
#         this team was given a task to build a RS. There are 3 available approaches to building a recommender system. As part of this project the
#         team explored two of these which were the `Content Based Filtering (CBF)` and `Collaborative Filtering (CF)` algorithm.
#             """)

#         st.subheader("**Collaborative Filtering (CF)**")
#         st.markdown("""This recommender engine was easy to implement in this work as it provides us with the recommendation of the 10 movies easily
#          as compared to the other approach. On the other hand, the CF is one of the most popular implemented recommender engines and it is based on
#          the assumption that the people were in agreement in the past and there is a high chance that they are in agreement in the future. An example
#           indicating what is meant by the statement about agreement is considering that a friend and the other friend have probably liked an identical
#           range of books in the past. Because the friend has now read new books that the other has not read there is a high chance that the other friend
#           will enjoy and probably like those same books. This logic describes what is known as `user-based` collaborative filtering which was implemented
#           in this application. """)

#         st.subheader("**Building the Recommender Sytem**")
#         st.markdown("""The recommender system application was built mainly for consumers to have an experience of watching movies that they are
#         likely to enjoy based on the three movies they have selected. Figure below shows a recommender engine from Netflix showing new release
#          movies. Ideally, more recommender systems look like the one from the figure below, however, the approach to building this one was somehow
#          different. """)
        st.subheader("** Introduction **")
        st.markdown("""
        The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient systems and this is where Recommendation Systems come into play. Recommendation Systems are a type of information filtering systems as they improve the quality of search results and provides items that are more relevant to the search item or are realted to the search history of the user. 
        """)

        st.subheader("** What is recommendation system? **")
        st.markdown("""
        Recommender System is a system that seeks to predict or filter preferences according to the user’s choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. Moreover, companies like Netflix and Spotify depend highly on the effectiveness of their recommendation engines for their business and sucees.
        """)


        image=Image.open("./images/rs.jpg")
        st.image(image, use_column_width=True)

        st.markdown("""The current recommendation systems that are bring used and are popular are the content-based filtering and collaborative filtering which works by implementing different information sources to make the recommendations.

- Content-based filtering (CBF) : makes recommendations based on user preferences for product features.
- Collaborative filtering (CF): mimics user-to-user recommendations (i.e. it relies on how other users have responded to the same items). 

It predicts users preferences as a linear, weighted combination of other user preferences.
We have to note that both of these methods have limitations: The CBF can recommend a new item but needs more data on user preferences to give out the best match. On the other hand, the CF requires large dataset with active users who rated the product before to make the most accurate predictions. The combination of both of these methods is known as hybrid recommendation systems.
            """)
        
        ## Problem statement:

        st.markdown("""
Construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences. """ )

    if page_selection == "About Us":
        # st.markdown("<h1 style='text-align: center; color: black;'>About Us</h1>", unsafe_allow_html=True)
        image = Image.open("./images/about_us3.jpg")
        st.image(image, use_column_width=True)
        
        st.title("Our Team")
#         st.write("Describe your winning approach on this page")

        st.markdown("""
- Kwanda Silekwa
- Thembinkosi Malefo
- Sihle Riti
- Nomfundo Manyisa
- Ofentse Sabe
- Thanyani Khedzi
            """)

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
