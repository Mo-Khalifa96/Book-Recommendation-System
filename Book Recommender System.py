#BOOK RECOMMENDATION SYSTEM

import re
import math
import requests
import textwrap
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform, pdist, jaccard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.simplefilter("ignore")
sns.set_context('paper')


#Defining helper functions
#Define function to display books by their covers
def get_covers(books_df: pd.DataFrame):
    n_books = len(books_df.index)
    n_cols = ((n_books + 1) // 2) if n_books > 5 else n_books
    n_rows = math.ceil(n_books / n_cols)
    
    #create figure and specify subplot characeristics
    plt.figure(figsize=(4.2*n_cols, 6.4*n_rows), facecolor='whitesmoke')
    plt.subplots_adjust(bottom=.1, top=.9, left=.02, right=.88, hspace=.32)  
    plt.rcParams.update({'font.family': 'Palatino Linotype'})   #adjust font type

    #request, access and plot each book cover 
    for i in range(n_books):
        try:
            response = requests.get(books_df['cover_image_uri'].iloc[i])
        except:
            print('\nCouldn\'t retrieve book cover. Check your internet connection and try again...\n\n', flush=True)
            return
        
        #access and resize image
        img = Image.open(BytesIO(response.content))
        img = img.resize((600, 900))
        
        #shorten and wrap book title
        full_title = books_df['book_title'].iloc[i]
        short_title = re.sub(r'[:?!].*', '', full_title)
        title_wrapped = "\n".join(textwrap.wrap(short_title, width=26))
        
        #plot book cover 
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        plt.title(title_wrapped, fontsize=21, pad=15)
        plt.axis('off')    
    plt.show()



#PART ONE: READING AND INSPECTING THE DATA
#Loading and reading the dataset
#access and read data into dataframe
df = pd.read_csv('Book_Details.csv', index_col='Unnamed: 0')

#drop unnecessary columns
df = df.drop(['book_id', 'format', 'authorlink', 'num_pages'], axis=1)

#Inspecting the data
#report the shape of the dataframe
shape = df.shape
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])

#Preview first 5 entries
df.head()

#Inspect coloumn headers, data type, and number of entries
df.info()

#get overall description of object columns
df.describe(include='object').T

#get statistical summary of the numerical data
df.describe().drop(['25%', '50%', '75%']).apply(lambda x: round(x))



#PART TWO: CLEANING AND UPDATING THE DATA
#Removing duplicate books
#first, normalize book titles by removing punctuation
df['normalized_title'] = df['book_title'].apply(lambda title: re.sub(r'[^\w\s]', '', title))

#drop duplicate book titles and reset dataframe index
df = df.drop_duplicates(subset='normalized_title', ignore_index=True)


#Dealing with missing or inappropriate book details
#check the number of books with inappropriate book description or NaN (not a number) values
print('Number of entries with NaN values in the book details column: ', df['book_details'].isna().sum())

#fill NaN book details with empty strings
df['book_details'] = df['book_details'].fillna('')

#check the number of entries after
print('\nNumber of entries with NaN values in the book details column (after): ', df['book_details'].isna().sum())


#Cleaning and updating the Genres column
#Changing string list to list then to string with the genres of books
df['genres'] = df['genres'].apply(lambda x: ', '.join(eval(x)))

#Updating rows with no genre
#get indices of books with no genre labels
no_genre_before = df[df['genres'].str.len() == 0].index
#we can preview the books identified
df.iloc[no_genre_before, 1:8].head(3)

#Get total number of books with no genre before the update
print('Total number of entries with missing genre (before): ', len(df.iloc[no_genre_before]))

#change empty strings with genres common to given author
for i in no_genre_before:
    genre_labels = df[df['author']==df['author'].iloc[i]]['genres'].iloc[0]
    if len(genre_labels) > 0:
        df.at[i, 'genres'] = genre_labels
    else:
        df.drop(index=i, inplace=True)
#resetting dataframe index
df.reset_index(drop=True, inplace=True)

#check number of books with no genre after the update
no_genre_after = df[df['genres'].str.len() == 0].index
print('\nTotal number of entries with missing genre (after): ', len(df.iloc[no_genre_after]))


#Dealing with conflicting book genres
#create empty list for storing indices of books with conflicting genres and set count to zero
indices=[]
count=0
#loop over and return all books with conflicting genres
for genre_string, title in zip(df['genres'], df['book_title']):
    if 'Fiction' in genre_string and 'Nonfiction' in genre_string:
        count += 1
        indices.append(df[df['book_title']==title].index)
        print(f'{count}. {title} // {genre_string}')

#create dictionary with sub-strings to be replaced or removed
replacements_dict = { 'Military Fiction': 'Military',
                      'Literary Fiction': 'Literary',
                      'Realistic Fiction': 'Realistic',
                      'Non Fiction': 'Nonfiction' }

#replace substrings according to specified values
df['genres'] = df['genres'].replace(replacements_dict, regex=True)

#Now we can check again
count=0
for genre_string, title in zip(df['genres'], df['book_title']):
    if 'Fiction' in genre_string and 'Nonfiction' in genre_string:
        count += 1
print(f'Number of books with conflicting genres:  {count}')


#Creating a column with publication year
#Changing string list in publication info column to normal string
df['publication_info'] = df['publication_info'].apply(lambda x: eval(x)[0] if len(eval(x)) > 0 else 'n.d.')

#extract year of publication from publication info column and assign it to a new data column, 'publication_year' (if 'n.d.' assign an empty string)
df['publication_year'] = df['publication_info'].str.extract(r'(\d{1,4}$)').fillna('')

#preview changes and new publication year column
df[['publication_info', 'publication_year']].sample(5)



#PART THREE: EXPLORATORY DATA ANALYSIS
#Identifying and visualizing the top 20 genres featured in the dataset
#Create one-hot encoded dataframe with all unique genres in the data
genres_df = df['genres'].str.get_dummies(', ').astype(int)

#preview genres dataframe
genres_df.head()

#Extract top 20 genres by genre frequency
top20_genres = genres_df.sum().sort_values(ascending=False)[:20]

#Visualize top 20 genres using bar chart
top20_genres.plot(kind='bar', color='#24799e', width=.8,
                        linewidth=.8, edgecolor='k', rot=90)


#Identifying the top 10 books in the dataset
#Assign appropriate data type to the rating distribution column
df['rating_distribution'] = df['rating_distribution'].apply(lambda x: eval(x))

#get total number of five star ratings per book from the rating distribution column
df['total_5star_ratings'] = [int(dic['5'].replace(',','')) for dic in df['rating_distribution']]

#sort data by books with highest frequency of 5 star ratings
top10_books = df.sort_values(by='total_5star_ratings', ascending=False).iloc[:10][['book_title', 'author', 'genres', 'cover_image_uri']].reset_index(drop=True)

#report the results table
top10_books.iloc[:,:3]

#get and display books by cover
get_covers(top10_books)


#Visualize Distribution of rating scores
#Aggregate ratings by rating star
rating_counts = {'5':0, '4':0, '3':0, '2':0, '1':0}
for ratings in df['rating_distribution']:
    for key, value in ratings.items():
        rating_counts[key] += int(value.replace(',',''))

#plot the ratings frequency distribution
plt.figure(figsize=(7,5))
plt.bar(rating_counts.keys(), rating_counts.values(), color='#24799e', width=.7, linewidth=.8, edgecolor='k')
plt.title('Frequency Distribution of Star Ratings', fontsize=11)
plt.xlabel('Star Rating', fontsize=10)
plt.ylabel('Frequency of Rating', fontsize=10)
plt.grid(axis='y', linestyle='-', alpha=.7)
plt.show()


#Relationship between number of ratings and average ratings
#Visualize the relationship between the number of ratings and the average rating
# score for a given book using scatter plot
plt.figure(figsize=(9,5))
sns.scatterplot(data=df, x='num_ratings', y='average_rating')
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.xticks(rotation=-30)
plt.title('Relationship between Number of Ratings and Average Rating', fontsize=12)
plt.xlabel('Number of Ratings', fontsize=11)
plt.ylabel('Average Book Rating', fontsize=11)
plt.show()



#PART FOUR: FEATURE ENGINEERING—COMBINING FEATURES
#Combine features for ovarall text processing
df['combined_features'] = (df['book_title'] + ' / ' + df['author'] + ' / ' + df['publication_year'] + ' / ' + df['genres'] + ' / ' + df['average_rating'].apply(lambda x: str(x)) + ' / ' + df['book_details'])

#preview a sample of the combined features column
for row in df['combined_features'].sample(5):
    print(row[:200],'\n')



#PART FIVE: TEXT VECTORIZATION AND PROCESSING
#Define custom tokenizer to process text better
def my_tokenizer(text):
    #Remove punctuation and standardize text (all in lowercase, no whitespace)
    tokens = re.findall(r'\b\w+\b', text.lower().strip())
    return tokens

#Identifying Overall Similarity: Text vectorization with TF-IDF
#Create TF-IDF object and set text vectorization characteristics
tfidf_vectorizer = TfidfVectorizer(stop_words='english',   #remove common english words (e.g., the, then)
                                   tokenizer=my_tokenizer,   #specify text tokenizer (to process and standardize terms)
                                    ngram_range=(1,2),      #specify n-gram range
                                    min_df=2)      #specify min_df to filter out uncommon terms

#fit and transform the data to get a TF-IDF matrix
tfidf_mtrx = tfidf_vectorizer.fit_transform(df['combined_features'])

#Now computing cosine distance similarity
#calculate cosine distance similarity to obtain similarity matrix
similarity_mtrx = cosine_similarity(tfidf_mtrx, tfidf_mtrx)



#Identifying Genre Similarity
#Convert genres_df to CSR matrix
genres_csr_mtrx = csr_matrix(genres_df.values).astype(bool).toarray()

#Compute jaccard distance similarity to obtain jaccard similarity matrix
genre_sim_mtrx = 1 - squareform(pdist(genres_csr_mtrx, metric=jaccard))

#normalize jaccard distance scores
genre_sim_mtrx = genre_sim_mtrx / np.max(genre_sim_mtrx) if np.max(genre_sim_mtrx) > 0 else genre_sim_mtrx



#PART SIX: BUILDING A BOOK RECOMMENDATION FUNCTION
#Define helper functions to return book recommendations
def Get_Recommendations(title: str, sim_mtrx: np.ndarray, genre_sim_mtrx: np.ndarray, alpha=0.5, N=10):
    """
    This function takes a book title and recommends similar books that cover similar themes
    or fall within the same genre categories.

    Parameters:
    - title (str): The title of the book for which recommendations are sought.
    - sim_mtrx (ndarray): A similarity matrix based on book overall similarities, where each
      row corresponds to a book and each column corresponds to its cosine similarity score
      with other books.
    - genre_sim_mtrx (ndarray): A similarity matrix based on book genres, where each row
      corresponds to a book and each column corresponds to its jaccard similarity score with
      other books based on genre.
    - alpha (float, optional): Weighting factor for combining overall similarity and genre
      similarity. Defaults to 0.5, balancing overall similarity and genre similarity together.
    - N (int, optional): Number of recommendations to return. Defaults to 10.

    Returns:
    - Data table (Series) with recommended books and plot of each book with its cover.

    Raises:
    - TypeError: If the title provided is not a string.

    Notes:
    - This function filters, preprocesses and standardizes the book titles given, identifies its genre
      categories, importantly, identifying whether it's Fiction or Nonfiction work to prevent genre
      overall while looking for recommendations.
    - It looks for book recommendations by combining similarity scores from two matrices: sim_mtrx
      (based on overall similarities) and genre_sim_mtrx (based on genres).
    - It prioritizes books with similar genre categories; otherwise, it recommends book based on
      overall book similarity.
    - Finally, recommendations are filtered to include books by a different variety of authors, limiting
      the number of recommendations to only 5 books per one author.
    - The number of book recommendations can be adjusted using the 'N' parameter. Default is 10 book recommendations.
    """

    #check if title provided is of the correct data type (string)
    try:
        curr_title = str(title)
    except:
        raise TypeError('Book title entered is not string.')

    #standardize titles for accurate comparisons
    title = curr_title.lower().strip()
    full_titles = df['book_title'].apply(lambda title: title.lower().strip())
    partial_titles = full_titles.str.extract(r'^(.*?):')[0].dropna()

    #check if provided title matches book title in the dataset and get index if found
    if title in full_titles.values:
        indx = df[full_titles == title].index[0]

    elif title in set(partial_titles.values):
        indx_partial = partial_titles[partial_titles == title].index[0]
        indx = df[df['book_title'] == df['book_title'].iloc[indx_partial]].index[0]

    else:
        #try normalizing book titles across the board by removing punctuations and removing 'the' if the book starts with it for better comparisons
        normalized_title = re.sub(r'(^\s*(the|a|an)\s+|[^\w\s])', '', title, flags=re.IGNORECASE)
        normalized_full_titles = full_titles.apply(lambda title: re.sub(r'(^\s*(the|a|an)\s+|[^\w\s])', '', title, flags=re.IGNORECASE))
        normalized_partial_titles = partial_titles.apply(lambda title: re.sub(r'(^\s*(the|a|an)\s+|[^\w\s])', '', title, flags=re.IGNORECASE))
        if normalized_title in normalized_full_titles.values:
            indx = df[normalized_full_titles == normalized_title].index[0]

        elif normalized_title in set(normalized_partial_titles.values):
            indx_partial = normalized_partial_titles[normalized_partial_titles==normalized_title].index[0]
            indx = df[df['book_title'] == df['book_title'].iloc[indx_partial]].index[0]

        else:
            print(f'\nBook with title \'{curr_title}\' is not found. Please try a different book.\n', flush=True)
            return False


    #Check if 'Fiction' is in the genre of the selected book
    is_fiction = 'Fiction' in df['genres'].iloc[indx]

    #Find books with the same genre category
    if is_fiction:
        book_indices_ByGenre = [i for i in df.index if ('Fiction' in df['genres'].iloc[i]) and (i != indx)]
    else:
        book_indices_ByGenre = [i for i in df.index if ('Fiction' not in df['genres'].iloc[i] or 'Nonfiction' in df['genres'].iloc[i]) and (i != indx)]


    #Combine the two similarity matrices using weighted sum
    weighed_similarity = (alpha * sim_mtrx[indx]) + ((1 - alpha) * genre_sim_mtrx[indx])

    #Get cosine similarity scores for books with the same genre
    similarity_scores = [(i, weighed_similarity[i]) for i in book_indices_ByGenre]

    #Filter scores to only include books with the same genre category
    similarity_scores = [score for score in similarity_scores if score[0] in book_indices_ByGenre]

    #Sort the books based on the genre similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    #If less than N books are found in the same genre category, add books by closest overall cosine distance
    if len(similarity_scores) < N:
        cos_scores = list(enumerate(weighed_similarity[indx]))
        cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse=True)
        cos_scores = [score for score in cos_scores if score[0] != indx and score[0] not in [x[0] for x in similarity_scores]]  #Exclude the same book and already recommended books
        similarity_scores += [score for score in cos_scores if score not in similarity_scores][:N - len(similarity_scores)]

    #Limit recommendations to 5 books per author
    author_counts = {}
    similarity_scores_filtered = []
    for score in similarity_scores:
        author = df['author'].iloc[score[0]]
        if author not in author_counts or author_counts[author] < 5:
            similarity_scores_filtered.append(score)
            author_counts[author] = author_counts.get(author, 0) + 1


    #Get the scores of the N most similar books
    most_similar_books = similarity_scores_filtered[:N]
    #Get the indices of the books selected
    most_similar_books_indices = [i[0] for i in most_similar_books]

    #Prepare DataFrame with recommended books and their details
    recommended_books = df.iloc[most_similar_books_indices][['book_title', 'author', 'cover_image_uri']]
    recommended_books['Recommendation'] = recommended_books.apply(lambda row: f"{row['book_title']} (by {row['author']})", axis=1)
    recommended_books.reset_index(drop=True, inplace=True)

    #Return book recommendations
    print(f"\nRecommendations for '{curr_title.title()}' (by {df['author'].iloc[indx]}):", flush=True)
    print(recommended_books['Recommendation'].to_frame().rename(lambda x:x+1), flush=True, end='\n\n')
    get_covers(recommended_books)
    return



#PART SEVEN: TESTING THE RECOMMENDATION SYSTEM
#Adjust pandas display settings to display entire title
pd.set_option('display.max_colwidth', None)

#Generating Book Recommendation for Famous Title
#Get book recommendations for 'Macbeth' (by Shakespeare)
book_title = 'Macbeth'
Get_Recommendations(book_title, similarity_mtrx, genre_sim_mtrx, alpha=0.7, N=10)


#Generating Book Recommendation from Random titles
#Get recommendations for titles chosen at random
random_titles = df.sample(5)[['book_title','author']]

#get recommendations for the selected titles
for title,author in zip(random_titles.iloc[:,0],random_titles.iloc[:,1]):
    Get_Recommendations(title, similarity_mtrx, genre_sim_mtrx, alpha=0.7, N=10)
    print('\n', 150*'_' + '\n')


#Generating Book Recommendations from User Input
#Defining custom function that requests a book title from the user and returns relevant book recommendations
def Get_Recommendations_fromUser():
    while True:
        book_title = input('\nEnter book title: ')     
        recommendations = Get_Recommendations(book_title, similarity_mtrx, genre_sim_mtrx, alpha=0.7, N=10)
        print('\n', 150*'_' + '\n', flush=True)
        if recommendations is not False:
            response = str(input('\nWould you like to get recommendations for more books? [Yes/no]\n')).lower().strip()
            if response in ['yes', 'y']: 
                continue 
            elif response in ['no', 'n']:
                print('\nThank you for trying the recommender.\nExiting...')
                break
            else: 
                print('\nResponse invalid.\nProcess terminating...')
                break

#Execute the user recommender function
Get_Recommendations_fromUser()  # The Great Gatsby; Return of the king; Atomic Habit; Atomic Habits; a brief history of time; Critique of pure reason
