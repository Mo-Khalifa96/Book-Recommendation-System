# Book Recommendation System 
<br>

## About The Project 
**This project utilizes unsupervised machine learning to develop a simple but robust book recommendation system that can deliver personalized book recommendations.  A recommendation system identifies the preferences of a given user and offers relevant suggestions or related content in return. For this recommendation system, the recommender would take input from the user with the name of a given book and delivers highly tailored book recommendations in return. It leverages both content-based and genre-based similarities in providing the final recommendations. Having been trained on a large dataset of books (taken from Goodreads books database) comprised of many different books, authors, genres, reviews, plot summaries and descriptions, it identifies similarities between the input book (given by the user) and other books in the database across all these different dimensions, selects and returns the most similar or most relevant ones. This book recommendation system can also filter, preprocess, and parse text to enable better matching and comparison. It also ensures author variety and can also be easily customized to increase or decrease the number of relevant recommendations or to control the degree to which the recommendations should be content-based or genre-based or a mixture of both. All this, and more, ultimately culminates into a powerful book recommender system that can be used to search for and explore new books based on one's prior preferences and book favorites.**<br>
**To develop the book recommendation system, the dataset is first inspected, cleaned, filtered, and updated in preparation for analysis and model development. After having prepared and analyzed the data, a Term Frequency - Inverse Document Frequency (TF-IDF) vectorizer model is then employed for text vectorization and processing, converting books' important attributes (including author, genres, and plot summary or book description) into numeric vectors with TF-IDF scores capturing and representing each book and how it compares to all others in the dataset. These TF-IDF scores are then compared using cosine distance similarity to measure and map out the overall similarities between the different books, returning a large data matrix with the overall similarities between books. In addition, a separate data matrix is developed for book genres alone to identify and map out the exact genre similarities between the books (using jaccard distance similarity). With the analysis and modeling coming to completion, a book recommendation function is then developed to utilize the similarity matrices obtained in order to deliver tailored book recommendations. As mentioned, this function also features different options to control the nature of the book recommendations such as whether to recommend by genre in particular or by overall similarity more generally and how many books are to be recommended. Finally, the book recommender is put to test, first testing it with well known books (e.g., Shakespeare's 'Macbeth'), then testing it using different book titles sampled at random from the database, and then lastly using user input, in which the user can pass any book they are looking for similar recommendations for and the recommendation function takes care of the rest. You can try the recommender yourself.** <br>

<br>

**Overall, the project is broken down into 7 sections: <br>
&emsp; 1) Reading and Inspecting the Data <br>
&emsp; 2) Cleaning and Updating the Data <br> 
&emsp; 3) Exploratory Data Analysis <br>
&emsp; 4) Feature Engineering: Combining features <br>
&emsp; 5) Text Vectorization and Processing <br>
&emsp; 6) Building a Book Recommendation Function <br>
&emsp; 7) Testing the Recommendation System** <br>

<br>
<br>


## About The Data  
**The dataset presented here was taken from Kaggle, which you can access easily by clicking [here](https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024). This dataset consists of thousands of books collected from Goodreads, a popular platform for discovering, reviewing, and discussing books. Indeed, it provides a comprehensive book collection of more than 16,000 books in total, covering a myriad of different authors, genres, and literary eras, ancient and modern. It covers all the major literary works from the ancient times and up to May 2024. Each book featured, represented by a data row, covers important details and descriptions about it, including the book title, author, genre classification, publication date, format, and its average rating score. As such, the data here can support a variety of purposes, from data analysis to studying user-preferences and performing sentiment analysis to building recommendation systems, as with the current case. This dataset has been licensed by MIT for free use for commercial and non-commercial purposes.** <br> 
<br>

**You can view each column and its description in the table below:** <br><br>  

| **Variable**      | **Description**                                                                                         |
| :-----------------| :------------------------------------------------------------------------------------------------------ |
| **book_id**       | Unique identifier for each book in the data                                                             |
| **cover_image_uri**| URI or URL pointing to the cover image of the book                                                     |
| **book_title**    | Title of the book                                                                                       |
| **book_details**  | Details about the book, including summary, plot, synopsis or other descriptive information              |
| **format**        | Details about the format of the book such as whether it's a hardcover, paperback, or audiobook          |
| **publication_info** | Information about the publication of the book including the publisher, publication date, or any other relevant details |
| **authorlink**    |   URI or URL pointing to more information about the author (if available)                               |
| **author**        | Name of the book author(s)                                                                              |
| **num_pages**     | Number of pages                                                                                         |
| **genres**        | Genre labels applying to the book                                                                       |
| **num_ratings**   | Total number of ratings                                                                                 |
| **num_reviews**   | Total number of reviews                                                                                 |
| **average_rating** | Overall average rating score                                                                           |
| **rating_distribution** | Number of ratings per rating star (for a 5-point rating system)                                   |

<br>
<br>


## Quick Access 
**To quickly access the project, I provided two links, both of which will direct you to a Jupyter Notebook with all the code and corresponding output, rendered, organized into sections and sub-sections, and supplied with thorough explanations and insights that gradually guide the unfolding of the project from one step to the next. The first link allows you to view the project only, whereas the second link allows you to view the project as well as interact with it directly and reproduce the results if you prefer so. To execute the code, please make sure to run the first two cells first in order to install and import the Python packages necessary for the task. To run any given block of code, simply select the cell and click on the 'Run' icon on the notebook toolbar.**
<br>
<br>
<br>

***To view the project only, click on the following link:*** <br>
https://nbviewer.org/github/Mo-Khalifa96/Book-Recommendation-System/blob/main/Book%20Recommender%20System%20%28Jupyter%20Notebook%29.ipynb
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/v2/gh/Mo-Khalifa96/Book-Recommendation-System/main?labpath=Book+Recommender+System+%28Jupyter+Notebook%29.ipynb
<br>
<br>


