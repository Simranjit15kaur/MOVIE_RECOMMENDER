import numpy as np
import pandas as pd
movies = pd.read_csv(r"C:\Users\hp\Downloads\archive\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\hp\Downloads\archive\tmdb_5000_credits.csv")

print(movies.head())
print("\nDISPLAY LIST OF ALL THE COLUMN NAMES")
print(movies.columns)
print("\nDISPLAY ALL VALUES FOR THE FIRST ROW")
print(movies.iloc[0].values)

print("CREDITS FILE DISPLAY")
print(credits.head())
print("\nDISPLAY LIST OF ALL THE COLUMN NAMES")
print(credits.columns)
print("\nDISPLAY ALL VALUES FOR THE FIRST ROW")
print(credits.iloc[0].values)

print("\nvalues of Ist index")
print(credits.head(1)['cast'].values)

#first merge the two dataset :its hectic to work with two datasets:
print("\nbefore merging : movies",movies.shape)
print("\nbefore merging : credits",credits.shape)
#here total should have been 24 ,but we have merged on the basis of title so it won't be counted twice
movies = movies.merge(credits,on='title')
print("\nafter merging result",movies.shape)
#merged movies with credits on the basis of title
print(movies.columns)


#Step1:DATA PREPROCESSING
#Remove the columns that are least important for decision making in this project:
#1)budget high is not a criteria to decide for a person in movie recommendation : REMOVE IT
#2)genre: is important like people like a specific kind of movies :REMOVE IT
#3)homepage:no importance
#4)id is important :need to fetch in the system
#5)keywords: basically tags,important
print(movies['original_language'].value_counts())
#6)original_language:here,we found that english movies are 96 percent almost ,huge difference: no need to keep this
#7)original_title can be in any regional language ,so we keep title only in english: Remove original_title
#8)overview:recommend on the basis of content similarity, just compare the overview to distinguish between two movies
#9)popularity: in this ,we are on the way to create tage, and popularity is a numeric value, so remove
#10)production_companies: we don't recommend movies on the basis of this in real life
#11)release_date: people like specific time period movies ,different choices for children and adults but, its numeric value, so remove
#tagline: its textual , but tagline is weird ,vage:Remove ,already we have overview
#cast: important  , crew: includes director

#genres, id , keywords, title, overview, cast, crew
movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.columns)
print(movies.head(1).values)

#now, we need to keep the movie_id ,title and tag :mere all others in the tag
#but in genere and keywords , we need to do preprocessing
print(movies.isnull().sum())
#here , we dont know the overview of 3 movies : so drop them
movies.dropna(inplace=True)
print(movies.isnull().sum())    #after dropping

#removing duplicates
print("Duplicated rows:",movies.duplicated().sum())

#genre column in right format:
print("\ngenre column: ",movies.iloc[0].genres)
# here, it in in the form of list of dictionary, convert it ['Action','Adventure','Science Fiction']

# def convert(obj):
#     L=[]
#     for i in obj:  #list of dictionary : i is one dictionary
#         L.append(i['name'])
#     return L

#here the covert function doesn't work because of the string involved :strings must be integers
#we need to convert the string of list to list
import ast
cov= ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
print("\n\nconverted : ",cov)

#updated function
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
print("\n\nconverted genre:\n",movies['genres'])

movies['keywords'] = movies['keywords'].apply(convert)
print("\n\nconverted keywords:\n",movies['keywords'])

print("\n",movies['cast'][0])
#fetch only name value and top 3 actors
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
print("\n\nconverted cast column" ,movies['cast'])


print("\n\n",movies['crew'][0])
#in crew, we only need director department
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
movies['crew']=movies['crew'].apply(fetch_director)
print("\nConverted crew\n",movies['crew'])


print("\n",movies['overview'].values)
#here, overview is string type, we need to convert the string to list
movies['overview']=movies['overview'].apply(lambda x:x.split())
print("\n\nconverted overview\n", movies['overview'])


#remove the spaces from like Sam Worthington to SamWorthington
#if spaces are not removed , then Sam and Worthington are treated as different entities , as different tags
#also confusion between Sam Worthington and Sam Mendes
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])    #replace space with nothing
print("\n",movies['genres'])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])    #replace space with nothing
print("\n",movies['keywords'])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])    #replace space with nothing
print("\n",movies['cast'])
movies['crew']=movies['crew']. apply(lambda x:[i.replace(" ","")for i in x])    #replace space with nothing
print("\n",movies['crew'])

#concatenation :overview,genres,keywords,cast,crew
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
print("\n",movies['tags'].head(1).values)

new_df= movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))    #converting list into string
print("\nFinal df\n",new_df.head(1).values)

#convert tags into lower letters
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
print("\nFinal df\n",new_df.head(1).values)

#Step 2:Text vectorization
#recommendation done by comparing the tags, if the tags contain more than a ceratin values with same woords ,they are similar movies, but not very mush useful
#vectorization : like the x,y and z axis :movies act as as vector and closest vectors are similar
#for this , we'll be using Bag of words
#Bag Of Words: all tags are combined/concatenated , now from this large text get 5000 most common words with large frequency
#like if action is most repeated word from the large text, then search in first movie that whats the frequency of word action in ist movie , similarly go for all the common words and keep searching them in the movies
#in short, searching the common words from the concatenated tag  for each movie, creates a table (5000,5000) ,
#for two common words, we get dim (5000,2) , 5000 vectors are in 2 dim
#now if someone ask that i like this movie ,then which all movies should be recommended : closest vectors are recommended
#why only 5000? because more words increases the dimensionality of the data that creates problems
#in vectorization , we won't be considering the stopwords, like are ,is , in ,to.... : remove them

#in sklearn , use CountVectorizer :
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000,stop_words='english')
vectors= cv.fit_transform(new_df['tags']).toarray()
print(vectors)
vectorsShape= cv.fit_transform(new_df['tags']).toarray().shape
print("\n",vectorsShape)

#now ,each movie is in vector form , we have taken 5000 words ,so in a single movie,5000 words are little impossible, so sparse matrix
corpus= cv.get_feature_names_out()
#corpus is after joining all the tags ,the text we get is corpus
print("\n".join(corpus))  # Prints each word on a new line



#Step3: stemming
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in  text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)
print("\n\n",new_df)


from sklearn.metrics.pairwise import cosine_similarity
similarity= (cosine_similarity(vectors))     #4806 vectors
#calculating the distance between the ist and all movies(vectors), then 2nd movie and all vectors ... so on
#total distances calculating : (4806*4806)
print(similarity[0])

#function : fetch the movie index, from movie index get into similarity matrix ; similarity matrix of eaxh movie
#sort the distances in the matrix: in descending order
def recommend(movie):
    movie_index= new_df[new_df['title']== movie].index[0]
    distances= similarity[movie_index]
    #by sorting the similarity matrix, index position is lost ;1 in the matrix for 0 index meant that its a ist movie
    #calling the enumerate function : list become list of tuples
    #sorted(list(enumerate(similarity[0]))),reverse=True,key=lambda x:x[1])
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    #here key represents that we need to sort on the basis of 2nd ; value not index
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


import pickle
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))








