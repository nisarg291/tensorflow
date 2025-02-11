# We'll start by loading up the MovieLens dataset. Using Pandas, we can very quickly load the rows of the u.data and u.item files that we care about, and merge them together so we can work with movie names instead of ID's. (In a real production job, you'd stick with ID's and worry about the names at the display layer to make things more efficient. But this lets us understand what's going on better for now.)



import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")
print(ratings)
m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")
print(movies)
ratings = pd.merge(movies, ratings)
# print(ratings.head())

# Now the amazing pivot_table function on a DataFrame will construct a user / movie rating matrix. Note how NaN indicates missing data - movies that specific users didn't rate.
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
# print(movieRatings.head())

# Let's extract a Series of users who rated Star Wars:
starWarsRatings = movieRatings['Star Wars (1977)']
# print(starWarsRatings.head())

# Pandas' corrwith function makes it really easy to compute the pairwise correlation of Star Wars' vector of user rating with every other movie! After that, we'll drop any results that have no data, and construct a new DataFrame of movies and their correlation score (similarity) to Star Wars:
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
# print('result',df.head(10))

# (That warning is safe to ignore.) Let's sort the results by similarity score, and we should have the movies most similar to Star Wars! Except... we don't. These results make no sense at all! This is why it's important to know your data - clearly we missed something important.
print("similar movies final result ",similarMovies.sort_values(ascending=False).head(60))

# Our results are probably getting messed up by movies that have only been viewed by a handful of people who also happened to like Star Wars. So we need to get rid of movies that were only watched by a few people that are producing spurious results. Let's construct a new DataFrame that counts up how many ratings exist for each movie, and also the average rating while we're at it - that could also come in handy later.
import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
# print(movieStats.head())

# Let's get rid of any movies rated by fewer than 100 people, and check the top-rated ones that are left:
popularMovies = movieStats['rating']['size'] >= 100
# print('result',movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15])

# 100 might still be too low, but these results look pretty good as far as "well rated movies that people have heard of." Let's join this data with our original set of similar movies to Star Wars:
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
print("result re ",df.sort_values('similarity', ascending=False).head(50))