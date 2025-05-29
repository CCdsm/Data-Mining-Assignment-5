import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Load the Movielens 100k dataset (ml-100k.zip) into Python using Pandas data frames.

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('u.data', sep='\t', names=column_names)

print("Rating data:")
print(ratings_df.head())


# Convert the ratings data into a utility matrix representation and find the 
# 10 most similar users for user 1 based on the cosine similarity of the centered 
# user ratings data.
utility_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating')
print("\nUtility matrix (top 5 users and top 5 items):")
print(utility_matrix.iloc[:5, :5])

user_mean_ratings = utility_matrix.mean(axis=1)
centered_utility_matrix = utility_matrix.subtract(user_mean_ratings, axis=0)
centered_utility_matrix_filled = centered_utility_matrix.fillna(0)
user_1_centered_ratings = centered_utility_matrix_filled.loc[1].values.reshape(1, -1)
user_similarities = cosine_similarity(user_1_centered_ratings, centered_utility_matrix_filled)[0]
similarity_scores = pd.Series(user_similarities, index=centered_utility_matrix_filled.index)
similar_users = similarity_scores.sort_values(ascending=False)
similar_users = similar_users.drop(1)

top_10_similar_users = similar_users.head(10)

print("\nThe top 10 users most similar to User 1:")
print(top_10_similar_users)
top_10_user_ids = top_10_similar_users.index.tolist()

# Based on the average of the ratings for item 508 from similar 
# users, what is the expected rating for this item for user 1?
item_id_to_predict = 508
ratings_for_item_508_by_similar_users = utility_matrix.loc[top_10_user_ids, item_id_to_predict]
ratings_for_item_508_by_similar_users = ratings_for_item_508_by_similar_users.dropna()

if ratings_for_item_508_by_similar_users.empty:
    expected_rating_for_item_508 = float('nan')
    print("\nNAN!!")
else:
    expected_rating_for_item_508 = ratings_for_item_508_by_similar_users.mean()

print(f"\nRatings for item {item_id_to_predict} from similar users:")
print(ratings_for_item_508_by_similar_users)
print(f"\nUser 1's expected rating for item {item_id_to_predict} is: {expected_rating_for_item_508:.2f}")
