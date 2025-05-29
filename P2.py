import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Load the Movielens 100k dataset (ml-100k.zip) into Python using Pandas data 
# frames.
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('u.data', sep='\t', names=column_names) 
utility_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating')
user_mean_ratings = utility_matrix.mean(axis=1)
centered_utility_matrix = utility_matrix.subtract(user_mean_ratings, axis=0)
centered_utility_matrix_filled = centered_utility_matrix.fillna(0)

# Build a user profile on centered data (by user rating) for both users 200 
# and 15
user_200_profile = centered_utility_matrix_filled.loc[200]
user_15_profile = centered_utility_matrix_filled.loc[15]
print("User 200's profile (partial data):")
print(user_200_profile.head())
print("\nUser 15's profile (partial data):")
print(user_15_profile.head())

# Create a vector of the same length as the user profile vector, 
# where only the position representing item 95 is 1 and the rest are 0.
item_id_target = 95
item_vector = pd.Series(0, index=centered_utility_matrix_filled.columns)

if item_id_target in item_vector.index:
    item_vector[item_id_target] = 1
else:
    print(f"Error: Item ID {item_id_target} is not in the columns of the utility matrix. Please check item_id.")
    exit()

print(f"\nVector representation of item {item_id_target} (partial, around item {item_id_target}):")
start_col_idx = max(0, item_vector.index.get_loc(item_id_target) - 2)
end_col_idx = min(len(item_vector.index), item_vector.index.get_loc(item_id_target) + 3)
print(item_vector.iloc[start_col_idx:end_col_idx])

# and calculate the cosine similarity and distance between the user’s 
# preferences and the item/movie 95.
user_200_profile_np = user_200_profile.values.reshape(1, -1)
item_vector_np = item_vector.values.reshape(1, -1)

cosine_sim_user_200_item_95 = cosine_similarity(user_200_profile_np, item_vector_np)[0][0]
distance_user_200_item_95 = euclidean_distances(user_200_profile_np, item_vector_np)[0][0]

print(f"\nUser 200 and Item {item_id_target}:")
print(f"  Cosine Similarity: {cosine_sim_user_200_item_95:.4f}")
print(f"  Euclidean Distance: {distance_user_200_item_95:.4f}")

user_15_profile_np = user_15_profile.values.reshape(1, -1)

cosine_sim_user_15_item_95 = cosine_similarity(user_15_profile_np, item_vector_np)[0][0]
distance_user_15_item_95 = euclidean_distances(user_15_profile_np, item_vector_np)[0][0]

print(f"\nUser 15 and Item {item_id_target}:")
print(f"  Cosine Similarity: {cosine_sim_user_15_item_95:.4f}")
print(f"  Euclidean Distance: {distance_user_15_item_95:.4f}")

print("\nRecommendation Decision：")
if cosine_sim_user_200_item_95 > cosine_sim_user_15_item_95:
    reco_by_cosine = "User 200"
elif cosine_sim_user_15_item_95 > cosine_sim_user_200_item_95:
    reco_by_cosine = "User 15"
else:
    reco_by_cosine = "Both have the same similarity"

if distance_user_200_item_95 < distance_user_15_item_95:
    reco_by_distance = "User 200"
elif distance_user_15_item_95 < distance_user_200_item_95:
    reco_by_distance = "User 15"
else:
    reco_by_distance = "Both have the same distance"

print(f"Based on cosine similarity, item {item_id_target} is more likely to be recommended to {reco_by_cosine}.")
print(f"Based on Euclidean distance, item {item_id_target} is more likely to be recommended to {reco_by_distance}.")

if reco_by_cosine == reco_by_distance and reco_by_cosine != "Both have the same similarity" and reco_by_cosine != "Both have the same distance":
    final_recommendation = reco_by_cosine
    print(f"Overall, the recommender system would suggest this movie to {final_recommendation}.")
elif reco_by_cosine != "Both have the same similarity" and reco_by_cosine != "Both have the same distance":
    final_recommendation = reco_by_cosine
    print(f"The metrics might differ or one might be decisive. If cosine similarity is the primary reference, this movie would be suggested to {final_recommendation}.")
    print(f"(User 200: sim={cosine_sim_user_200_item_95:.4f}, dist={distance_user_200_item_95:.4f})")
    print(f"(User 15: sim={cosine_sim_user_15_item_95:.4f}, dist={distance_user_15_item_95:.4f})")
else:
    if reco_by_cosine != "Both have the same similarity":
        final_recommendation = reco_by_cosine
        print(f"Distances are the same, but based on cosine similarity, the recommender system would suggest this movie to {final_recommendation}.")
    elif reco_by_distance != "Both have the same distance":
        final_recommendation = reco_by_distance
        print(f"Cosine similarities are the same, but based on Euclidean distance, the recommender system would suggest this movie to {final_recommendation}.")
    else:
        final_recommendation = "either user (or no specific recommendation to one over the other)"
        print(f"Both users have a similar degree of preference for item {item_id_target}; the recommender system might not specifically favor one user.")