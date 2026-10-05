from svd_recommender import SVDRecommender


print("=" * 70)
print("BUILDING SVD MODEL")
print("=" * 70)

recommender = SVDRecommender()

recommender.fit()

recommender.save_model()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)