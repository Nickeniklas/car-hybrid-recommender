import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import math

class CaseBasedRecommender:
    def __init__(self, cars_data):
        self.cars_data = cars_data
        self.tfidf_matrix = None
        self.car_indices = None

    def fit(self):
        """ Data preprocess and feature extraction. """
        self.cars_data['Features'] = self.cars_data['Features'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.cars_data['Features'])
        self.car_indices = pd.Series(self.cars_data.index, index=self.cars_data['Make Model Year']).drop_duplicates()

    def recommend(self, car, n=10):
        """ 
        Return top-n similiar cars

        Parameters:
            car = make, model and year (small letter and space ex. toyota sienna 2020).
            n = amount of recommends.
        """
        if car not in self.car_indices: return pd.Series([], dtype=float)
        value = self.car_indices[car]
        # Get row index of the car
        idx = value.iloc[0] if isinstance(value, pd.Series) else int(value) # single match and multiple matches handled
        # Compute cosine similarity with all cars as pandas series
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores = pd.Series(sim_scores, index=self.cars_data['carID'])
        # Exclude the seed car itself
        seed_id = self.cars_data.iloc[idx]['carID']
        scores.drop(index=seed_id, inplace=True)
        # normalize CB scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores.sort_values(ascending=False).head(n)
    
class CollaborativeRecommender:
    def __init__(self, ratings_data):
        self.ratings_data = ratings_data
        self.user_item_matrix = None
        self.user_mapper = None
        self.car_mapper = None
        self.nmf_model = None

    def fit(self):
        """ Train NMF on user-item matrix. """
        # user item matrix
        self.user_item_matrix = self.ratings_data.pivot_table(index='userID', columns='carID', values='Rating').fillna(0)
        # map user ID with row index and car ID with column index.
        self.user_mapper = {uid: i for i, uid in enumerate(self.user_item_matrix.index)}
        self.car_mapper = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
        # Initialize the NMF model
        self.nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=500)
        self.nmf_model.fit(self.user_item_matrix)

    def recommend(self, user_id, n=10):
        """Predict scores for all cars."""
        if user_id not in self.user_mapper:
            return pd.Series()  # No data for this user
        # Get user and item latent features
        user_idx = self.user_mapper[user_id]
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        user_P = self.nmf_model.transform(user_vector)
        item_Q = self.nmf_model.components_
        # Compute predicted scores for all cars
        scores = np.dot(user_P, item_Q).flatten()
        scores = pd.Series(scores, index=self.user_item_matrix.columns)
        # Exclude cars the user has already rated
        user_rated = self.ratings_data[self.ratings_data['userID'] == user_id]['carID']
        scores = scores[~scores.index.isin(user_rated)]
        # Normalize CF scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) 
        return scores.sort_values(ascending=False).head(n)

class HybridRecommender:
    def __init__(self, cars_data, ratings_data):
        # data
        self.cars_data = cars_data
        self.ratings_data = ratings_data
        # recommenders
        self.cb_model = CaseBasedRecommender(self.cars_data)
        self.cf_model = CollaborativeRecommender(self.ratings_data)

    def fit(self):
        self.cb_model.fit()
        self.cf_model.fit()

    def id_to_title(self, score_series, top_n=None):
        """ Map carID → 'Make Model Year" for readability. """
        # dataframe with car titles 
        df = self.cars_data[['carID', 'Make Model Year']].set_index('carID')
        merged = df.join(score_series.rename('Score'), how='inner')
        if top_n:
            merged = merged.sort_values('Score', ascending=False).head(top_n)
        return merged

    def recommend(self, user_id, car, n=10, alpha=0.5):
        """ Combine collaborative and case based predictions. """
        # Get collaborative and case based predictions
        cf_scores = self.cf_model.recommend(user_id)
        cb_scores = self.cb_model.recommend(car)
        # Align indexes (fill missing with 0)
        cf_scores, cb_scores = cf_scores.align(cb_scores, fill_value=0)
        # Weighted combination
        hybrid_score  = alpha * cf_scores + (1 - alpha) * cb_scores 
        # Sort and select top N
        hybrid_score = hybrid_score.sort_values(ascending=False).drop_duplicates().head(n) # OPTIONAL: .drop_duplicates() 
        return hybrid_score
    
class Evaluator:
    def __init__(self, cars_data, ratings_data):
        # data
        self.cars_df = cars_data
        self.ratings_df = ratings_data
        # Pre-calculate popularity for Novelty to save time
        self.car_popularity = self.ratings_df['carID'].value_counts()
        # Total user count for popularity fraction in Novelty 
        self.total_users = self.ratings_df['userID'].nunique()

    def precision_recall_at_k(self, actuals, recommended, k=10):
        if not actuals or not recommended: return 0, 0
        rec_k = recommended[:k]
        hits = len(set(rec_k) & set(actuals))
        
        precision = hits / k
        recall = hits / len(actuals)
        return precision, recall
    
    def coverage_at_k(self, all_unique_recs):
        """ Catalog coverage of the recommendations. Fraction of unique cars recommended at least once."""
        total_inventory = self.cars_df['carID'].nunique()
        return len(all_unique_recs) / total_inventory

    def novelty(self, all_unique_recs):
        """ Average novelty of the recommendations for all users. """
        novelty_scores = []
        for car_id in all_unique_recs:
            p = self.car_popularity.get(car_id, 1) / self.total_users
            novelty_scores.append(-np.log2(p))
        return np.mean(novelty_scores) if novelty_scores else 0
    
    def evaluate_all(self, prediction_dict, k=10):
        """
        Calculates average Precision and Recall across the batch.
        """
        precisions = []
        recalls = []
        all_unique_recs = set()

        for uid, recs in prediction_dict.items():
            actuals = self.ratings_df[self.ratings_df['userID'] == uid]['carID'].tolist()
            precisions.append(self.precision_at_k(actuals, recs, k))
            recalls.append(self.recall_at_k(actuals, recs, k))
            all_unique_recs.update(recs)

        return {
            "avg_precision": np.mean(precisions),
            "avg_recall": np.mean(recalls),
            "coverage": len(all_unique_recs) / self.cars_df['carID'].nunique()
        }

if __name__ == "__main__":
    # load data
    cars_data = pd.read_csv("./data/cars_clean.csv")
    ratings_data = pd.read_csv("./data/ratings_clean.csv")

    # test seed
    car = 'volkswagen passat 2.0 tdi sel 2012'

    # initialize and fit recommender model
    recommender = HybridRecommender(cars_data, ratings_data)
    # Fit the model
    recommender.fit()

    evaluator = Evaluator(cars_data, ratings_data)

    # Batch generation of recommendations for evaluation
    test_users = ratings_data['userID'].unique()[:10] # Sample 50 users
    prediction_dict = {}

    avg_p, avg_r = [], []
    all_recs_set = set()

    print("Generating recommendations...")
    for uid in test_users:
        # Get ground truth
        actuals = ratings_data[ratings_data['userID'] == uid]['carID'].tolist()
        
        # Get model output
        # case based predictions 
        cb_scores = recommender.cb_model.recommend(car)
        print("Case-based predictions recommendations:\n", recommender.id_to_title(cb_scores, 5))
        # collaborative
        cf_scores = recommender.cf_model.recommend(uid)
        print("Collaborative recommendations:\n", recommender.id_to_title(cf_scores, 5))
        # hybrid 
        hybrid_scores = recommender.recommend(uid, car, n=10, alpha=0.5)
        print("Hybrid recommendations:\n", recommender.id_to_title(hybrid_scores, 10))
        rec_ids = hybrid_scores.index.tolist() if isinstance(hybrid_scores, pd.Series) else hybrid_scores
        
        # Store for batch metrics
        prediction_dict[uid] = rec_ids
        all_recs_set.update(rec_ids)
        
        # Calculate per-user metrics
        p, r = evaluator.precision_recall_at_k(actuals, rec_ids, k=10)
        avg_p.append(p)
        avg_r.append(r)

    # 3. Final Evaluation
    print(f"--- Results ---")
    print(f"Precision@10: {np.mean(avg_p):.4f}")
    print(f"Recall@10:    {np.mean(avg_r):.4f}")
    print(f"Coverage:     {evaluator.coverage_at_k(all_recs_set):.4f}")
    print(f"Novelty:      {evaluator.novelty(all_recs_set):.4f}")

    