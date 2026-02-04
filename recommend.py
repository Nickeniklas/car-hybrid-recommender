import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import math

class ContentBasedRecommender:
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
    def __init__(self, cars_data_path, ratings_data_path):
        # data
        self.cars_data = pd.read_csv(cars_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        # recommenders
        self.cb_model = ContentBasedRecommender(self.cars_data)
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
        """ Combine collaborative and content-based predictions. """
        # Get collaborative and content-based scores
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
    def __init__(self, recommender, cars_data_path, ratings_data_path, user_id):
        self.recommender = recommender
        self.cars_data = pd.read_csv(cars_data_path)
        self.ratings_data = pd.read_csv(ratings_data_path)
        self.user_actuals = self.ratings_data[self.ratings_data['userID'] == user_id]['carID'].tolist()
        self.user_id = user_id
    
    def precision_at_k(self, recs, k=10):
        """ Proportion of relevant items that were recommended. """
        if recs.empty or not self.user_actuals:
            print("precision@k error: no recommendations")
            return 0
        rec_k = recs[:k]
        relevant = len(set(rec_k.index) & set(self.user_actuals))
        return relevant / len(self.user_actuals)
    
    def coverage_at_k(self, k=10):
        """ Catalog coverage of the recommendations. Fraction of unique cars recommended at least once."""
        all_recs = self.get_all_recs(k) 
        total_cars = self.cars_data['Make Model Year'].drop_duplicates()
        return len(all_recs) / len(total_cars)

    def novelty(self, k=10):
        """ Average novelty of the recommendations for all users. """
        # all recommendations for all users
        all_recs = self.get_all_recs(k)
        # Popularity scores
        car_popularity = self.ratings_data['carID'].value_counts()
        total_users = self.ratings_data['userID'].nunique()
        popularity_fraction = car_popularity / total_users

        novelty_scores = []
        for car_id in all_recs:
            # calculate novelty
            p = popularity_fraction.get(car_id, 1 / (total_users + 1))
            novelty_scores.append(-np.log2(p))

        return np.mean(novelty_scores) # average of all scores
    
    def get_all_recs(self, k=10):
        """ Get all unique car recommendations for all users. """
        all_user_ids = self.ratings_data['userID'].drop_duplicates()
        all_recs = set()  

        for user_id in all_user_ids:
            recs = self.recommender.recommend(user_id=user_id, car=None, n=k) # car=None - only do CF
            # if returned as series, convert index to list
            if isinstance(recs, pd.Series):
                recs = recs.index.tolist()
            all_recs.update(recs)  # add unique recommendations
        return all_recs

if __name__ == "__main__":
    # dataset paths
    cars_data_path = "./data/cars_clean.csv"
    ratings_data_path = "./data/ratings_clean.csv"

    # initialize hybrid recommender
    hybrid = HybridRecommender(cars_data_path, ratings_data_path)
    hybrid.fit()

    # test seeds
    car = 'volkswagen passat 2.0 tdi sel 2012'
    user_id = 27583
    
    # content based
    cb_scores = hybrid.cb_model.recommend(car)
    print("Content-based recommendations:\n", hybrid.id_to_title(cb_scores, 5))

    # collaborative
    cf_scores = hybrid.cf_model.recommend(user_id)
    print("Collaborative recommendations:\n", hybrid.id_to_title(cf_scores, 5))

    # hybrid 
    hybrid_scores = hybrid.recommend(user_id, car, n=10, alpha=0.5)
    print("Hybrid recommendations:\n", hybrid.id_to_title(hybrid_scores, 10))

    # Evaluation
    eval = Evaluator(hybrid, cars_data_path, ratings_data_path, user_id)

    # precision@k
    prec_at_k = eval.precision_at_k(hybrid_scores, user_id)
    print("Precision@k:\n", prec_at_k)

    # coverage@k
    cov_at_k = eval.coverage_at_k()
    print("Coverage@k:\n", cov_at_k)

    # novelty
    novelty = eval.novelty()
    print("Mean Novelty of all users:", novelty)