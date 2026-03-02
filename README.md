# Hybrid recommendation system
## Introduction
**A hybrid recommendation system for cars:**

### 3 Classes
**CaseBasedRecommender**

**CollaborativeRecommender**

**HybridRecommender**

### Datasets
- Cars for sale and their information

- Ratings and reviews for cars by many users.

Car data (CarsForSale): https://www.kaggle.com/datasets/chancev/carsforsale?resource=download

Car reviews data (Edmunds car review): https://www.kaggle.com/datasets/shreemunpranav/edmunds-car-review

## Exploratory Data Analysis (EDA) and Preprocessing of Data
**Python notebook file for EDA and preprocessing**

The car ratings dataset is aligned with the main cars dataset by creating a common carID for each car.

## Hybrid recommendation system 
Case-Based → Collaborative

### 1) Case-based recommendations 
Case-based filtering is implemented using TF–IDF vectorization and cosine similarity:
1. Fits TF-IDF vectors
2. Calculates cosine similarity between cars.
2. **Normalize** scores.

#### Features we extract for content based recommendations:

**Year, Drivetrain, Fueltype, Transmission, ExteriorColor**

**Similarity / ranking:** Cosine similarity

### 2) Collaborative-filtering (CF) 
CF is implemented using Non-negative Matrix Factorization (NMF):
1. Build a **user–item matrix** where rows are users, columns are cars, and values are ratings.
2. **NMF** decomposes this matrix into latent features representing user preferences and item characteristics.
3. **Predict** ratings for unseen cars by combining user and item latent features.
4. **Normalize** scores.

This allows the system to recommend cars a user is likely to like based on patterns from similar users.

### 3) Hybrid Recommendation
The hybrid recommender combines:

Case-based filtering 

Collaborative filtering 

The final recommendation list merges case-based and CF recommendations while removing duplicates.

#### Merge formula
**score-based weighted hybrid**

final_score=α×CF_score+(1−α)×CB_score

where α (alpha) controls the balance between the two methods.

## Evaluation & tuning 

Batch evaluation on many users:

Metrics:
- Precision @ k
- Recall @ k
- Coverage @ k
- Novelty
