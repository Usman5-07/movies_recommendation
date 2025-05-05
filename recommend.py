import joblib 
import logging

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

try: 
    logging.info("Loading dataset...")
    df = joblib.load('cleaned_movie_df.pkl')
    logging.info("Dataset loaded successfully.")
except FileNotFoundError as e: 
    logging.error("Dataset file not found. Please check the file path.")
    raise e

df = joblib.load('cleaned_movie_df.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')

def recommend_movies(movie_title, n_top):
    movie_id = df[df['title'] == movie_title].index
    if movie_id is None:
        logging.error("Movie not found in the dataset.")
        return None
    movie_id = movie_id[0]
    sim_score = list(enumerate(cosine_sim[movie_id]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:n_top + 1]
    movie_indices = [i[0] for i in sim_score]
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = 'S.No'
    return result_df

