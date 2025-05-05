import joblib 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging 

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('preprocess.log', encoding='utf-8'),
                              logging.StreamHandler()]
                    )

logging.info("Starting preprocessing script...")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stopword = set(stopwords.words('english'))

# Load the dataset
logging.info("Loading dataset...")

try:
    df = pd.read_csv('D:/_Portfolio/ML Coding Projects/Movie Recommendation System/movies.csv')
    logging.info("✅ Dataset loaded successfully.")
except FileNotFoundError as e:
    logging.error("❌ Dataset file not found. Please check the file path.")
    raise e


def preprocess_text(text):
    """Preprocess the text by removing special characters and stopwords."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stopword]  # Remove stopwords
    return ' '.join(tokens)  # Join tokens back to string

required_colums = ['title', 'genres', 'keywords', 'cast', 'director']
df = df[required_colums]
df = df.dropna()
df = df.reset_index(drop=True)

logging.info(f"Features selected and NaN values dropped.{df.shape}")
df['cleaned_text'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['cast'] + ' ' + df['director']
df['cleaned_text'] = df['cleaned_text'].apply(preprocess_text)

logging.info("Text ✅ preprocessing completed.")


# Vectorize the text data using TF-IDF
logging.info("Vectorizing 📉 text data...")

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

logging.info("✅ Vectorization completed.")

# Calculate cosine similarity
logging.info("Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

logging.info("✅ Cosine similarity calculated.")
# Save the model and vectorizer

joblib.dump(cosine_sim, 'cosine_similarity.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(df, 'cleaned_movie_df.pkl')

logging.info("Model and vectorizer saved successfully.")
logging.info("Preprocessing script completed successfully.")


