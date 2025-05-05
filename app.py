import json
import streamlit as st
from recommend import recommend_movies, df
from omdb_util import get_movie_details

OMDB_API_KEY = st.secrets["my_config"]["api_key"]


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="centered",
)

st.title("Movie Recommendation System")
st.subheader("Get movie recommendations based on your favorite movie!")

movie_list = sorted(df['title'].dropna().unique())

selected_movie = st.selectbox("Select a movie:", movie_list)
num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        recommendations = recommend_movies(selected_movie, num_recommendations)
        if recommendations is None or recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.success("Top similar movies!")
            st.dataframe(recommendations, use_container_width=True)
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_details(movie_title, OMDB_API_KEY)

                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster != 'N/A':
                            st.image(poster, width=150)
                        else:
                            st.error("Poster not available.")
                    with col2:
                        st.subheader(movie_title)
                        st.write(plot if plot != 'N/A' else "Plot not available.")
                        st.markdown("---")
            st.success("Recommendations fetched successfully!")
