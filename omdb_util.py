import requests
import json

def get_movie_details(movie_title, api_key):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url).json()
    if response.get("Response") == "True":
        plot = response.get("Plot", "N/A")
        poster = response.get("Poster", "N/A")
        return plot, poster
    return "N/A", "N/A"