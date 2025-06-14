# movie.py
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 1. Streamlit page config  (must be first Streamlit command)
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ─────────────────────────────────────────────────────────────
# 2. Imports
# ─────────────────────────────────────────────────────────────
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# 3. Load CSV and build similarity matrix (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data and building similarity matrix…")
def load_and_build(csv_path: str):
    # Read CSV in tolerant chunks
    chunks = []
    for chunk in pd.read_csv(
        csv_path,
        engine="python",
        sep=",",
        chunksize=100_000,
        on_bad_lines="skip"
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Minimal cleanup
    df['Title'] = df['Title'].astype(str)
    df['Overview'] = df['Overview'].fillna("").astype(str)

    # Combine textual fields if you wish (e.g., Overview + Genre + Title)
    text_corpus = (
        df['Overview']
        + " " + df['Genre'].fillna("")
        + " " + df['Title']
    )

    # TF‑IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text_corpus)

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Deduplicate titles (case‑insensitive)
    df = df[~df['Title'].str.lower().duplicated()].reset_index(drop=True)

    # Build lookup: lowercase title → dataframe row index (int)
    indices = pd.Series(df.index, index=df['Title'].str.lower())

    return df, cosine_sim, indices


df, cosine_sim, indices = load_and_build("mymoviedb.csv")

# ─────────────────────────────────────────────────────────────
# 4. Recommendation logic
# ─────────────────────────────────────────────────────────────
def recommend_movies(title: str, df: pd.DataFrame, cosine_sim, indices, top_n: int = 10):
    """Return a DataFrame (all columns) of top‑n similar movies."""
    title = title.lower().strip()
    if title not in indices:
        return pd.DataFrame()                     # empty → not found

    idx = int(indices[title])                     # unique after dedup
    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1],
        reverse=True
    )[1 : top_n + 1]                              # skip self‑match

    movie_ids = [i for i, _ in sim_scores]
    return df.loc[movie_ids].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────
st.title("🎥 Content‑Based Movie Recommendation System")
st.markdown("Enter a movie you like and get similar movies based on plot, genre, and title text.")

movie_input = st.text_input("🔍 Movie title", placeholder="e.g., Inception")

if st.button("Recommend"):

    if not movie_input.strip():
        st.warning("Please type a movie title.")
    else:
        recs = recommend_movies(movie_input, df, cosine_sim, indices, top_n=10)

        if recs.empty:
            st.error("Movie not found. Please check spelling or try another.")
        else:
            st.success(f"Top {len(recs)} recommendations similar to **{movie_input.title()}**")

            # Display each recommendation with poster + full metadata
            for _, row in recs.iterrows():
                with st.container():
                    cols = st.columns([1, 3])  # poster | details

                    # ── Poster ──
                    with cols[0]:
                        if isinstance(row['Poster_Url'], str) and row['Poster_Url'].startswith(("http://", "https://")):
                            st.image(row['Poster_Url'], width=160)
                        else:
                            st.image("https://via.placeholder.com/120x180?text=No+Image", width=160)

                    # ── Details ──
                    with cols[1]:
                        st.markdown(f"### {row['Title']}")
                        st.markdown(f"**Release Date:** {row.get('Release_Date', 'N/A')}")
                        st.markdown(f"**Genre:** {row.get('Genre', 'N/A')}")
                        st.markdown(f"**Original Language:** {row.get('Original_Language', 'N/A')}")
                        st.markdown(f"**Popularity:** {row.get('Popularity', 'N/A')}")
                        st.markdown(f"**Vote Count:** {row.get('Vote_Count', 'N/A')}")
                        st.markdown(f"**Average Rating:** {row.get('Vote_Average', 'N/A')}")
                        st.markdown("**Overview:**")
                        st.write(row.get('Overview', 'No overview available.'))

                st.markdown("---")
