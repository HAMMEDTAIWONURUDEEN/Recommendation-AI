
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Page setup ----------
st.set_page_config(
    page_title="Course Recommender",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    margin: 0;
    color: white;
}

.hero p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    margin-right: 6px;
    border-radius: 999px;
    background: #e0e7ff;
    color: #1e3a8a;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load saved files ----------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_assets():

    df_path = BASE_DIR / "courses_df.joblib"
    sim_path = BASE_DIR / "sim.joblib"

    if not df_path.exists():
        raise FileNotFoundError(
            "courses_df.joblib was not found in the project folder."
        )

    if not sim_path.exists():
        raise FileNotFoundError(
            "sim.joblib was not found in the project folder."
        )

    df = joblib.load(df_path)
    sim = joblib.load(sim_path)

    # Reset index to keep DataFrame positions aligned
    # with the rows and columns of the similarity matrix.
    df = df.reset_index(drop=True)

    # Ensure the required columns exist.
    required_columns = [
        "course_title",
        "course_organization",
        "course_difficulty",
        "course_rating"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Recreate the text used during model training.
    if "text" not in df.columns:
        df["text"] = df["course_title"].fillna("").str.lower()

    # Rebuild TF-IDF using the same settings as the original model.
    extra_words = {
        "course", "crash", "introduction", "intro", "using"
    }

    custom_stops = list(
        ENGLISH_STOP_WORDS.union(extra_words)
    )

    tfidf = TfidfVectorizer(
        stop_words=custom_stops,
        ngram_range=(1, 2)
    )

    X = tfidf.fit_transform(df["text"].fillna(""))

    # Confirm that the saved similarity matrix matches the dataset.
    sim = np.asarray(sim)

    if sim.shape != (len(df), len(df)):
        raise ValueError(
            "The similarity matrix dimensions do not match "
            "the course dataset. Please regenerate sim.joblib."
        )

    # Prepare ratings.
    df["course_rating"] = pd.to_numeric(
        df["course_rating"], errors="coerce"
    )

    return df, sim, X, tfidf


try:
    df, sim, X, tfidf = load_assets()

except Exception as error:
    st.error(f"Could not load recommendation model: {error}")
    st.stop()


# ---------- Recommendation logic ----------
def rank(
    scores,
    exclude_idx,
    n,
    difficulty,
    rating_weight
):

    # Start with all course positions.
    candidate_indices = np.arange(len(df))

    # Exclude the selected course itself.
    if exclude_idx is not None:
        candidate_indices = candidate_indices[
            candidate_indices != exclude_idx
        ]

    # Keep courses with positive similarity.
    candidate_indices = candidate_indices[
        scores[candidate_indices] > 0
    ]

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    # Filter difficulty before ranking.
    if difficulty != "Any":

        difficulty_values = (
            df["course_difficulty"]
            .fillna("")
            .str.lower()
        )

        candidate_indices = candidate_indices[
            difficulty_values.iloc[candidate_indices].values
            == difficulty.lower()
        ]

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    result = df.iloc[candidate_indices].copy()

    result["similarity"] = scores[candidate_indices]

    # Normalize course ratings.
    r_min = df["course_rating"].min()
    r_max = df["course_rating"].max()

    if pd.isna(r_min) or pd.isna(r_max):
        result["rating_norm"] = 0.5

    elif r_max == r_min:
        result["rating_norm"] = 0.5

    else:
        result["rating_norm"] = (
            (result["course_rating"] - r_min)
            / (r_max - r_min)
        )

    result["rating_norm"] = result["rating_norm"].fillna(0.5)

    # Combine similarity and rating.
    result["score"] = (
        (1 - rating_weight) * result["similarity"]
        + rating_weight * result["rating_norm"]
    )

    return (
        result.sort_values("score", ascending=False)
        .head(n)
    )


# ---------- Display recommendation cards ----------
def show_cards(results):

    for _, row in results.iterrows():

        with st.container(border=True):

            left, right = st.columns([4, 1])

            with left:

                st.markdown(f"**{row['course_title']}**")

                organization = html.escape(
                    str(row["course_organization"])
                )

                difficulty = html.escape(
                    str(row["course_difficulty"])
                )

                rating = (
                    f"{row['course_rating']:.2f}"
                    if pd.notna(row["course_rating"])
                    else "N/A"
                )

                st.markdown(
                    f"<span class='badge'>{organization}</span>"
                    f"<span class='badge'>{difficulty}</span>"
                    f"<span class='badge'>⭐ {rating}</span>",
                    unsafe_allow_html=True
                )

            with right:

                st.metric(
                    "Match",
                    f"{row['similarity'] * 100:.0f}%"
                )


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🎓 Course Recommender</h1>
    <p>
        Find courses similar to one you like, or search by topic.
        Built with TF-IDF and cosine similarity.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Sidebar filters ----------
with st.sidebar:

    st.header("Filters")

    difficulty_options = (
        df["course_difficulty"]
        .dropna()
        .unique()
        .tolist()
    )

    difficulty_options = ["Any"] + sorted(difficulty_options)

    difficulty = st.selectbox(
        "Difficulty",
        difficulty_options
    )

    n = st.slider(
        "Number of results",
        3, 10, 5
    )

    rating_weight = st.slider(
        "Importance of course rating",
        0.0, 0.5, 0.2, 0.05
    )

    st.caption(
        "Higher rating importance favours well-rated courses "
        "among similar ones."
    )


# ---------- Tabs ----------
tab1, tab2 = st.tabs([
    "📚 Similar to a course",
    "🔎 Search by topic"
])


# ---------- Tab 1: Similar course recommendations ----------
with tab1:

    idx = st.selectbox(
        "Choose a course",

        options=df.index.tolist(),

        format_func=lambda i: (
            f"{df.loc[i, 'course_title']} "
            f"({df.loc[i, 'course_organization']})"
        )
    )

    if st.button(
        "Recommend",
        key="btn_course",
        type="primary"
    ):

        # Retrieve precomputed similarities.
        scores = sim[idx]

        results = rank(
            scores,
            idx,
            n,
            difficulty,
            rating_weight
        )

        if results.empty:

            st.warning(
                "No similar courses found at that difficulty. "
                "Try selecting 'Any'."
            )

        else:

            st.subheader("Recommended courses")
            show_cards(results)


# ---------- Tab 2: Search by topic ----------
with tab2:

    query = st.text_input(
        "What do you want to learn?",
        placeholder="e.g. machine learning, cybersecurity, python"
    )

    if st.button(
        "Search",
        key="btn_topic",
        type="primary"
    ):

        if not query.strip():

            st.warning("Please type a topic first.")

        else:

            # Convert the user's search into a TF-IDF vector.
            vec = tfidf.transform([query.lower()])

            # Compare the query with every course.
            scores = cosine_similarity(vec, X).ravel()

            results = rank(
                scores,
                None,
                n,
                difficulty,
                rating_weight
            )

            if results.empty:

                st.warning(
                    "No matches found. Try different search words."
                )

            else:

                st.subheader(f"Courses for: {query}")
                show_cards(results)


# ---------- Footer ----------
st.caption(
    "Content-based recommender: similarity is calculated "
    "from course titles."
)
