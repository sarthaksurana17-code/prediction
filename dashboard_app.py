"""
dashboard_app.py

A lightweight Streamlit dashboard for the World Cup prediction project.
This turns the project from "a script I ran locally" into something you can
share as a live link or screen-share in an interview -- worth more on a
resume than any bullet point.

Setup:
    pip install streamlit plotly
    streamlit run dashboard_app.py

Deploy for free (so you can put a live link on your resume):
    https://streamlit.io/cloud -- connect your GitHub repo, it deploys
    automatically from this file.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.elo_ratings import EloRatingSystem
from src.data_loader import DataLoader

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")

st.title("⚽ World Cup 2026 Prediction Model")
st.caption("Elo-based match predictions, backtested against 2014 / 2018 / 2022 results.")


@st.cache_resource
def load_elo():
    loader = DataLoader()
    matches = loader.get_all_data()["historical_matches"]
    elo = EloRatingSystem()
    elo.process_match_history(matches, competition_col="competition")
    return elo


elo = load_elo()

tab1, tab2, tab3 = st.tabs(["Match Predictor", "Team Rankings", "Model Accuracy"])

with tab1:
    col1, col2 = st.columns(2)
    teams = sorted(elo.ratings.keys())
    team_a = col1.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    team_b = col2.selectbox("Team B", teams, index=teams.index("Mexico") if "Mexico" in teams else 1)

    if st.button("Predict Match", type="primary"):
        p_a, p_draw, p_b = elo.win_probability(team_a, team_b)
        fig = px.bar(
            x=[team_a, "Draw", team_b], y=[p_a, p_draw, p_b],
            labels={"x": "Outcome", "y": "Probability"},
            color=[team_a, "Draw", team_b],
            title=f"{team_a} vs {team_b}",
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.metric(f"{team_a} Elo", f"{elo.get_rating(team_a):.0f}")
        st.metric(f"{team_b} Elo", f"{elo.get_rating(team_b):.0f}")

with tab2:
    top = elo.rankings(top_n=25)
    rank_df = pd.DataFrame(top, columns=["Team", "Elo Rating"])
    rank_df["Elo Rating"] = rank_df["Elo Rating"].round(0)
    rank_df.index = rank_df.index + 1
    st.dataframe(rank_df, use_container_width=True)

with tab3:
    st.write("Run `backtest.py` to generate these numbers, then hardcode or load them here.")
    st.info(
        "Suggested resume line once you have real numbers:\n\n"
        "\"Built and backtested an Elo-based match prediction model against "
        "3 World Cup tournaments (2014–2022), achieving X% outcome accuracy "
        "vs. a Y% naive baseline.\""
    )
