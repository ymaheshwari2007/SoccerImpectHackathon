import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 — Plays Builder
    Split a match into **plays** (possession sequences) and score each play by ball progression.

    A **play** begins when a team gains possession and ends when possession is lost.
    We start with one game — **FC Bayern München vs Borussia Dortmund** — then generalise.

    **Kernel:** Select `../venv` (Python 3.13) as your kernel before running.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Find the Bayern vs Dortmund Match
    Query the matches dataset for head-to-head games between the two clubs.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import requests
    from kloppy.utils import github_resolve_raw_data_url

    COMPETITION_ID = 743  # Bundesliga 2023/24

    match_url = github_resolve_raw_data_url(
        repository="ImpectAPI/open-data",
        branch="main",
        file="data/matches/matches_743.json",
    )
    squads_url = github_resolve_raw_data_url(
        repository="ImpectAPI/open-data",
        branch="main",
        file="data/squads/squads_743.json",
    )

    matches_raw = pd.DataFrame(requests.get(match_url).json())
    # Flatten the nested matchDay column
    match_day_df = pd.json_normalize(matches_raw["matchDay"])
    matches = matches_raw.drop(columns=["matchDay"]).join(match_day_df)
    matches = matches.rename(columns={"iterationId": "competitionId", "id": "matchId"})
    matches["matchDay"] = matches["index"] + 1
    matches = matches.drop(columns=["index"])

    squads = pd.DataFrame(requests.get(squads_url).json())
    squads = squads.drop(columns=["type", "gender", "imageUrl", "idMappings", "access", "countryId"], errors="ignore")

    matches = (
        matches
        .merge(squads.rename(columns={"name": "homeTeam"}), left_on="homeSquadId", right_on="id", how="left")
        .drop(columns=["id"])
        .merge(squads.rename(columns={"name": "awayTeam"}), left_on="awaySquadId", right_on="id", how="left")
        .drop(columns=["id"])
    )

    # Find Bayern vs Dortmund games (home or away)
    bayern_dortmund = matches[
        ((matches["homeTeam"] == "FC Bayern München") & (matches["awayTeam"] == "Borussia Dortmund")) |
        ((matches["homeTeam"] == "Borussia Dortmund") & (matches["awayTeam"] == "FC Bayern München"))
    ]

    print(f"Bayern vs Dortmund games found: {len(bayern_dortmund)}")
    bayern_dortmund[["matchId", "matchDay", "homeTeam", "awayTeam", "scheduledDate"]]
    return COMPETITION_ID, bayern_dortmund


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load the Match Event Stream
    Pick the first Bayern vs Dortmund game and load every event via Kloppy.
    """)
    return


@app.cell
def _(COMPETITION_ID, bayern_dortmund):
    from kloppy import impect

    # Pick the first match
    MATCH_ID = bayern_dortmund["matchId"].iloc[0]
    print(f"Loading match {MATCH_ID}...")

    dataset = impect.load_open_data(
        match_id=MATCH_ID,
        competition_id=COMPETITION_ID,
    )

    events_df = dataset.transform(to_coordinate_system="secondspectrum").to_df(engine="pandas")

    print(f"Total events: {len(events_df)}")
    print(f"Columns ({len(events_df.columns)}): {list(events_df.columns)}")
    events_df
    return (events_df,)


@app.cell
def _(events_df):
    x = events_df[events_df['event_type'] == 'GENERIC:NO_VIDEO'][['event_id']]
    events_df
    return


@app.cell
def _(events_df):
    # --- Build plays from possession changes ---
    df = events_df.copy()

    # Fill in dead ball moments (fouls, goals, etc.) with the team that had the ball before
    df['play_owner'] = df['ball_owning_team'].ffill()

    # New play starts whenever a different team gets the ball
    df['play_id'] = (df['play_owner'] != df['play_owner'].shift(1)).cumsum()

    # Find time gap between each event and the next one
    df['next_timestamp'] = df['timestamp'].shift(-1)
    df['gap_seconds'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds()

    # Remove plays where a NO_VIDEO event has a 3+ second gap (bad data)
    bad_plays = df[
        (df['event_type'] == 'GENERIC:NO_VIDEO') & 
        (df['gap_seconds'] > 3) &
        (df['play_id'] == df['play_id'].shift(-1))
    ]['play_id'].unique()

    df = df[~df['play_id'].isin(bad_plays)]
    df = df.drop(columns=['next_timestamp', 'gap_seconds'])

    # --- plays_df: one row per play (summary table) ---
    plays_df = df.groupby('play_id').agg(
        team_id=('play_owner', 'first'),
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last'),
        start_x=('coordinates_x', 'first'),
        start_y=('coordinates_y', 'first'),
        end_x=('coordinates_x', 'last'),
        end_y=('coordinates_y', 'last'),
        event_count=('event_id', 'count'),
        start_event=('event_type', 'first'),
        end_event=('event_type', 'last'),
        first_player=('player_id', 'first'),
        last_player=('player_id', 'last'),
    ).reset_index()

    # How far did the ball move during this play?
    plays_df['delta_x'] = plays_df['end_x'] - plays_df['start_x']
    plays_df['delta_y'] = plays_df['end_y'] - plays_df['start_y']

    # --- play_events_df: every event tagged with which play it belongs to ---
    play_events_df = df[['play_id', 'event_id', 'event_type', 'team_id', 
                          'player_id', 'receiver_player_id',
                          'coordinates_x', 'coordinates_y',
                          'end_coordinates_x', 'end_coordinates_y',
                          'timestamp', 'success', 'ball_owning_team']].copy()

    plays_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
