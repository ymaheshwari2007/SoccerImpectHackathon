import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 — Data Exploration
    Understand what the IMPECT open data contains before building anything.

    **Kernel:** Select `../venv` (Python 3.13) as your kernel before running.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Matches & Squads
    The IMPECT open data lives on GitHub. We fetch two JSON files:
    - `matches_743.json` — all 306 Bundesliga 2023/24 matches
    - `squads_743.json` — all teams in the competition
    """)
    return


@app.cell
def _():
    import io
    import polars as pl
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

    response = requests.get(match_url)
    matches = (
        pl.read_json(io.StringIO(response.text))
        .unnest("matchDay")
        .rename({"iterationId": "competitionId", "id": "matchId"})
        .drop(["idMappings", "lastCalculationDate", "name", "available"])
        .with_columns([(pl.col("index") + 1).alias("matchDay")])
        .drop("index")
    )

    response = requests.get(squads_url)
    squads = pl.read_json(io.StringIO(response.text)).drop(
        ["type", "gender", "imageUrl", "idMappings", "access", "countryId"]
    )

    matches = (
        matches
        .join(squads.rename({"name": "homeTeam"}), left_on="homeSquadId", right_on="id", how="left")
        .join(squads.rename({"name": "awayTeam"}), left_on="awaySquadId", right_on="id", how="left")
    )

    print(f"Total matches: {matches.height}")
    print(f"Match days: {matches['matchDay'].min()} to {matches['matchDay'].max()}")
    matches.head(5)
    return COMPETITION_ID, matches, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. All Teams in the Dataset
    18 Bundesliga clubs. These are the exact name strings to use when filtering.
    """)
    return


@app.cell
def _(matches, pl):
    all_teams = (
        pl.concat([
            matches.select(pl.col("homeTeam").alias("team")),
            matches.select(pl.col("awayTeam").alias("team")),
        ])
        .unique()
        .sort("team")
    )

    print(f"Total teams: {all_teams.height}")
    for t in all_teams["team"].to_list():
        print(f"  {t}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Matches Per Team
    Every team should have 34 matches (17 home + 17 away).
    """)
    return


@app.cell
def _(matches, pl):
    home_counts = matches.group_by("homeTeam").agg(pl.len().alias("home_matches")).rename({"homeTeam": "team"})
    away_counts = matches.group_by("awayTeam").agg(pl.len().alias("away_matches")).rename({"awayTeam": "team"})

    match_counts = (
        home_counts
        .join(away_counts, on="team", how="inner")
        .with_columns((pl.col("home_matches") + pl.col("away_matches")).alias("total"))
        .sort("team")
    )

    match_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Load a Single Match — Full Event Stream
    Load one match to see the complete event structure. We'll use match 122838: Werder Bremen vs FC Bayern München (Matchday 1).
    """)
    return


@app.cell
def _(COMPETITION_ID):
    from kloppy import impect

    SAMPLE_MATCH_ID = 122838

    dataset = impect.load_open_data(
        match_id=SAMPLE_MATCH_ID,
        competition_id=COMPETITION_ID,
    )

    events_df = dataset.transform(to_coordinate_system="secondspectrum").to_df(engine="polars")

    print(f"Total events: {events_df.height}")
    print(f"Columns: {events_df.columns}")
    events_df
    return dataset, events_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Event Type Breakdown
    What types of events are in the data, and how many of each?
    """)
    return


@app.cell
def _(events_df, pl):
    event_breakdown = (
        events_df
        .group_by("event_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    print("Event types in one match:")
    event_breakdown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Team & Player Metadata
    From `dataset.metadata` we get team names, formations, and every player's name, position, and jersey number.
    """)
    return


@app.cell
def _(dataset):
    for _team in dataset.metadata.teams:
        print(f'\nTeam: {_team.name} (ID: {_team.team_id}) | Ground: {_team.ground}')
        starters = [_p for _p in _team.players if _p.starting]
        bench = [_p for _p in _team.players if not _p.starting]
        print(f'  Starters ({len(starters)}):')
        for _p in starters:
            print(f'    #{_p.jersey_no:>2}  {_p.name:<30}  {str(_p.position)}')
        print(f'  Bench ({len(bench)}):')
        for _p in bench:
            print(f'    #{_p.jersey_no:>2}  {_p.name:<30}  {str(_p.position)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Passes Deep Dive
    Passes are the core of our pass network. Let's look at the structure of pass events specifically — who passed to whom, from where, with what success rate.
    """)
    return


@app.cell
def _(events_df, pl):
    passes = events_df.filter(pl.col("event_type") == "PASS")

    print(f"Total passes: {passes.height}")
    print(f"Successful:   {passes.filter(pl.col('success') == True).height}")
    print(f"Failed:       {passes.filter(pl.col('success') == False).height}")

    passes.select([
        "player_id", "receiver_player_id",
        "coordinates_x", "coordinates_y",
        "end_coordinates_x", "end_coordinates_y",
        "success", "pass_type", "is_under_pressure"
    ]).head(10)
    return (passes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Receiver Coverage
    `receiver_player_id` is what lets us build the pass network (passer → receiver edge). Let's check how often it's populated.
    """)
    return


@app.cell
def _(passes, pl):
    total_passes     = passes.height
    with_receiver    = passes.filter(pl.col("receiver_player_id").is_not_null()).height
    successful_only  = passes.filter(pl.col("success") == True).height
    success_with_rcv = passes.filter((pl.col("success") == True) & (pl.col("receiver_player_id").is_not_null())).height

    print(f"All passes:                        {total_passes}")
    print(f"Passes WITH receiver_player_id:    {with_receiver} ({100*with_receiver/total_passes:.1f}%)")
    print(f"Successful passes:                 {successful_only}")
    print(f"Successful WITH receiver_player_id:{success_with_rcv} ({100*success_with_rcv/successful_only:.1f}%)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Passer → Receiver Pairs (proto pass network)
    This is the raw material for our graph. Each unique `(player_id, receiver_player_id)` pair becomes an edge.
    """)
    return


@app.cell
def _(dataset, passes, pl):
    # Focus on one team's successful passes
    home_team_id = str(dataset.metadata.teams[0].team_id)
    home_team_name = dataset.metadata.teams[0].name

    home_passes = passes.filter(
        (pl.col("team_id") == home_team_id) &
        (pl.col("success") == True) &
        (pl.col("receiver_player_id").is_not_null())
    )

    pass_pairs = (
        home_passes
        .group_by(["player_id", "receiver_player_id"])
        .agg(pl.len().alias("pass_count"))
        .sort("pass_count", descending=True)
    )

    print(f"Team: {home_team_name}")
    print(f"Unique passer→receiver pairs (edges): {pass_pairs.height}")
    pass_pairs.head(10)
    return home_passes, home_team_id, home_team_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Player Average Positions
    Node positions in the pass network = average x/y of all a player's pass events. This approximates where they operated on the pitch.
    """)
    return


@app.cell
def _(dataset, home_passes, pl):
    # Build a player_id → name lookup from metadata
    player_lookup = {}
    for _team in dataset.metadata.teams:
        for _p in _team.players:
            player_lookup[str(_p.player_id)] = _p.name
    avg_positions = home_passes.group_by('player_id').agg([pl.col('coordinates_x').mean().alias('avg_x'), pl.col('coordinates_y').mean().alias('avg_y'), pl.len().alias('total_passes')]).with_columns(pl.col('player_id').replace(player_lookup).alias('player_name')).sort('total_passes', descending=True)
    print(f'Players with pass data: {avg_positions.height}')
    avg_positions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Substitutions — Why They Matter
    For lineup analysis, substitutions break formation integrity. We'll only use events **before the first sub** when building match-level networks.
    """)
    return


@app.cell
def _(events_df, home_passes, home_team_id, home_team_name, pl):
    subs = events_df.filter(
        (pl.col("event_type") == "SUBSTITUTION") &
        (pl.col("team_id") == home_team_id)
    )

    print(f"Substitutions for {home_team_name}: {subs.height}")
    if subs.height > 0:
        first_sub_time = subs.select("timestamp").to_series()[0]
        print(f"First substitution at: {first_sub_time}")
        pre_sub_passes = home_passes.filter(pl.col("timestamp") < first_sub_time)
        print(f"Passes before first sub: {pre_sub_passes.height} / {home_passes.height} total")
    subs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. What We'll Use — Summary

    | Data | Where it comes from | Used for |
    |------|--------------------|---------|
    | `player_id`, `receiver_player_id` | pass events | graph edges |
    | `coordinates_x/y` | pass events | node positions |
    | `success == True` | pass events | filter to valid edges only |
    | `team_id` | pass events | separate home/away networks |
    | `timestamp` | all events | cut off at first substitution |
    | `player.name`, `player.position`, `player.jersey_no` | dataset.metadata | node labels |
    | `matchId` | matches DataFrame | loop over all season games |
    | `homeTeam`, `awayTeam` | matches DataFrame | filter by team name |

    Next step → `src/data_loader.py` to wrap all of this into reusable functions.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
