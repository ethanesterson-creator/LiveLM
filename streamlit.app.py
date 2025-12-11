import streamlit as st
import pandas as pd
from datetime import date, datetime
from pathlib import Path

# -----------------------------------------
# Leagues
# -----------------------------------------

LEAGUES = [
    {"name": "Sophomore League", "slug": "soph"},
    {"name": "Junior League", "slug": "junior"},
    {"name": "Senior League", "slug": "senior"},
]


def get_league_by_name(name: str):
    for lg in LEAGUES:
        if lg["name"] == name:
            return lg
    return LEAGUES[-1]  # default to Senior if something weird happens


def get_league_by_slug(slug: str):
    for lg in LEAGUES:
        if lg["slug"] == slug:
            return lg
    return LEAGUES[-1]


# -----------------------------------------
# Stat Categories by Sport
# -----------------------------------------

SPORT_STAT_CATEGORIES = {
    "Basketball": [
        ("basket_points", "Points"),
        ("basket_assists", "Assists"),
        ("basket_rebounds", "Rebounds"),
        ("basket_steals", "Steals"),
        ("basket_blocks", "Blocks"),
    ],
    "Softball": [
        ("soft_hits", "Hits"),
        ("soft_doubles", "Doubles"),
        ("soft_home_runs", "Home Runs"),
        ("soft_rbis", "RBIs"),
        ("soft_strikeouts", "Strikeouts"),
    ],
    "Kickball": [
        ("kick_runs", "Runs"),
        ("kick_rbis", "RBIs"),
    ],
    "Hockey": [
        ("hockey_goals", "Goals"),
        ("hockey_assists", "Assists"),
    ],
    "Soccer": [
        ("soccer_goals", "Goals"),
        ("soccer_assists", "Assists"),
    ],
    "Euro": [
        ("euro_goals", "Goals"),
        ("euro_assists", "Assists"),
    ],
    "Speedball": [
        ("speed_points", "Points"),
        ("speed_assists", "Assists"),
    ],
    "Flag Football": [
        ("ff_touchdowns", "Touchdowns"),
        ("ff_catches", "Catches"),
        ("ff_interceptions", "Interceptions"),
    ],
    "Other": [
        ("points", "Points"),
        ("assists", "Assists"),
    ],
}

DEFAULT_SPORTS = list(SPORT_STAT_CATEGORIES.keys())
LEVELS = ["A", "B", "C", "D"]

# -----------------------------------------
# League point values by sport/level
# -----------------------------------------

GAME_POINT_VALUES = {
    "Basketball": {"A": 15, "B": 10, "C": 7, "D": 5},
    "Softball": {"A": 15, "B": 10, "C": 7, "D": 5},
    "Kickball": {"A": 10, "B": 7, "C": 5, "D": 3},
    "Hockey": {"A": 15, "B": 10, "C": 7, "D": 5},
    "Soccer": {"A": 15, "B": 10, "C": 7, "D": 5},
    "Euro": {"A": 12, "B": 9, "C": 6, "D": 4},
    "Speedball": {"A": 12, "B": 9, "C": 6, "D": 4},
    "Flag Football": {"A": 15, "B": 10, "C": 7, "D": 5},
    "Other": {"A": 10, "B": 7, "C": 5, "D": 3},
}
DEFAULT_GAME_POINTS = {"A": 10, "B": 7, "C": 5, "D": 3}


def get_game_points(sport: str, level: str) -> int:
    sport_map = GAME_POINT_VALUES.get(sport, {})
    return sport_map.get(level, DEFAULT_GAME_POINTS.get(level, 0))


# -----------------------------------------
# Non-game point categories (from proposal)
# -----------------------------------------

NON_GAME_CATEGORIES = [
    "Friday Night Songs",
    "Cheering / Spirit",
    "Sportsmanship",
    "Bonding Event / Team Community",
    "Neb Event – Whist",
    "Neb Event – Gaga",
    "Neb Event – Tennis",
    "Neb Event – Pickleball",
    "Neb Event – Chess",
    "Neb Event – Tetherball",
    "Staff Game Performance",
    "Leadership – Captains",
    "Other (write-in)",
]

# -----------------------------------------
# Paths – always per league now
# -----------------------------------------

def get_paths(slug: str):
    base = slug.lower()
    return {
        "roster": Path(f"{base}_roster.csv"),
        "teams": Path(f"{base}_teams.csv"),
        "games": Path(f"{base}_games.csv"),
        "stats": Path(f"{base}_stats.csv"),
        "highlights": Path(f"{base}_highlights.csv"),
        "videos_dir": Path(f"{base}_highlight_videos"),
        "nongame": Path(f"{base}_nongame.csv"),
    }


# -----------------------------------------
# Dataframe helpers
# -----------------------------------------

def new_games_df():
    return pd.DataFrame(columns=[
        "game_id", "date", "sport", "level", "team1", "team2", "score1", "score2"
    ])


def new_stats_df():
    return pd.DataFrame(columns=[
        "game_id", "sport", "team_name", "player_id", "first_name",
        "last_name", "bunk", "stat_type", "value"
    ])


def new_highlights_df():
    return pd.DataFrame(columns=[
        "highlight_id", "date", "title", "description", "video_path",
        "sport", "level", "team1", "team2", "featured"
    ])


def new_nongame_df():
    return pd.DataFrame(columns=[
        "id", "date", "team_name", "category", "reason", "points"
    ])


def load_csv(path: Path, columns):
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            except Exception:
                pass
        return df
    else:
        return pd.DataFrame(columns=columns)


def save_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False)


def load_league_data(paths: dict):
    """Always load fresh from files for this league."""
    # Roster
    if paths["roster"].exists():
        roster = pd.read_csv(paths["roster"])
        roster.columns = (
            roster.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )
    else:
        roster = None

    # Teams: derive from roster if possible
    if roster is not None and "team_name" in roster.columns:
        teams = roster[["team_name"]].drop_duplicates().reset_index(drop=True)
    elif paths["teams"].exists():
        teams = pd.read_csv(paths["teams"])
    else:
        teams = None

    # Games
    games = load_csv(paths["games"], new_games_df().columns)
    if "sport" not in games.columns:
        games["sport"] = "Other"
    if "level" not in games.columns:
        games["level"] = "A"

    # Stats
    stats = load_csv(paths["stats"], new_stats_df().columns)
    if "sport" not in stats.columns:
        stats["sport"] = "Other"

    # Highlights
    highlights = load_csv(paths["highlights"], new_highlights_df().columns)
    if "featured" not in highlights.columns:
        highlights["featured"] = False
    if "video_path" not in highlights.columns:
        highlights["video_path"] = ""

    # Videos folder
    paths["videos_dir"].mkdir(exist_ok=True)

    return roster, teams, games, stats, highlights


def load_nongame(paths: dict):
    return load_csv(paths["nongame"], new_nongame_df().columns)


# -----------------------------------------
# Core calculations
# -----------------------------------------

def compute_standings(games: pd.DataFrame,
                      teams: pd.DataFrame,
                      nongame: pd.DataFrame | None = None,
                      rank_by: str = "total"):
    """
    rank_by: "game" or "total"
    """
    if teams is None or teams.empty:
        return pd.DataFrame()

    standings = pd.DataFrame({"team_name": teams["team_name"].unique()})
    standings["gp"] = 0
    standings["w"] = 0
    standings["l"] = 0
    standings["t"] = 0
    standings["game_points"] = 0
    standings["points_for"] = 0
    standings["points_against"] = 0
    standings["diff"] = 0

    if not games.empty:
        for _, g in games.iterrows():
            team1 = g["team1"]
            team2 = g["team2"]
            s1 = int(g["score1"])
            s2 = int(g["score2"])
            sport = g.get("sport", "Other")
            level = g.get("level", "A")
            win_points = get_game_points(sport, level)

            for team_name, scored, allowed in [(team1, s1, s2), (team2, s2, s1)]:
                idx = standings["team_name"] == team_name
                standings.loc[idx, "gp"] += 1
                standings.loc[idx, "points_for"] += scored
                standings.loc[idx, "points_against"] += allowed

            if s1 > s2:
                standings.loc[standings["team_name"] == team1, "w"] += 1
                standings.loc[standings["team_name"] == team2, "l"] += 1
                standings.loc[standings["team_name"] == team1, "game_points"] += win_points
            elif s2 > s1:
                standings.loc[standings["team_name"] == team2, "w"] += 1
                standings.loc[standings["team_name"] == team1, "l"] += 1
                standings.loc[standings["team_name"] == team2, "game_points"] += win_points
            else:
                half = win_points // 2
                standings.loc[standings["team_name"] == team1, "t"] += 1
                standings.loc[standings["team_name"] == team2, "t"] += 1
                standings.loc[standings["team_name"] == team1, "game_points"] += half
                standings.loc[standings["team_name"] == team2, "game_points"] += half

    # Non-game points
    standings["non_game_points"] = 0
    if nongame is not None and not nongame.empty:
        ng_sum = nongame.groupby("team_name")["points"].sum().reset_index()
        standings = standings.merge(ng_sum, on="team_name", how="left", suffixes=("", "_ng"))
        standings["non_game_points"] = standings["points"].fillna(0)
        standings.drop(columns=["points"], inplace=True)

    # Total
    standings["total_points"] = standings["game_points"] + standings["non_game_points"]

    standings["diff"] = standings["points_for"] - standings["points_against"]

    if rank_by == "game":
        sort_key = "game_points"
    else:
        sort_key = "total_points"

    standings = standings.sort_values(
        by=[sort_key, "diff", "points_for", "team_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # For backward-compat (if anything expects "pts")
    standings["pts"] = standings["total_points"]

    return standings


def compute_leaderboard(stats: pd.DataFrame, sport: str, stat_type: str):
    if stats.empty:
        return pd.DataFrame()

    df = stats[(stats["sport"] == sport) & (stats["stat_type"] == stat_type)]
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(
        ["player_id", "first_name", "last_name", "bunk", "team_name"],
        as_index=False
    )["value"].sum()

    agg = agg.sort_values(by="value", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    return agg[["rank", "first_name", "last_name", "bunk", "team_name", "value"]]


# -----------------------------------------
# League-specific pages (Setup, Scores, Standings, Leaderboards, Highlights, Non-Game Points)
# -----------------------------------------

def page_setup(league_slug: str, league_name: str):
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)

    st.header(f"{league_name} – Upload Roster")

    st.write(
        """
        Upload a single CSV with **all kids** in this league (sophomores OR juniors OR seniors),
        including which of the 4 league teams each kid is on.
        """
    )
    st.markdown("**Required columns:** `player_id, first_name, last_name, team_name, bunk`")

    example = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "first_name": ["Alex", "Ben", "Charlie", "Dylan"],
        "last_name": ["R", "S", "T", "U"],
        "team_name": ["Red", "Red", "Blue", "Blue"],
        "bunk": ["1", "1", "2", "2"],
    })
    st.caption("Example format:")
    st.dataframe(example, use_container_width=True)

    file = st.file_uploader(f"Upload {league_name} roster CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            df.columns = (
                df.columns
                .astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
                .str.lower()
            )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        required_cols = {"player_id", "first_name", "last_name", "team_name", "bunk"}
        if not required_cols.issubset(df.columns):
            st.error(
                "CSV must contain columns: player_id, first_name, last_name, team_name, bunk. "
                f"Current columns are: {list(df.columns)}"
            )
            return

        roster = df
        roster.to_csv(paths["roster"], index=False)

        teams = roster[["team_name"]].drop_duplicates().reset_index(drop=True)
        teams.to_csv(paths["teams"], index=False)

        st.success(f"{league_name} roster loaded and saved successfully!")

    if roster is not None:
        st.subheader(f"{league_name} – Current Roster")
        st.dataframe(roster, use_container_width=True)

        st.subheader(f"{league_name} – Teams")
        if teams is not None:
            st.dataframe(teams, use_container_width=True)


def page_enter_scores_and_stats(league_slug: str, league_name: str):
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)

    st.header(f"{league_name} – Enter Scores & Stats")

    if roster is None or teams is None:
        st.warning("You need to upload a roster first on the 'Setup' page for this league.")
        return

    teams_list = teams["team_name"].tolist()
    sports_list = DEFAULT_SPORTS

    # ----- Add new game -----
    st.subheader("Add New Game")

    col_date, col_sport, col_level = st.columns(3)
    with col_date:
        game_date = st.date_input("Game Date", value=date.today())
    with col_sport:
        sport = st.selectbox("Sport", sports_list, index=0)
    with col_level:
        level = st.selectbox("Level (A/B/C/D)", LEVELS, index=0)

    col_team1, col_team2 = st.columns(2)
    with col_team1:
        team1 = st.selectbox("Team 1", teams_list, key=f"{league_slug}_team1_select")
    with col_team2:
        team2 = st.selectbox("Team 2", teams_list, key=f"{league_slug}_team2_select")

    if team1 == team2:
        st.error("Team 1 and Team 2 must be different.")
    else:
        col_score1, col_score2 = st.columns(2)
        with col_score1:
            score1 = st.number_input(
                f"{team1} Score",
                min_value=0,
                step=1,
                value=0,
                key=f"{league_slug}_score1",
            )
        with col_score2:
            score2 = st.number_input(
                f"{team2} Score",
                min_value=0,
                step=1,
                value=0,
                key=f"{league_slug}_score2",
            )

        if st.button("Save Game", key=f"{league_slug}_save_game"):
            game_id = f"G{len(games) + 1}"
            new_game = pd.DataFrame([{
                "game_id": game_id,
                "date": pd.to_datetime(game_date).date(),
                "sport": sport,
                "level": level,
                "team1": team1,
                "team2": team2,
                "score1": score1,
                "score2": score2,
            }])
            games = pd.concat([games, new_game], ignore_index=True)
            save_csv(paths["games"], games)

            pts = get_game_points(sport, level)
            st.success(
                f"Saved game {game_id}: {sport} ({level}) – {team1} {score1}-{score2} {team2} "
                f"(win worth {pts} league pts)."
            )

    st.markdown("---")

    # ----- Enter / edit stats -----
    st.subheader("Enter / Edit Stats for an Existing Game")

    if games.empty:
        st.info("No games yet. Add a game above first.")
        return

    games_sorted = games.sort_values("date")
    game_options = {}
    for _, g in games_sorted.iterrows():
        d = g["date"]
        label = f"{g['game_id']} – {d} – {g['sport']} ({g['level']}) – {g['team1']} vs {g['team2']}"
        game_options[label] = g["game_id"]

    selected_label = st.selectbox(
        "Choose a game to enter stats for",
        list(game_options.keys()),
        key=f"{league_slug}_game_select_for_stats",
    )
    selected_game_id = game_options[selected_label]
    game_row = games[games["game_id"] == selected_game_id].iloc[0]

    game_sport = game_row["sport"]
    team1 = game_row["team1"]
    team2 = game_row["team2"]

    st.caption(f"Game: {selected_game_id} • {game_sport} • {team1} vs {team2}")

    categories = SPORT_STAT_CATEGORIES.get(game_sport, SPORT_STAT_CATEGORIES["Other"])

    home_roster = roster[roster["team_name"] == team1]
    away_roster = roster[roster["team_name"] == team2]

    existing_stats_game = stats[stats["game_id"] == selected_game_id]
    existing_lookup = {}
    for _, row in existing_stats_game.iterrows():
        key = (row["player_id"], row["stat_type"])
        existing_lookup[key] = row["value"]

    st.caption("Enter stat totals for THIS game only. The app will handle season totals.")

    # Stats for team1
    st.markdown(f"### Stats for {team1}")
    with st.expander(f"{team1} Players", expanded=True):
        for _, p in home_roster.iterrows():
            player_key_base = f"{league_slug}_{selected_game_id}_{team1}_{p['player_id']}"
            st.markdown(f"**{p['first_name']} {p['last_name']} (Bunk {p['bunk']})**")
            cols = st.columns(len(categories))
            for (stat_code, stat_label), col in zip(categories, cols):
                default_val = existing_lookup.get((p["player_id"], stat_code), 0)
                with col:
                    st.number_input(
                        stat_label,
                        min_value=0,
                        step=1,
                        value=int(default_val),
                        key=f"{player_key_base}_{stat_code}",
                    )

    # Stats for team2
    st.markdown(f"### Stats for {team2}")
    with st.expander(f"{team2} Players", expanded=True):
        for _, p in away_roster.iterrows():
            player_key_base = f"{league_slug}_{selected_game_id}_{team2}_{p['player_id']}"
            st.markdown(f"**{p['first_name']} {p['last_name']} (Bunk {p['bunk']})**")
            cols = st.columns(len(categories))
            for (stat_code, stat_label), col in zip(categories, cols):
                default_val = existing_lookup.get((p["player_id"], stat_code), 0)
                with col:
                    st.number_input(
                        stat_label,
                        min_value=0,
                        step=1,
                        value=int(default_val),
                        key=f"{player_key_base}_{stat_code}",
                    )

    if st.button("Save Stats for This Game", key=f"{league_slug}_save_stats"):
        # Remove old stats for this game
        stats = stats[stats["game_id"] != selected_game_id]

        new_stats_rows = []

        def collect_stats_for_team(team_name, team_roster):
            for _, p in team_roster.iterrows():
                player_key_base = f"{league_slug}_{selected_game_id}_{team_name}_{p['player_id']}"
                for (stat_code, _) in categories:
                    widget_key = f"{player_key_base}_{stat_code}"
                    val = st.session_state.get(widget_key, 0)
                    if val and int(val) > 0:
                        new_stats_rows.append({
                            "game_id": selected_game_id,
                            "sport": game_sport,
                            "team_name": team_name,
                            "player_id": p["player_id"],
                            "first_name": p["first_name"],
                            "last_name": p["last_name"],
                            "bunk": p["bunk"],
                            "stat_type": stat_code,
                            "value": int(val),
                        })

        collect_stats_for_team(team1, home_roster)
        collect_stats_for_team(team2, away_roster)

        if new_stats_rows:
            new_stats_df = pd.DataFrame(new_stats_rows)
            stats = pd.concat([stats, new_stats_df], ignore_index=True)
            save_csv(paths["stats"], stats)

        st.success(f"Saved stats for game {selected_game_id}.")

    st.subheader(f"{league_name} – Games Entered So Far")
    if games.empty:
        st.info("No games yet.")
    else:
        st.dataframe(games, use_container_width=True)


def page_standings(league_slug: str, league_name: str):
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)
    nongame = load_nongame(paths)

    st.header(f"{league_name} – Standings")

    if roster is None or teams is None:
        st.warning("You need to upload a roster first on the 'Setup' page for this league.")
        return

    standings = compute_standings(games, teams, nongame, rank_by="total")
    if standings.empty:
        st.info("No games yet. Enter some results first.")
        return

    display = standings.copy()
    display.insert(0, "Rank", range(1, len(display) + 1))
    display = display.rename(columns={
        "team_name": "Team",
        "gp": "GP",
        "w": "W",
        "l": "L",
        "t": "T",
        "game_points": "Game Pts",
        "non_game_points": "Non-Game Pts",
        "total_points": "Total Pts",
        "points_for": "PF",
        "points_against": "PA",
        "diff": "Diff",
    })

    st.dataframe(display, use_container_width=True)


def page_leaderboards(league_slug: str, league_name: str):
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)

    st.header(f"{league_name} – Leaderboards")

    if roster is None or teams is None:
        st.warning("You need to upload a roster first on the 'Setup' page for this league.")
        return

    if stats.empty:
        st.info("No stats yet. Enter some game stats first.")
        return

    sports_with_stats = sorted(stats["sport"].unique().tolist())
    selected_sport = st.selectbox("Sport", sports_with_stats, key=f"{league_slug}_lb_sport")

    categories = SPORT_STAT_CATEGORIES.get(selected_sport, SPORT_STAT_CATEGORIES["Other"])
    label_to_code = {label: code for code, label in categories}
    stat_label = st.selectbox("Stat Category", list(label_to_code.keys()),
                              key=f"{league_slug}_lb_stat")
    stat_code = label_to_code[stat_label]

    lb = compute_leaderboard(stats, selected_sport, stat_code)
    if lb.empty:
        st.info(f"No stats recorded yet for {selected_sport} – {stat_label}.")
        return

    display = lb.rename(columns={
        "first_name": "First",
        "last_name": "Last",
        "bunk": "Bunk",
        "team_name": "Team",
        "value": stat_label,
    })

    top_row = display.iloc[0]
    st.success(
        f"🏆 Current leader in {selected_sport} – {stat_label}: "
        f"{top_row['First']} {top_row['Last']} ({top_row['Team']}), {stat_label}: {top_row[stat_label]}"
    )

    st.subheader(f"{league_name} – {selected_sport} – {stat_label} Leaders")
    st.dataframe(display, use_container_width=True)


def page_highlights(league_slug: str, league_name: str):
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)

    st.header(f"{league_name} – Highlights & Videos")

    if teams is None:
        teams_list = []
    else:
        teams_list = teams["team_name"].tolist()

    st.write("Upload highlight videos from that day for the mess hall monitor.")

    col_form, col_list = st.columns([2, 3])

    # Add highlight form
    with col_form:
        st.subheader("Add New Highlight")
        with st.form(f"{league_slug}_add_highlight_form", clear_on_submit=True):
            h_date = st.date_input("Date", value=date.today())
            title = st.text_input("Title (e.g., 'A Basketball: Red vs Blue')")
            video_file = st.file_uploader(
                "Upload highlight video",
                type=["mp4", "mov", "avi", "mkv"],
            )
            description = st.text_area("Description (optional)", height=80)

            sport = st.selectbox("Sport", DEFAULT_SPORTS + ["Other"])
            level = st.selectbox("Level", LEVELS + ["N/A"], index=0)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                team1 = st.selectbox("Team 1 (optional)", [""] + teams_list)
            with col_t2:
                team2 = st.selectbox("Team 2 (optional)", [""] + teams_list)

            featured = st.checkbox("Feature this on today's display board", value=True)

            submitted = st.form_submit_button("Save Highlight")

            if submitted:
                if not title.strip():
                    st.error("Please enter a title.")
                elif video_file is None:
                    st.error("Please upload a video file.")
                else:
                    next_id = 1 if highlights.empty else int(highlights["highlight_id"].max()) + 1

                    videos_dir = paths["videos_dir"]
                    videos_dir.mkdir(exist_ok=True)
                    safe_name = f"highlight_{next_id}_{video_file.name}"
                    video_path = videos_dir / safe_name
                    with open(video_path, "wb") as f:
                        f.write(video_file.getbuffer())

                    new_row = {
                        "highlight_id": next_id,
                        "date": h_date,
                        "title": title.strip(),
                        "description": description.strip(),
                        "video_path": str(video_path),
                        "sport": sport,
                        "level": level,
                        "team1": team1 or "",
                        "team2": team2 or "",
                        "featured": bool(featured),
                    }

                    highlights = pd.concat(
                        [highlights, pd.DataFrame([new_row])], ignore_index=True
                    )
                    save_csv(paths["highlights"], highlights)
                    st.success("Highlight saved!")

    # Highlight list & preview
    with col_list:
        st.subheader("Existing Highlights")

        if highlights.empty:
            st.info("No highlights yet.")
        else:
            display = highlights.copy()
            st.dataframe(
                display[["highlight_id", "date", "title", "sport", "level", "featured"]],
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("Preview a Highlight")
            ids = display["highlight_id"].tolist()
            id_to_title = {int(r["highlight_id"]): r["title"] for _, r in display.iterrows()}
            if ids:
                selected_id = st.selectbox(
                    "Choose a highlight to preview",
                    ids,
                    format_func=lambda x: f"{x} – {id_to_title.get(x, '')}",
                    key=f"{league_slug}_preview_highlight",
                )
                row = display[display["highlight_id"] == selected_id].iloc[0]
                st.markdown(f"**{row['title']}**")
                if isinstance(row["date"], (datetime, date)):
                    st.caption(str(row["date"]))
                elif isinstance(row["date"], str):
                    st.caption(row["date"])
                if row.get("description"):
                    st.write(row["description"])
                vp = row.get("video_path", "")
                if vp and Path(vp).exists():
                    st.video(vp)
                else:
                    st.warning("Video file not found. It may have been moved or deleted.")


def page_non_game_points(league_slug: str, league_name: str):
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)
    nongame = load_nongame(paths)

    st.header(f"{league_name} – Non-Game Points")

    if teams is None:
        st.warning("You need a roster/teams loaded on the Setup page before adding non-game points.")
        return

    teams_list = teams["team_name"].tolist()

    st.write(
        """
        Use this to award points for **non-game things** from the Crest League proposal:
        Friday night songs, cheering, sportsmanship, Neb events, staff games, leadership, etc.
        """
    )

    col_form, col_list = st.columns([2, 3])

    with col_form:
        st.subheader("Add Non-Game Points")
        with st.form(f"{league_slug}_nongame_form", clear_on_submit=True):
            ng_date = st.date_input("Date", value=date.today())
            team_name = st.selectbox("Team", teams_list)
            category = st.selectbox("Category", NON_GAME_CATEGORIES)

            reason = ""
            if category == "Other (write-in)":
                reason = st.text_input("Describe the reason for these points")
            else:
                reason = st.text_input(
                    "Optional note (e.g., 'Week 2 Friday Night Songs')",
                    value="",
                )

            points = st.number_input(
                "Points to award",
                min_value=-1000,
                max_value=1000,
                value=5,
                step=1,
            )

            submitted = st.form_submit_button("Add Points")
            if submitted:
                if not team_name:
                    st.error("Please choose a team.")
                elif category == "Other (write-in)" and not reason.strip():
                    st.error("Please describe the reason for 'Other (write-in)'.")
                else:
                    if nongame.empty:
                        next_id = 1
                    else:
                        next_id = int(nongame["id"].max()) + 1

                    new_row = {
                        "id": next_id,
                        "date": ng_date,
                        "team_name": team_name,
                        "category": category,
                        "reason": reason.strip(),
                        "points": int(points),
                    }
                    nongame = pd.concat(
                        [nongame, pd.DataFrame([new_row])],
                        ignore_index=True,
                    )
                    save_csv(paths["nongame"], nongame)
                    st.success(f"Awarded {int(points)} non-game points to {team_name}.")

    with col_list:
        st.subheader("Non-Game Points History")
        if nongame.empty:
            st.info("No non-game points recorded yet.")
        else:
            display = nongame.sort_values("date", ascending=False).copy()
            display = display.rename(columns={
                "date": "Date",
                "team_name": "Team",
                "category": "Category",
                "reason": "Reason",
                "points": "Points",
            })
            st.dataframe(display, use_container_width=True)


# -----------------------------------------
# GLOBAL DISPLAY BOARD
# -----------------------------------------

def render_display_view(display_type: str, league_slug: str):
    """Render the clean content-only view for the TV."""
    league = get_league_by_slug(league_slug)
    league_name = league["name"]
    paths = get_paths(league_slug)
    roster, teams, games, stats, highlights = load_league_data(paths)
    nongame = load_nongame(paths)

    if display_type == "Standings":
        if teams is None:
            st.info(f"No roster/teams yet for {league_name}.")
            return

        basis = st.session_state.get("display_standings_basis", "Game + Non-Game (Total)")
        rank_by = "game" if basis == "Game Points Only" else "total"

        standings = compute_standings(games, teams, nongame, rank_by=rank_by)
        if standings.empty:
            st.info("No games yet.")
            return

        display = standings.copy()
        display.insert(0, "Rank", range(1, len(display) + 1))
        display = display.rename(columns={
            "team_name": "Team",
            "gp": "GP",
            "w": "W",
            "l": "L",
            "t": "T",
            "game_points": "Game Pts",
            "non_game_points": "Non-Game Pts",
            "total_points": "Total Pts",
            "points_for": "PF",
            "points_against": "PA",
            "diff": "Diff",
        })
        st.markdown(f"## {league_name} – Standings ({basis})")
        st.dataframe(display, use_container_width=True)

    elif display_type == "Stat Leaders":
        if stats.empty:
            st.info("No stats yet.")
            return

        sport = st.session_state.get("display_stat_sport")
        stat_code = st.session_state.get("display_stat_code")
        stat_label = st.session_state.get("display_stat_label")

        if not sport or not stat_code or not stat_label:
            st.info("Use Control Panel to choose sport and stat first.")
            return

        lb = compute_leaderboard(stats, sport, stat_code)
        if lb.empty:
            st.info(f"No stats yet for {sport} – {stat_label}.")
            return

        top_n = lb.head(10)
        display_lb = top_n.rename(columns={
            "first_name": "First",
            "last_name": "Last",
            "bunk": "Bunk",
            "team_name": "Team",
            "value": stat_label,
        })
        st.markdown(f"## {league_name} – {sport} – {stat_label} Leaders")
        st.dataframe(display_lb, use_container_width=True)

    elif display_type == "Highlights Reel":
        if highlights.empty:
            st.info("No highlights yet.")
            return

        today_str = date.today().isoformat()

        def is_today(val):
            if isinstance(val, str):
                return val.startswith(today_str)
            if isinstance(val, (datetime, date)):
                return val == date.today()
            return False

        today_highlights = highlights[
            highlights["date"].apply(is_today) | highlights["featured"].astype(bool)
        ]
        if today_highlights.empty:
            st.info("No highlights marked for today yet.")
            return

        st.markdown(f"## {league_name} – Highlights Reel")
        for _, row in today_highlights.sort_values("date").iterrows():
            st.markdown(f"**{row['title']}** ({row['sport']} {row['level']})")
            if row.get("description"):
                st.write(row["description"])
            vp = row.get("video_path", "")
            if vp and Path(vp).exists():
                st.video(vp)
            else:
                st.warning("Video file not found.")
            st.markdown("---")


def page_display_board_global():
    st.header("Mess Hall Display Board")

    # Defaults for display config
    if "display_league_slug" not in st.session_state:
        st.session_state.display_league_slug = "senior"
    if "display_type" not in st.session_state:
        st.session_state.display_type = "Standings"
    if "display_standings_basis" not in st.session_state:
        st.session_state.display_standings_basis = "Game + Non-Game (Total)"

    view_mode = st.radio(
        "View Mode",
        ["Control Panel", "Screen View"],
        key="display_view_mode",
    )

    if view_mode == "Control Panel":
        st.subheader("Control Panel (set up what the TV should show)")

        # League
        league_slugs = [lg["slug"] for lg in LEAGUES]
        league_names = [lg["name"] for lg in LEAGUES]
        current_slug = st.session_state.display_league_slug
        default_index = league_slugs.index(current_slug) if current_slug in league_slugs else 0

        league_name_choice = st.selectbox(
            "Which league do you want to display?",
            league_names,
            index=default_index,
        )
        selected_league = get_league_by_name(league_name_choice)
        st.session_state.display_league_slug = selected_league["slug"]

        # Display type
        display_type = st.selectbox(
            "What do you want to show on the screen?",
            ["Standings", "Stat Leaders", "Highlights Reel"],
            index=["Standings", "Stat Leaders", "Highlights Reel"].index(
                st.session_state.display_type
            ),
        )
        st.session_state.display_type = display_type

        # Extra options for Standings
        if display_type == "Standings":
            basis = st.selectbox(
                "Standings based on:",
                ["Game Points Only", "Game + Non-Game (Total)"],
                index=["Game Points Only", "Game + Non-Game (Total)"].index(
                    st.session_state.display_standings_basis
                ),
            )
            st.session_state.display_standings_basis = basis

        # Extra options for Stat Leaders
        if display_type == "Stat Leaders":
            paths = get_paths(selected_league["slug"])
            roster, teams, games, stats, highlights = load_league_data(paths)
            if stats.empty:
                st.info("No stats yet in this league.")
            else:
                sports_with_stats = sorted(stats["sport"].unique().tolist())
                sport = st.selectbox(
                    "Sport",
                    sports_with_stats,
                    key="display_stat_sport_select",
                )
                categories = SPORT_STAT_CATEGORIES.get(sport, SPORT_STAT_CATEGORIES["Other"])
                label_to_code = {label: code for code, label in categories}
                stat_label = st.selectbox(
                    "Stat",
                    list(label_to_code.keys()),
                    key="display_stat_label_select",
                )
                stat_code = label_to_code[stat_label]

                # Save config to session_state
                st.session_state.display_stat_sport = sport
                st.session_state.display_stat_code = stat_code
                st.session_state.display_stat_label = stat_label

        st.markdown("---")
        st.subheader("Preview")
        render_display_view(st.session_state.display_type, st.session_state.display_league_slug)
        st.caption("When you're happy, switch 'View Mode' to **Screen View** and put the browser in full-screen on the TV.")

    else:  # Screen View
        # Only the content, no controls (aside from sidebar)
        render_display_view(st.session_state.display_type, st.session_state.display_league_slug)


# -----------------------------------------
# GLOBAL ADMIN / CLEAR PAGE
# -----------------------------------------

def page_admin_global():
    st.header("Admin / Clear Data – All Leagues")

    # ---------------- PASSWORD GATE ----------------
    # Require password Hyaffa26 before showing any admin controls
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.info("Admin access is password protected.")
        pw = st.text_input("Enter admin password", type="password", key="admin_pw_input")
        col_pw_btn, col_pw_spacer = st.columns([1, 3])
        with col_pw_btn:
            if st.button("Unlock Admin", key="admin_pw_btn"):
                if pw == "Hyaffa26":
                    st.session_state.admin_authenticated = True
                    st.success("Access granted.")
                    # Immediately rerun so we skip the password gate on this run
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        # Stop rendering the rest of the page until password is correct
        st.stop()

    # Optional: show a small logout button
    with st.expander("Admin Session", expanded=False):
        st.caption("You are logged in to the admin area.")
        if st.button("Lock Admin Area", key="admin_logout_btn"):
            st.session_state.admin_authenticated = False
            st.success("Admin area locked again.")
            # Immediately rerun so we go back to the password screen
            st.rerun()

    # ---------------- ORIGINAL ADMIN CONTENT ----------------

    # Summary table
    summary_rows = []
    for lg in LEAGUES:
        slug = lg["slug"]
        name = lg["name"]
        paths = get_paths(slug)
        roster, teams, games, stats, highlights = load_league_data(paths)
        nongame = load_nongame(paths)
        summary_rows.append({
            "League": name,
            "Roster rows": 0 if roster is None else len(roster),
            "Teams": 0 if teams is None else len(teams),
            "Games": len(games),
            "Stat entries": len(stats),
            "Highlights": len(highlights),
            "Non-Game entries": len(nongame),
        })
    st.subheader("Current Data Overview")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown("---")

    # ----- Clear rosters -----
    st.subheader("Clear Rosters")

    roster_target = st.selectbox(
        "Which rosters do you want to clear?",
        ["All Leagues"] + [lg["name"] for lg in LEAGUES],
        key="admin_roster_target",
    )
    if st.button("Clear Selected Roster(s)", key="admin_clear_rosters_btn"):
        if roster_target == "All Leagues":
            targets = LEAGUES
        else:
            targets = [lg for lg in LEAGUES if lg["name"] == roster_target]

        for lg in targets:
            paths = get_paths(lg["slug"])
            for p in [paths["roster"], paths["teams"]]:
                if p.exists():
                    p.unlink()

        st.success(f"Cleared roster(s) for: {', '.join([t['name'] for t in targets])}")

    st.markdown("---")

    # ----- Delete specific games for one league -----
    st.subheader("Delete Specific Games (and Their Stats)")

    league_for_games = st.selectbox(
        "Choose league",
        [lg["name"] for lg in LEAGUES],
        key="admin_specific_games_league",
    )
    league_obj = get_league_by_name(league_for_games)
    paths = get_paths(league_obj["slug"])
    roster, teams, games, stats, highlights = load_league_data(paths)

    if games.empty:
        st.info(f"No games stored for {league_for_games}.")
    else:
        games_sorted = games.sort_values("date")
        labels = []
        ids = []
        for _, g in games_sorted.iterrows():
            d = g["date"]
            label = f"{g['game_id']} – {d} – {g['sport']} ({g['level']}) – {g['team1']} vs {g['team2']}"
            labels.append(label)
            ids.append(g["game_id"])

        selected_labels = st.multiselect(
            f"Select games to delete from {league_for_games}",
            labels,
            key="admin_games_multiselect",
        )
        label_to_id = dict(zip(labels, ids))
        selected_ids = [label_to_id[l] for l in selected_labels]

        if selected_ids and st.button("Delete Selected Games", key="admin_delete_selected_games_btn"):
            games = games[~games["game_id"].isin(selected_ids)]
            stats = stats[~stats["game_id"].isin(selected_ids)]
            save_csv(paths["games"], games)
            save_csv(paths["stats"], stats)
            st.success(f"Deleted {len(selected_ids)} game(s) and their stats from {league_for_games}.")

    st.markdown("---")

    # ----- Delete ALL games & stats -----
    st.subheader("Delete ALL Games & Stats")

    games_target = st.selectbox(
        "Delete games & stats for:",
        ["All Leagues"] + [lg["name"] for lg in LEAGUES],
        key="admin_games_target",
    )
    confirm_all_games = st.checkbox(
        "I understand this will permanently delete games & stats.",
        key="admin_confirm_all_games",
    )
    if confirm_all_games and st.button("Delete ALL Games & Stats", key="admin_delete_all_games_btn"):
        if games_target == "All Leagues":
            targets = LEAGUES
        else:
            targets = [lg for lg in LEAGUES if lg["name"] == games_target]

        for lg in targets:
            paths = get_paths(lg["slug"])
            games = new_games_df()
            stats = new_stats_df()
            save_csv(paths["games"], games)
            save_csv(paths["stats"], stats)

        st.success(f"Deleted all games & stats for: {', '.join([t['name'] for t in targets])}")

    st.markdown("---")

    # ----- Delete ALL Highlights -----
    st.subheader("Delete ALL Highlights")

    hl_target = st.selectbox(
        "Delete highlights for:",
        ["All Leagues"] + [lg["name"] for lg in LEAGUES],
        key="admin_hl_target",
    )
    confirm_hl = st.checkbox(
        "I understand this will permanently delete highlights and videos.",
        key="admin_confirm_hl",
    )
    if confirm_hl and st.button("Delete ALL Highlights", key="admin_delete_all_hl_btn"):
        if hl_target == "All Leagues":
            targets = LEAGUES
        else:
            targets = [lg for lg in LEAGUES if lg["name"] == hl_target]

        for lg in targets:
            paths = get_paths(lg["slug"])
            highlights = new_highlights_df()
            save_csv(paths["highlights"], highlights)
            videos_dir = paths["videos_dir"]
            if videos_dir.exists():
                for p in videos_dir.iterdir():
                    if p.is_file():
                        p.unlink()

        st.success(f"Deleted all highlights for: {', '.join([t['name'] for t in targets])}")

    st.markdown("---")

    # ----- Delete ALL Non-Game Points -----
    st.subheader("Delete ALL Non-Game Points")

    ng_target = st.selectbox(
        "Delete non-game points for:",
        ["All Leagues"] + [lg["name"] for lg in LEAGUES],
        key="admin_ng_target",
    )
    confirm_ng = st.checkbox(
        "I understand this will permanently delete all non-game points.",
        key="admin_confirm_ng",
    )
    if confirm_ng and st.button("Delete ALL Non-Game Points", key="admin_delete_all_ng_btn"):
        if ng_target == "All Leagues":
            targets = LEAGUES
        else:
            targets = [lg for lg in LEAGUES if lg["name"] == ng_target]

        for lg in targets:
            paths = get_paths(lg["slug"])
            nongame = new_nongame_df()
            save_csv(paths["nongame"], nongame)

        st.success(f"Deleted all non-game points for: {', '.join([t['name'] for t in targets])}")

    st.markdown("---")

    # ----- Full reset -----
    st.subheader("FULL RESET – All Data for All Leagues")
    st.error(
        "This will completely reset ALL leagues. "
        "You will need to upload new rosters and re-enter all games, stats, highlights, and non-game points."
    )
    confirm_everything = st.checkbox(
        "I REALLY understand, delete EVERYTHING for all leagues.",
        key="admin_confirm_full_reset",
    )
    if confirm_everything and st.button("Full Reset: Clear All Data for ALL Leagues", key="admin_full_reset_btn"):
        for lg in LEAGUES:
            paths = get_paths(lg["slug"])
            for p in [paths["roster"], paths["teams"], paths["games"], paths["stats"], paths["highlights"], paths["nongame"]]:
                if p.exists():
                    p.unlink()

            videos_dir = paths["videos_dir"]
            if videos_dir.exists():
                for p in videos_dir.iterdir():
                    if p.is_file():
                        p.unlink()

        st.success("All data cleared for ALL leagues. Go to Setup to upload fresh rosters.")



# -----------------------------------------
# Main
# -----------------------------------------

def main():
    st.set_page_config(page_title="Crest League Manager", layout="wide")

    st.sidebar.title("Crest League Manager")

    # League selector in sidebar for normal league work
    league_names = [lg["name"] for lg in LEAGUES]
    default_index = next(i for i, lg in enumerate(LEAGUES) if lg["slug"] == "senior")
    selected_league_name = st.sidebar.selectbox("League (for Setup/Stats)", league_names, index=default_index)
    league = get_league_by_name(selected_league_name)
    league_slug = league["slug"]
    league_name = league["name"]

    # Bauercrest logo
    logo_path = Path("logo-header-2.png")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_column_width=True)

    st.sidebar.caption(f"Managing league data for: **{league_name}**")

    page = st.sidebar.radio(
        "Go to",
        [
            "Setup",
            "Enter Scores & Stats",
            "Standings",
            "Leaderboards",
            "Highlights",
            "Non-Game Points",
            "Display Board",
            "Admin / Clear Data",
        ],
        key=f"{league_slug}_page_radio",
    )

    if page == "Setup":
        page_setup(league_slug, league_name)
    elif page == "Enter Scores & Stats":
        page_enter_scores_and_stats(league_slug, league_name)
    elif page == "Standings":
        page_standings(league_slug, league_name)
    elif page == "Leaderboards":
        page_leaderboards(league_slug, league_name)
    elif page == "Highlights":
        page_highlights(league_slug, league_name)
    elif page == "Non-Game Points":
        page_non_game_points(league_slug, league_name)
    elif page == "Display Board":
        page_display_board_global()
    elif page == "Admin / Clear Data":
        page_admin_global()


if __name__ == "__main__":
    main()

