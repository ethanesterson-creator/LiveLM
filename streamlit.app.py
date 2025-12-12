import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# --------------------------------------------------
# CONFIG & STYLING
# --------------------------------------------------

st.set_page_config(
    page_title="Crest League Manager v2",
    layout="wide",
)

# Simple CSS to make scoreboard/timer big and bold
st.markdown(
    """
    <style>
    .clm-scoreboard {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .clm-team-name {
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .clm-timer {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .clm-section-title {
        font-size: 20px;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .clm-flash {
        color: green;
        font-size: 12px;
        margin-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LEAGUES = {
    "Sophomore League": {"key": "soph", "weight": 1.0},
    "Junior League": {"key": "junior", "weight": 1.3},
    "Senior League": {"key": "senior", "weight": 1.6},
}

ADMIN_PASSWORD = "Hyaffa26"

NON_GAME_CATEGORIES = [
    "League Spirit",
    "Sportsmanship",
    "Cleanup / Organization",
    "Participation / Effort",
    "Other",
]

# Sport-specific timer presets (label, minutes)
SPORT_TIMER_PRESETS = {
    "Basketball": [
        ("No timer", 0),
        ("15:00 – per half (running)", 15),
        ("20:00 – per half (running)", 20),
        ("15:00 – per half (stopped)", 15),
        ("20:00 – per half (stopped)", 20),
        ("30:00 – full game (running)", 30),
    ],
    "Softball": [
        ("No timer (innings)", 0),
        ("60:00 – time limit", 60),
        ("75:00 – time limit", 75),
    ],
    "Hockey": [
        ("No timer", 0),
        ("12:00 – per period", 12),
        ("15:00 – per period", 15),
        ("45:00 – full game", 45),
    ],
    "Soccer": [
        ("No timer", 0),
        ("25:00 – per half (running)", 25),
        ("30:00 – per half (running)", 30),
        ("50:00 – full game", 50),
    ],
    "Flag Football": [
        ("No timer", 0),
        ("20:00 – per half (running)", 20),
        ("40:00 – full game", 40),
    ],
    "Kickball": [
        ("No timer (innings)", 0),
        ("45:00 – time limit", 45),
        ("60:00 – time limit", 60),
    ],
    "Euro": [
        ("No timer", 0),
        ("20:00 – per half (running)", 20),
        ("40:00 – full game", 40),
    ],
    "Speedball": [
        ("No timer", 0),
        ("20:00 – per half (running)", 20),
        ("40:00 – full game", 40),
    ],
}
DEFAULT_TIMER_PRESETS = [("No timer", 0), ("30:00 – running clock", 30)]

# Sport-specific stat types
SPORT_STATS = {
    "Basketball": ["points", "assists", "rebounds", "steals", "blocks"],
    "Softball": ["hits", "doubles", "triples", "home_runs", "rbis"],
    "Hockey": ["goals", "assists", "saves"],
    "Soccer": ["goals", "assists", "saves"],
    "Flag Football": ["tds", "catches", "interceptions", "flags_pulled"],
    "Kickball": ["runs", "hits", "rbi"],
    "Euro": ["goals", "assists", "saves"],
    "Speedball": ["goals", "assists", "steals"],
}
SPORT_OPTIONS = list(SPORT_STATS.keys())

# Labels for stat buttons
STAT_LABELS = {
    "points": "PTS",
    "assists": "AST",
    "rebounds": "REB",
    "steals": "STL",
    "blocks": "BLK",
    "hits": "H",
    "doubles": "2B",
    "triples": "3B",
    "home_runs": "HR",
    "rbis": "RBI",
    "goals": "G",
    "saves": "SV",
    "tds": "TD",
    "catches": "REC",
    "interceptions": "INT",
    "flags_pulled": "FLAG",
    "runs": "R",
    "rbi": "RBI",
}

# Base game point values by league_key -> sport -> level
# Senior A Softball = 50; everything else scaled reasonably.
GAME_POINT_VALUES = {
    "senior": {
        "Softball": {"A": 50, "B": 45, "C": 40, "D": 35},
        "Flag Football": {"A": 40, "B": 35, "C": 30, "D": 25},
        "Basketball": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Hockey": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Soccer": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Euro": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Speedball": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Kickball": {"A": 25, "B": 20, "C": 15, "D": 10},
    },
    "junior": {
        "Softball": {"A": 45, "B": 40, "C": 35, "D": 30},
        "Flag Football": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Basketball": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Hockey": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Soccer": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Euro": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Speedball": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Kickball": {"A": 20, "B": 15, "C": 10, "D": 5},
    },
    "soph": {
        "Softball": {"A": 35, "B": 30, "C": 25, "D": 20},  # per your note
        "Flag Football": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Basketball": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Hockey": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Soccer": {"A": 20, "B": 15, "C": 10, "D": 5},
        "Euro": {"A": 20, "B": 15, "C": 10, "D": 5},
        "Speedball": {"A": 20, "B": 15, "C": 10, "D": 5},
        "Kickball": {"A": 15, "B": 10, "C": 5, "D": 5},
    },
}

# --------------------------------------------------
# HELPERS: PATHS / IO
# --------------------------------------------------


def league_key_from_name(league_name: str) -> str:
    return LEAGUES[league_name]["key"]


def league_name_from_key(key: str) -> str:
    for name, cfg in LEAGUES.items():
        if cfg["key"] == key:
            return name
    return key


def league_paths(league_key: str) -> dict:
    """Return all CSV paths for a given league key."""
    return {
        "roster": DATA_DIR / f"{league_key}_roster.csv",
        "games": DATA_DIR / f"{league_key}_games.csv",
        "stats": DATA_DIR / f"{league_key}_stats.csv",
        "non_game": DATA_DIR / f"{league_key}_non_game_points.csv",
        "highlights": DATA_DIR / f"{league_key}_highlights.csv",
    }


def read_csv_safe(path: Path, cols=None) -> pd.DataFrame:
    if not path.exists():
        if cols is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        if cols is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=cols)
    if cols is not None:
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
    return df


def write_csv_safe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)


# --------------------------------------------------
# HELPERS: CORE DATA ACCESS
# --------------------------------------------------


def get_roster(league_key: str) -> pd.DataFrame:
    p = league_paths(league_key)["roster"]
    cols = ["player_id", "first_name", "last_name", "team_name", "bunk"]
    return read_csv_safe(p, cols)


def save_roster(league_key: str, df: pd.DataFrame):
    p = league_paths(league_key)["roster"]
    write_csv_safe(df, p)


def get_games(league_key: str) -> pd.DataFrame:
    p = league_paths(league_key)["games"]
    cols = [
        "game_id",
        "date",
        "league_key",
        "sport",
        "level",
        "match_type",  # "1v1" or "2v2"
        "team_a",
        "team_a2",
        "team_b",
        "team_b2",
        "score_a",
        "score_b",
        "points_a",
        "points_b",
    ]
    return read_csv_safe(p, cols)


def save_games(league_key: str, df: pd.DataFrame):
    p = league_paths(league_key)["games"]
    write_csv_safe(df, p)


def get_stats(league_key: str) -> pd.DataFrame:
    p = league_paths(league_key)["stats"]
    cols = [
        "game_id",
        "league_key",
        "team_name",
        "player_id",
        "player_name",
        "sport",
        "level",
        "stat_type",
        "value",
    ]
    return read_csv_safe(p, cols)


def save_stats(league_key: str, df: pd.DataFrame):
    p = league_paths(league_key)["stats"]
    write_csv_safe(df, p)


def get_non_game_points(league_key: str) -> pd.DataFrame:
    p = league_paths(league_key)["non_game"]
    cols = ["id", "date", "category", "reason", "team_name", "points"]
    return read_csv_safe(p, cols)


def save_non_game_points(league_key: str, df: pd.DataFrame):
    p = league_paths(league_key)["non_game"]
    write_csv_safe(df, p)


def get_highlights(league_key: str) -> pd.DataFrame:
    p = league_paths(league_key)["highlights"]
    cols = ["id", "date", "title", "description", "file_path", "file_name"]
    return read_csv_safe(p, cols)


def save_highlights(league_key: str, df: pd.DataFrame):
    p = league_paths(league_key)["highlights"]
    write_csv_safe(df, p)


# --------------------------------------------------
# GAME POINT LOGIC
# --------------------------------------------------


def get_base_points_for_game(league_key: str, sport: str, level: str) -> float:
    league_map = GAME_POINT_VALUES.get(league_key, {})
    sport_map = league_map.get(sport, {})
    base = sport_map.get(level)
    if base is None:
        # fallback if something not configured
        return 10.0
    return float(base)


def assign_points_for_result(
    league_key: str, sport: str, level: str, score_a: int, score_b: int
) -> tuple[float, float]:
    base = get_base_points_for_game(league_key, sport, level)
    if score_a > score_b:
        return base, 0.0
    elif score_b > score_a:
        return 0.0, base
    else:
        # Tie: split points
        return base / 2.0, base / 2.0


# --------------------------------------------------
# STANDINGS / COMBINED STANDINGS
# --------------------------------------------------


def calc_team_standings(
    league_key: str, include_non_game: bool = True
) -> pd.DataFrame:
    """Return a standings table for one league (no age weighting). Handles 1v1 and 2v2."""
    games = get_games(league_key)
    non_game = get_non_game_points(league_key)

    roster = get_roster(league_key)
    roster_teams = sorted(roster["team_name"].dropna().unique().tolist())

    if games.empty:
        return pd.DataFrame(
            {
                "Team": roster_teams,
                "Wins": 0,
                "Losses": 0,
                "Ties": 0,
                "Game Points": 0.0,
                "Non-Game Points": 0.0,
                "Total Points": 0.0,
            }
        )

    # Ensure numeric
    games["score_a"] = pd.to_numeric(games["score_a"], errors="coerce").fillna(0)
    games["score_b"] = pd.to_numeric(games["score_b"], errors="coerce").fillna(0)
    games["points_a"] = pd.to_numeric(games["points_a"], errors="coerce").fillna(0.0)
    games["points_b"] = pd.to_numeric(games["points_b"], errors="coerce").fillna(0.0)

    # Collect all teams that have ever appeared
    teams = set()
    for col in ["team_a", "team_b", "team_a2", "team_b2"]:
        teams.update(games[col].dropna().unique().tolist())
    teams = {t for t in teams if t}  # remove empty strings

    # Guarantee teams from roster also show up
    teams.update(roster_teams)

    wins = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    ties = {t: 0 for t in teams}
    game_pts = {t: 0.0 for t in teams}

    for _, row in games.iterrows():
        side_a = [row["team_a"]]
        if pd.notna(row["team_a2"]) and row["team_a2"]:
            side_a.append(row["team_a2"])
        side_b = [row["team_b"]]
        if pd.notna(row["team_b2"]) and row["team_b2"]:
            side_b.append(row["team_b2"])

        score_a = int(row["score_a"])
        score_b = int(row["score_b"])
        p_a = float(row["points_a"])
        p_b = float(row["points_b"])

        if score_a > score_b:
            for t in side_a:
                if t:
                    wins[t] += 1
                    game_pts[t] += p_a
            for t in side_b:
                if t:
                    losses[t] += 1
                    game_pts[t] += p_b
        elif score_b > score_a:
            for t in side_b:
                if t:
                    wins[t] += 1
                    game_pts[t] += p_b
            for t in side_a:
                if t:
                    losses[t] += 1
                    game_pts[t] += p_a
        else:
            for t in side_a:
                if t:
                    ties[t] += 1
                    game_pts[t] += p_a
            for t in side_b:
                if t:
                    ties[t] += 1
                    game_pts[t] += p_b

    standings = pd.DataFrame(
        {
            "Team": sorted(teams),
            "Wins": [wins[t] for t in sorted(teams)],
            "Losses": [losses[t] for t in sorted(teams)],
            "Ties": [ties[t] for t in sorted(teams)],
            "Game Points": [game_pts[t] for t in sorted(teams)],
        }
    )

    # Non-game points
    if not non_game.empty:
        non_game["points"] = pd.to_numeric(non_game["points"], errors="coerce").fillna(0)
        non_sum = non_game.groupby("team_name")["points"].sum().reset_index()
        non_sum = non_sum.rename(columns={"team_name": "Team", "points": "Non-Game Points"})
        standings = standings.merge(non_sum, on="Team", how="left")
    else:
        standings["Non-Game Points"] = 0.0

    standings["Non-Game Points"] = standings["Non-Game Points"].fillna(0.0)

    if include_non_game:
        standings["Total Points"] = standings["Game Points"] + standings["Non-Game Points"]
    else:
        standings["Total Points"] = standings["Game Points"]

    standings = standings.sort_values(
        by=["Total Points", "Wins"], ascending=[False, False]
    ).reset_index(drop=True)

    return standings


def calc_combined_standings(include_non_game: bool = True) -> pd.DataFrame:
    """Combine standings across all leagues using league weights."""
    combined_rows = []

    for league_name, cfg in LEAGUES.items():
        key = cfg["key"]
        weight = cfg["weight"]
        st_df = calc_team_standings(key, include_non_game=include_non_game)
        if st_df.empty:
            continue
        for _, row in st_df.iterrows():
            team = row["Team"]
            total = row["Total Points"]
            combined_rows.append(
                {
                    "Team": team,
                    "League": league_name,
                    "League Weight": weight,
                    "League Points": total,
                    "Weighted Points": total * weight,
                }
            )

    if not combined_rows:
        return pd.DataFrame(
            columns=["Team", "Total Raw Points", "Total Weighted Points"]
        )

    df = pd.DataFrame(combined_rows)
    agg = df.groupby("Team").agg(
        Total_Raw=("League Points", "sum"),
        Total_Weighted=("Weighted Points", "sum"),
    ).reset_index()

    agg = agg.sort_values(
        by=["Total_Weighted", "Total_Raw"], ascending=[False, False]
    ).reset_index(drop=True)

    agg = agg.rename(
        columns={
            "Team": "Team",
            "Total_Raw": "Total Raw Points",
            "Total_Weighted": "Total Weighted Points",
        }
    )
    return agg


# --------------------------------------------------
# LIVE GAME STATE HELPERS
# --------------------------------------------------


def get_live_state(league_key: str) -> dict:
    """Get or initialize live game state for a league (stored in session_state)."""
    key = f"live_{league_key}"
    if key not in st.session_state:
        st.session_state[key] = {"status": "idle"}
    return st.session_state[key]


def reset_live_state(league_key: str):
    key = f"live_{league_key}"
    st.session_state[key] = {"status": "idle"}


def compute_timer_display(live: dict) -> str:
    """Compute timer text from live state."""
    minutes = live.get("timer_minutes", 0)
    if minutes <= 0:
        return "No timer set"

    total_sec = minutes * 60
    offset = int(live.get("timer_offset", 0))
    running = live.get("timer_running", False)
    start_iso = live.get("timer_start", "")

    if running and start_iso:
        try:
            start = datetime.fromisoformat(start_iso)
            now = datetime.now()
            offset = offset + max(0, int((now - start).total_seconds()))
        except Exception:
            pass

    elapsed = max(0, min(offset, total_sec))
    remaining = max(0, total_sec - elapsed)

    m_el = elapsed // 60
    s_el = elapsed % 60
    m_rem = remaining // 60
    s_rem = remaining % 60

    return f"Elapsed: {m_el:02d}:{s_el:02d}  |  Remaining: {m_rem:02d}:{s_rem:02d}"


def set_flash_stat(league_key: str, player_name: str, team_name: str, stat_label: str):
    """Set a short-lived 'flash' confirmation for a stat change."""
    key = f"flash_{league_key}"
    st.session_state[key] = {
        "player_name": player_name,
        "team_name": team_name,
        "stat_label": stat_label,
        "expires_at": time.time() + 1.5,  # ~1–2 seconds
    }


def get_flash_stat(league_key: str):
    key = f"flash_{league_key}"
    fs = st.session_state.get(key)
    if not fs:
        return None
    if time.time() > fs.get("expires_at", 0):
        # expire
        st.session_state[key] = None
        return None
    return fs


# --------------------------------------------------
# PAGES
# --------------------------------------------------


def page_setup(league_name: str):
    st.header(f"Setup – {league_name}")
    league_key = league_key_from_name(league_name)

    st.subheader("Upload / Replace Roster CSV")
    st.caption(
        "CSV must contain columns: player_id, first_name, last_name, team_name, bunk"
    )

    roster_file = st.file_uploader(
        "Upload roster CSV", type="csv", key=f"roster_upload_{league_key}"
    )
    if roster_file is not None:
        df = pd.read_csv(roster_file)
        save_roster(league_key, df)
        st.success("Roster uploaded and saved.")

    df_roster = get_roster(league_key)
    if not df_roster.empty:
        st.subheader("Current Roster")
        st.dataframe(df_roster, use_container_width=True)
    else:
        st.info("No roster uploaded yet.")


def page_enter_scores(league_name: str):
    st.header(f"Enter Scores & Stats – {league_name}")
    league_key = league_key_from_name(league_name)

    mode = st.radio(
        "Entry mode",
        ["Post-game entry", "Live game (in-progress)"],
        horizontal=True,
        key=f"entry_mode_{league_key}",
    )

    if mode == "Post-game entry":
        page_post_game_entry(league_key, league_name)
    else:
        page_live_game(league_key, league_name)


def page_post_game_entry(league_key: str, league_name: str):
    roster = get_roster(league_key)
    if roster.empty:
        st.warning("Upload a roster first on the Setup page.")
        return

    teams = sorted(roster["team_name"].dropna().unique().tolist())
    if len(teams) < 2:
        st.warning("You need at least two teams in the roster.")
        return

    st.subheader("Game Result (Post-Game)")

    match_type = st.radio(
        "Game type",
        ["One team vs one team", "Two teams vs two teams (combined)"],
        key=f"pg_match_type_{league_key}",
    )
    is_2v2 = match_type.startswith("Two teams")

    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Game date", value=datetime.today())
    with col2:
        sport = st.selectbox(
            "Sport",
            SPORT_OPTIONS,
            key=f"pg_sport_{league_key}",
        )
    with col3:
        level = st.selectbox(
            "Level (A/B/C/D)",
            ["A", "B", "C", "D"],
            key=f"pg_level_{league_key}",
        )

    if not is_2v2:
        col4, col5 = st.columns(2)
        with col4:
            team_a = st.selectbox("Team A", teams, key=f"{league_key}_pg_team_a")
            score_a = st.number_input("Score A", min_value=0, step=1, value=0)
        with col5:
            team_b = st.selectbox(
                "Team B", [t for t in teams if t != team_a], key=f"{league_key}_pg_team_b"
            )
            score_b = st.number_input("Score B", min_value=0, step=1, value=0)

        team_a2 = ""
        team_b2 = ""
    else:
        st.caption("Select two teams for each side. Scores are combined for each side.")
        col4, col5 = st.columns(2)
        with col4:
            team_a = st.selectbox("Side A – Team 1", teams, key=f"{league_key}_pg_team_a1")
            remaining_a = [t for t in teams if t != team_a]
            team_a2 = st.selectbox("Side A – Team 2", remaining_a, key=f"{league_key}_pg_team_a2")
        with col5:
            remaining_for_b = [t for t in teams if t not in [team_a, team_a2]]
            team_b = st.selectbox("Side B – Team 1", remaining_for_b, key=f"{league_key}_pg_team_b1")
            remaining_b2 = [t for t in remaining_for_b if t != team_b]
            team_b2 = st.selectbox(
                "Side B – Team 2",
                remaining_b2 if remaining_b2 else remaining_for_b,
                key=f"{league_key}_pg_team_b2",
            )

        score_a = st.number_input("Combined Score for Side A", min_value=0, step=1, value=0)
        score_b = st.number_input("Combined Score for Side B", min_value=0, step=1, value=0)

    st.caption(
        "Game points are based on league, sport, and level (e.g., Senior A Softball = 50). "
        "Winner gets full points; losers get 0; ties split the points."
    )

    if st.button("Save Game Result", key=f"save_game_pg_{league_key}"):
        games = get_games(league_key)
        new_id = 1 if games.empty else int(games["game_id"].max()) + 1

        points_a, points_b = assign_points_for_result(
            league_key, sport, level, int(score_a), int(score_b)
        )

        new_row = {
            "game_id": new_id,
            "date": date.isoformat(),
            "league_key": league_key,
            "sport": sport,
            "level": level,
            "match_type": "2v2" if is_2v2 else "1v1",
            "team_a": team_a,
            "team_a2": team_a2 if is_2v2 else "",
            "team_b": team_b,
            "team_b2": team_b2 if is_2v2 else "",
            "score_a": int(score_a),
            "score_b": int(score_b),
            "points_a": points_a,
            "points_b": points_b,
        }
        games = pd.concat([games, pd.DataFrame([new_row])], ignore_index=True)
        save_games(league_key, games)
        st.success("Game saved.")

    st.markdown("---")

    st.subheader("Existing Games")
    games = get_games(league_key)
    if games.empty:
        st.info("No games recorded yet.")
    else:
        st.dataframe(games, use_container_width=True)


def page_live_game(league_key: str, league_name: str):
    roster = get_roster(league_key)
    if roster.empty:
        st.warning("Upload a roster first on the Setup page.")
        return

    teams = sorted(roster["team_name"].dropna().unique().tolist())
    if len(teams) < 2:
        st.warning("You need at least two teams in the roster.")
        return

    live = get_live_state(league_key)

    # ------------ NO GAME IN PROGRESS: SETUP ------------
    if live.get("status", "idle") != "in_progress":
        st.subheader("Start New Live Game")

        match_type = st.radio(
            "Game type",
            ["One team vs one team", "Two teams vs two teams (combined)"],
            key=f"lg_match_type_{league_key}",
        )
        is_2v2 = match_type.startswith("Two teams")

        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input("Game date", value=datetime.today(), key=f"lg_date_{league_key}")
        with col2:
            sport = st.selectbox(
                "Sport",
                SPORT_OPTIONS,
                key=f"lg_sport_{league_key}",
            )
        with col3:
            level = st.selectbox(
                "Level (A/B/C/D)",
                ["A", "B", "C", "D"],
                key=f"lg_level_{league_key}",
            )

        if not is_2v2:
            col4, col5 = st.columns(2)
            with col4:
                team_a = st.selectbox("Team A", teams, key=f"lg_team_a_{league_key}")
            with col5:
                team_b = st.selectbox(
                    "Team B", [t for t in teams if t != team_a], key=f"lg_team_b_{league_key}"
                )
            team_a2 = ""
            team_b2 = ""
        else:
            st.caption("Select two teams for each side. Stats still track per original team.")
            col4, col5 = st.columns(2)
            with col4:
                team_a = st.selectbox("Side A – Team 1", teams, key=f"lg_team_a1_{league_key}")
                remaining_a = [t for t in teams if t != team_a]
                team_a2 = st.selectbox("Side A – Team 2", remaining_a, key=f"lg_team_a2_{league_key}")
            with col5:
                remaining_for_b = [t for t in teams if t not in [team_a, team_a2]]
                team_b = st.selectbox("Side B – Team 1", remaining_for_b, key=f"lg_team_b1_{league_key}")
                remaining_b2 = [t for t in remaining_for_b if t != team_b]
                team_b2 = st.selectbox(
                    "Side B – Team 2",
                    remaining_b2 if remaining_b2 else remaining_for_b,
                    key=f"lg_team_b2_{league_key}",
                )

        # Timer options based on sport
        presets = SPORT_TIMER_PRESETS.get(sport, DEFAULT_TIMER_PRESETS)
        preset_labels = [p[0] for p in presets]
        timer_label = st.selectbox(
            "Timer preset",
            preset_labels,
            key=f"lg_timer_{league_key}",
        )
        timer_minutes = dict(presets)[timer_label]

        # Active lineups
        st.markdown("<div class='clm-section-title'>Active Lineups for This Game</div>",
                    unsafe_allow_html=True)

        # Helper to build roster per team
        def with_display(df):
            df = df.copy()
            df["display"] = df.apply(
                lambda r: f"{r['first_name']} {r['last_name']} ({r['player_id']})",
                axis=1,
            )
            return df

        roster_a1 = with_display(roster[roster["team_name"] == team_a])
        roster_b1 = with_display(roster[roster["team_name"] == team_b])
        roster_a2 = with_display(roster[roster["team_name"] == team_a2]) if team_a2 else pd.DataFrame()
        roster_b2 = with_display(roster[roster["team_name"] == team_b2]) if team_b2 else pd.DataFrame()

        col_pa, col_pb = st.columns(2)
        with col_pa:
            active_a1_disp = st.multiselect(
                f"Active players for {team_a}",
                roster_a1["display"].tolist(),
                default=roster_a1["display"].tolist(),
                key=f"lg_active_a1_{league_key}",
            )
            active_a2_disp = []
            if is_2v2 and not roster_a2.empty:
                active_a2_disp = st.multiselect(
                    f"Active players for {team_a2}",
                    roster_a2["display"].tolist(),
                    default=roster_a2["display"].tolist(),
                    key=f"lg_active_a2_{league_key}",
                )
        with col_pb:
            active_b1_disp = st.multiselect(
                f"Active players for {team_b}",
                roster_b1["display"].tolist(),
                default=roster_b1["display"].tolist(),
                key=f"lg_active_b1_{league_key}",
            )
            active_b2_disp = []
            if is_2v2 and not roster_b2.empty:
                active_b2_disp = st.multiselect(
                    f"Active players for {team_b2}",
                    roster_b2["display"].tolist(),
                    default=roster_b2["display"].tolist(),
                    key=f"lg_active_b2_{league_key}",
                )

        # Softball batting order (simple number per active player)
        batting_orders = {}
        if sport == "Softball":
            st.markdown("<div class='clm-section-title'>Softball Batting Order</div>",
                        unsafe_allow_html=True)
            st.caption("Optional: set batting order numbers (1, 2, 3, ...) for each active player.")
            for label, roster_side, active_list in [
                (team_a, roster_a1, active_a1_disp),
                (team_a2, roster_a2, active_a2_disp if is_2v2 else []),
                (team_b, roster_b1, active_b1_disp),
                (team_b2, roster_b2, active_b2_disp if is_2v2 else []),
            ]:
                if not label:
                    continue
                if not active_list:
                    continue
                st.write(f"**{label} batting order**")
                for disp in active_list:
                    key = f"bo_{league_key}_{label}_{disp}"
                    spot = st.number_input(
                        f"{disp}",
                        min_value=1,
                        max_value=len(active_list),
                        step=1,
                        value=active_list.index(disp) + 1,
                        key=key,
                    )
                    batting_orders[disp] = spot

        if st.button("Start Live Game", key=f"lg_start_{league_key}"):
            # Validate active players
            if not active_a1_disp or not active_b1_disp:
                st.error("Please select at least one active player for each side.")
                return

            stat_fields = SPORT_STATS.get(sport, [])
            players = []

            def add_players(roster_side, active_display, team_name):
                # Sort by batting order if defined, else keep original order
                if sport == "Softball":
                    sorted_disp = sorted(
                        active_display,
                        key=lambda d: batting_orders.get(d, 999),
                    )
                else:
                    sorted_disp = active_display

                for disp in sorted_disp:
                    row = roster_side[roster_side["display"] == disp].iloc[0]
                    player = {
                        "player_id": str(row["player_id"]),
                        "player_name": f"{row['first_name']} {row['last_name']}",
                        "team_name": team_name,
                    }
                    for sf in stat_fields:
                        player[sf] = 0
                    players.append(player)

            add_players(roster_a1, active_a1_disp, team_a)
            if is_2v2 and not roster_a2.empty and active_a2_disp:
                add_players(roster_a2, active_a2_disp, team_a2)
            add_players(roster_b1, active_b1_disp, team_b)
            if is_2v2 and not roster_b2.empty and active_b2_disp:
                add_players(roster_b2, active_b2_disp, team_b2)

            # Initialize live state
            live["status"] = "in_progress"
            live["date"] = date.isoformat()
            live["sport"] = sport
            live["level"] = level
            live["match_type"] = "2v2" if is_2v2 else "1v1"
            live["team_a"] = team_a
            live["team_a2"] = team_a2 if is_2v2 else ""
            live["team_b"] = team_b
            live["team_b2"] = team_b2 if is_2v2 else ""
            live["score_a"] = 0
            live["score_b"] = 0
            live["timer_minutes"] = timer_minutes
            live["timer_start"] = ""  # will be set when user hits Start
            live["timer_offset"] = 0
            live["timer_running"] = False
            live["players"] = players
            live["stat_fields"] = stat_fields

            st.success("Live game started. Use the scoreboard below to track the game.")
            st.rerun()

        return

    # ------------ GAME IN PROGRESS ------------
    st.subheader("Live Game In Progress")

    # Game summary
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**League:** {league_name}")
        st.markdown(f"**Sport:** {live.get('sport', '')}")
        st.markdown(f"**Level:** {live.get('level', '')}")
        side_label_a = (
            live.get("team_a", "")
            if live.get("match_type") != "2v2"
            else f"{live.get('team_a', '')} + {live.get('team_a2', '')}"
        )
        side_label_b = (
            live.get("team_b", "")
            if live.get("match_type") != "2v2"
            else f"{live.get('team_b', '')} + {live.get('team_b2', '')}"
        )
        st.markdown(f"**Sides:** {side_label_a} vs {side_label_b}")
        st.markdown(f"**Date:** {live.get('date', '')}")
    with col_info2:
        st.markdown("<div class='clm-section-title'>Timer</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='clm-timer'>{compute_timer_display(live)}</div>",
            unsafe_allow_html=True,
        )
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            if st.button("Start / Resume", key=f"timer_start_{league_key}"):
                if live.get("timer_minutes", 0) > 0 and not live.get("timer_running", False):
                    live["timer_running"] = True
                    live["timer_start"] = datetime.now().isoformat()
                    st.rerun()
        with col_t2:
            if st.button("Pause", key=f"timer_pause_{league_key}"):
                if live.get("timer_minutes", 0) > 0 and live.get("timer_running", False):
                    try:
                        start = datetime.fromisoformat(live.get("timer_start", ""))
                        now = datetime.now()
                        delta_sec = max(0, int((now - start).total_seconds()))
                    except Exception:
                        delta_sec = 0
                    live["timer_offset"] = int(live.get("timer_offset", 0)) + delta_sec
                    live["timer_running"] = False
                    live["timer_start"] = ""
                    st.rerun()
        with col_t3:
            if st.button("Reset", key=f"timer_reset_{league_key}"):
                live["timer_offset"] = 0
                live["timer_running"] = False
                live["timer_start"] = ""
                st.rerun()

    st.markdown("---")

    # Big SCOREBOARD
    match_type = live.get("match_type", "1v1")
    team_a = live.get("team_a", "")
    team_a2 = live.get("team_a2", "")
    team_b = live.get("team_b", "")
    team_b2 = live.get("team_b2", "")
    score_a = int(live.get("score_a", 0))
    score_b = int(live.get("score_b", 0))
    sport = live.get("sport", "")

    col_sa, col_timer_mid, col_sb = st.columns([3, 2, 3])

    with col_sa:
        st.markdown(
            f"<div class='clm-team-name'>{side_label_a}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='clm-scoreboard'>{score_a}</div>", unsafe_allow_html=True)

        # Team A scoring buttons
        if sport == "Basketball":
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("+1", key=f"lg_score_a_plus1_{league_key}"):
                live["score_a"] = score_a + 1
                st.rerun()
            if b2.button("+2", key=f"lg_score_a_plus2_{league_key}"):
                live["score_a"] = score_a + 2
                st.rerun()
            if b3.button("+3", key=f"lg_score_a_plus3_{league_key}"):
                live["score_a"] = score_a + 3
                st.rerun()
            if b4.button("-1", key=f"lg_score_a_minus1_{league_key}"):
                live["score_a"] = max(0, score_a - 1)
                st.rerun()
        else:
            b1, b2 = st.columns(2)
            if b1.button("+1", key=f"lg_score_a_plus_{league_key}"):
                live["score_a"] = score_a + 1
                st.rerun()
            if b2.button("-1", key=f"lg_score_a_minus_{league_key}"):
                live["score_a"] = max(0, score_a - 1)
                st.rerun()

    with col_timer_mid:
        # spacer
        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    with col_sb:
        st.markdown(
            f"<div class='clm-team-name'>{side_label_b}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='clm-scoreboard'>{score_b}</div>", unsafe_allow_html=True)

        # Team B scoring buttons
        if sport == "Basketball":
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("+1 ", key=f"lg_score_b_plus1_{league_key}"):
                live["score_b"] = score_b + 1
                st.rerun()
            if b2.button("+2 ", key=f"lg_score_b_plus2_{league_key}"):
                live["score_b"] = score_b + 2
                st.rerun()
            if b3.button("+3 ", key=f"lg_score_b_plus3_{league_key}"):
                live["score_b"] = score_b + 3
                st.rerun()
            if b4.button("-1 ", key=f"lg_score_b_minus1_{league_key}"):
                live["score_b"] = max(0, score_b - 1)
                st.rerun()
        else:
            b1, b2 = st.columns(2)
            if b1.button("+1 ", key=f"lg_score_b_plus_{league_key}"):
                live["score_b"] = score_b + 1
                st.rerun()
            if b2.button("-1 ", key=f"lg_score_b_minus_{league_key}"):
                live["score_b"] = max(0, score_b - 1)
                st.rerun()

    st.markdown("---")

    # Current stats table + per-player controls
    players = live.get("players", [])
    stat_fields = live.get("stat_fields", [])

    if not players or not stat_fields:
        st.warning("No active players or stat fields set for this game.")
    else:
        df_players = pd.DataFrame(players)

        st.markdown("<div class='clm-section-title'>Current Player Stats (This Game)</div>",
                    unsafe_allow_html=True)

        side_a_teams = [team_a]
        if team_a2:
            side_a_teams.append(team_a2)
        side_b_teams = [team_b]
        if team_b2:
            side_b_teams.append(team_b2)

        flash = get_flash_stat(league_key)

        col_ta, col_tb = st.columns(2)

        with col_ta:
            st.markdown(f"#### {side_label_a}")
            df_a = df_players[df_players["team_name"].isin(side_a_teams)].copy()
            if df_a.empty:
                st.info("No active players on this side.")
            else:
                for idx, row in df_a.iterrows():
                    p_name = row["player_name"]
                    team_name = row["team_name"]
                    st.write(f"**{p_name}** ({team_name})", unsafe_allow_html=True)
                    cols_row = st.columns(len(stat_fields))
                    for j, sf in enumerate(stat_fields):
                        label = STAT_LABELS.get(sf, sf.replace("_", " ").title())
                        if cols_row[j].button(
                            f"+1 {label}",
                            key=f"{league_key}_A_{idx}_{sf}_plus",
                        ):
                            for p in players:
                                if p["player_name"] == p_name and p["team_name"] == team_name:
                                    p[sf] = p.get(sf, 0) + 1
                                    break
                            live["players"] = players
                            set_flash_stat(league_key, p_name, team_name, label)
                            st.rerun()
                    # flash confirmation
                    if flash and flash["player_name"] == p_name and flash["team_name"] == team_name:
                        st.markdown(
                            f"<span class='clm-flash'>+1 {flash['stat_label']} added</span>",
                            unsafe_allow_html=True,
                        )

        with col_tb:
            st.markdown(f"#### {side_label_b}")
            df_b = df_players[df_players["team_name"].isin(side_b_teams)].copy()
            if df_b.empty:
                st.info("No active players on this side.")
            else:
                for idx, row in df_b.iterrows():
                    p_name = row["player_name"]
                    team_name = row["team_name"]
                    st.write(f"**{p_name}** ({team_name})", unsafe_allow_html=True)
                    cols_row = st.columns(len(stat_fields))
                    for j, sf in enumerate(stat_fields):
                        label = STAT_LABELS.get(sf, sf.replace("_", " ").title())
                        if cols_row[j].button(
                            f"+1 {label}",
                            key=f"{league_key}_B_{idx}_{sf}_plus",
                        ):
                            for p in players:
                                if p["player_name"] == p_name and p["team_name"] == team_name:
                                    p[sf] = p.get(sf, 0) + 1
                                    break
                            live["players"] = players
                            set_flash_stat(league_key, p_name, team_name, label)
                            st.rerun()
                    if flash and flash["player_name"] == p_name and flash["team_name"] == team_name:
                        st.markdown(
                            f"<span class='clm-flash'>+1 {flash['stat_label']} added</span>",
                            unsafe_allow_html=True,
                        )

        st.markdown("---")

        # Adjust active lineup if needed (remove players mid-game)
        st.markdown("<div class='clm-section-title'>Adjust Active Lineup</div>",
                    unsafe_allow_html=True)

        all_names = df_players["player_name"].tolist()
        current_keep = st.multiselect(
            "Players currently active in this game",
            all_names,
            default=all_names,
            key=f"lg_keep_players_{league_key}",
        )

        if st.button("Apply lineup changes", key=f"lg_apply_lineup_{league_key}"):
            new_players = [p for p in players if p["player_name"] in current_keep]
            live["players"] = new_players
            st.success("Lineup updated for this game.")
            st.rerun()

        st.markdown("---")
        st.markdown("<div class='clm-section-title'>Stats Summary for This Game</div>",
                    unsafe_allow_html=True)

        # Stats summary table for this game
        if stat_fields:
            summary_df = df_players[["team_name", "player_name"] + stat_fields].copy()
            summary_df = summary_df.sort_values(["team_name", "player_name"]).reset_index(drop=True)
            st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # Finalize or cancel game
    col_end1, col_end2 = st.columns(2)
    with col_end1:
        if st.button("Finalize Game & Save", key=f"lg_finalize_{league_key}"):
            games = get_games(league_key)
            stats_df = get_stats(league_key)

            # Determine new game_id
            new_game_id = 1 if games.empty else int(games["game_id"].max()) + 1

            score_a_final = int(live.get("score_a", 0))
            score_b_final = int(live.get("score_b", 0))

            points_a, points_b = assign_points_for_result(
                league_key,
                live.get("sport", ""),
                live.get("level", ""),
                score_a_final,
                score_b_final,
            )

            game_row = {
                "game_id": new_game_id,
                "date": live.get("date", datetime.today().isoformat()),
                "league_key": league_key,
                "sport": live.get("sport", ""),
                "level": live.get("level", ""),
                "match_type": live.get("match_type", "1v1"),
                "team_a": live.get("team_a", ""),
                "team_a2": live.get("team_a2", ""),
                "team_b": live.get("team_b", ""),
                "team_b2": live.get("team_b2", ""),
                "score_a": score_a_final,
                "score_b": score_b_final,
                "points_a": points_a,
                "points_b": points_b,
            }

            games = pd.concat([games, pd.DataFrame([game_row])], ignore_index=True)
            save_games(league_key, games)

            # Save stats in long format
            rows = []
            sport = live.get("sport", "")
            level = live.get("level", "")
            for p in live.get("players", []):
                for sf in live.get("stat_fields", []):
                    val = int(p.get(sf, 0))
                    if val != 0:
                        rows.append(
                            {
                                "game_id": new_game_id,
                                "league_key": league_key,
                                "team_name": p["team_name"],
                                "player_id": p["player_id"],
                                "player_name": p["player_name"],
                                "sport": sport,
                                "level": level,
                                "stat_type": sf,
                                "value": val,
                            }
                        )
            if rows:
                stats_df = pd.concat([stats_df, pd.DataFrame(rows)], ignore_index=True)
                save_stats(league_key, stats_df)

            reset_live_state(league_key)
            st.success("Game and stats saved.")
            st.rerun()

    with col_end2:
        if st.button("Cancel Live Game (discard)", key=f"lg_cancel_{league_key}"):
            reset_live_state(league_key)
            st.warning("Live game discarded (no data saved).")
            st.rerun()

    # Auto-refresh while timer running (for live ticking)
    if live.get("timer_running", False) and live.get("timer_minutes", 0) > 0:
        time.sleep(1)
        st.rerun()


def page_standings(league_name: str):
    st.header(f"Standings – {league_name}")
    league_key = league_key_from_name(league_name)

    include_non = st.checkbox(
        "Include non-game points in totals", value=True, key=f"include_non_{league_key}"
    )

    standings = calc_team_standings(league_key, include_non_game=include_non)
    if standings.empty:
        st.info("No data yet.")
        return

    st.dataframe(standings, use_container_width=True)


def page_leaderboards(league_name: str):
    st.header(f"Leaderboards – {league_name}")
    league_key = league_key_from_name(league_name)

    stats_df = get_stats(league_key)
    if stats_df.empty:
        st.info("No stats recorded yet. Use Live Game mode to record stats.")
        return

    stats_df["value"] = pd.to_numeric(stats_df["value"], errors="coerce").fillna(0)

    sports = sorted(stats_df["sport"].dropna().unique().tolist())
    stat_types = sorted(stats_df["stat_type"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        sport_choice = st.selectbox(
            "Sport filter",
            ["All sports"] + sports,
            key=f"lb_sport_{league_key}",
        )
    with col2:
        stat_choice = st.selectbox(
            "Stat type",
            stat_types,
            key=f"lb_stat_{league_key}",
        )

    df = stats_df[stats_df["stat_type"] == stat_choice].copy()
    if sport_choice != "All sports":
        df = df[df["sport"] == sport_choice]

    if df.empty:
        st.info("No matching stats for this filter yet.")
        return

    agg = (
        df.groupby(["player_id", "player_name", "team_name"])
        .agg(Total=("value", "sum"))
        .reset_index()
    )

    agg = agg.sort_values(by="Total", ascending=False).reset_index(drop=True)
    agg.index = agg.index + 1  # rank starting at 1

    st.subheader(f"Leaderboard – {stat_choice} ({sport_choice})")
    st.dataframe(agg, use_container_width=True)


def page_non_game_points(league_name: str):
    st.header(f"Non-Game Points – {league_name}")
    league_key = league_key_from_name(league_name)

    teams = sorted(get_roster(league_key)["team_name"].dropna().unique().tolist())
    if not teams:
        st.warning("Upload a roster first so teams are available.")
        return

    df_non = get_non_game_points(league_key)

    st.subheader("Add Non-Game Points")
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=datetime.today(), key=f"non_date_{league_key}")
        team = st.selectbox("Team", teams, key=f"non_team_{league_key}")
    with col2:
        category = st.selectbox(
            "Category", NON_GAME_CATEGORIES, key=f"non_cat_{league_key}"
        )
        points = st.number_input("Points", step=1, value=1, key=f"non_points_{league_key}")

    reason = st.text_input("Reason / details", key=f"non_reason_{league_key}")

    if st.button("Add Non-Game Points", key=f"non_add_{league_key}"):
        new_id = 1 if df_non.empty else int(df_non["id"].max()) + 1
        new_row = {
            "id": new_id,
            "date": date.isoformat(),
            "category": category,
            "reason": reason,
            "team_name": team,
            "points": points,
        }
        df_non = pd.concat([df_non, pd.DataFrame([new_row])], ignore_index=True)
        save_non_game_points(league_key, df_non)
        st.success("Non-game points added.")

    st.markdown("---")
    st.subheader("Existing Non-Game Points")
    df_non = get_non_game_points(league_key)
    if df_non.empty:
        st.info("No non-game points yet.")
    else:
        st.dataframe(df_non, use_container_width=True)


def page_highlights(league_name: str):
    st.header(f"Highlights – {league_name}")
    league_key = league_key_from_name(league_name)

    df_high = get_highlights(league_key)

    st.subheader("Add Highlight (upload file)")
    date = st.date_input("Date", value=datetime.today(), key=f"hl_date_{league_key}")
    title = st.text_input("Title", key=f"hl_title_{league_key}")
    desc = st.text_area("Description", key=f"hl_desc_{league_key}")
    uploaded_file = st.file_uploader(
        "Highlight video file",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key=f"hl_file_{league_key}",
    )

    if st.button("Save Highlight", key=f"hl_save_{league_key}"):
        if uploaded_file is None:
            st.error("Please upload a highlight file.")
        else:
            league_dir = DATA_DIR / "highlights_files" / league_key
            league_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            save_path = league_dir / filename
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            new_id = 1 if df_high.empty else int(df_high["id"].max()) + 1
            new_row = {
                "id": new_id,
                "date": date.isoformat(),
                "title": title,
                "description": desc,
                "file_path": str(save_path),
                "file_name": uploaded_file.name,
            }
            df_high = pd.concat([df_high, pd.DataFrame([new_row])], ignore_index=True)
            save_highlights(league_key, df_high)
            st.success("Highlight saved.")

    st.markdown("---")
    st.subheader("Highlights List")
    df_high = get_highlights(league_key)
    if df_high.empty:
        st.info("No highlights yet.")
    else:
        st.dataframe(df_high[["id", "date", "title", "description", "file_name"]], use_container_width=True)
        st.markdown("### Preview (first few videos)")
        # Show up to 3 previews
        for _, row in df_high.head(3).iterrows():
            path = row.get("file_path")
            if path and Path(path).exists():
                st.write(f"**{row['title']}** – {row['file_name']}")
                st.video(path)


def page_display_board(current_league_name: str):
    st.header("Display Board")

    mode = st.selectbox(
        "Display mode",
        ["This league only", "All leagues overview"],
        key="display_mode",
    )

    include_non = st.checkbox(
        "Include non-game points in totals",
        value=True,
        key="display_include_non",
    )

    if mode == "This league only":
        league_name = st.selectbox(
            "League to display",
            list(LEAGUES.keys()),
            index=list(LEAGUES.keys()).index(current_league_name),
            key="display_single_league",
        )
        league_key = league_key_from_name(league_name)
        st.subheader(f"{league_name} Standings")

        standings = calc_team_standings(league_key, include_non_game=include_non)
        if standings.empty:
            st.info("No data to show yet.")
        else:
            st.dataframe(standings, use_container_width=True)
        st.caption("Use your browser full-screen view when showing this on the mess hall TV.")

    else:
        st.subheader("All-Leagues Standings Overview")

        # Top: three leagues side by side
        col1, col2, col3 = st.columns(3)
        for (league_name, cfg), col in zip(LEAGUES.items(), [col1, col2, col3]):
            with col:
                st.markdown(f"### {league_name}")
                league_key = cfg["key"]
                st_df = calc_team_standings(league_key, include_non_game=include_non)
                if st_df.empty:
                    st.info("No data yet.")
                else:
                    st.dataframe(
                        st_df[["Team", "Wins", "Losses", "Ties", "Total Points"]],
                        use_container_width=True,
                    )

        st.markdown("---")
        st.subheader("Combined All-League Standings (Age-Weighted)")

        combined = calc_combined_standings(include_non_game=include_non)
        if combined.empty:
            st.info("No data yet.")
        else:
            st.dataframe(combined, use_container_width=True)

        st.caption(
            "Weights by league (can be adjusted in code): "
            + ", ".join(
                f"{name}: x{cfg['weight']}" for name, cfg in LEAGUES.items()
            )
        )


def page_admin():
    st.header("Admin / Clear Data")

    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False

    if not st.session_state.admin_ok:
        pw = st.text_input("Enter admin password", type="password")
        if st.button("Unlock Admin"):
            if pw == ADMIN_PASSWORD:
                st.session_state.admin_ok = True
                st.success("Admin access granted.")
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    st.success("Admin area unlocked.")
    if st.button("Lock Admin Area"):
        st.session_state.admin_ok = False
        st.rerun()

    st.markdown("---")
    st.subheader("Clear Data")

    target = st.selectbox(
        "What do you want to clear?",
        [
            "Nothing",
            "This league – Games only",
            "This league – Stats only",
            "This league – Non-game points only",
            "This league – Highlights only",
            "This league – ALL data (games, stats, non-game, highlights, roster)",
            "ALL leagues – ALL data",
        ],
    )

    league_name = st.selectbox(
        "League (for league-specific clears)",
        list(LEAGUES.keys()),
    )

    confirm = st.checkbox("I understand this cannot be undone.")

    if st.button("Execute Clear") and confirm and target != "Nothing":
        league_key = league_key_from_name(league_name)
        if target.startswith("This league"):
            paths = league_paths(league_key)
            if "Games only" in target:
                write_csv_safe(pd.DataFrame(), paths["games"])
            elif "Stats only" in target:
                write_csv_safe(pd.DataFrame(), paths["stats"])
            elif "Non-game points" in target:
                write_csv_safe(pd.DataFrame(), paths["non_game"])
            elif "Highlights" in target:
                write_csv_safe(pd.DataFrame(), paths["highlights"])
            elif "ALL data" in target:
                for p in paths.values():
                    write_csv_safe(pd.DataFrame(), p)
        elif "ALL leagues" in target:
            for cfg in LEAGUES.values():
                paths = league_paths(cfg["key"])
                for p in paths.values():
                    write_csv_safe(pd.DataFrame(), p)

        st.success("Clear operation complete.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def main():
    # Sidebar: league + navigation
    st.sidebar.title("Crest League Manager v2")

    logo_path = Path("logo-header-2.png")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)

    st.sidebar.markdown("Managing league data for:")
    current_league = st.sidebar.selectbox(
        "League (for setup/stats)",
        list(LEAGUES.keys()),
        key="sidebar_league",
    )

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Go to",
        [
            "Setup",
            "Enter Scores & Stats",
            "Standings",
            "Leaderboards",
            "Non-Game Points",
            "Highlights",
            "Display Board",
            "Admin / Clear Data",
        ],
    )

    if page == "Setup":
        page_setup(current_league)
    elif page == "Enter Scores & Stats":
        page_enter_scores(current_league)
    elif page == "Standings":
        page_standings(current_league)
    elif page == "Leaderboards":
        page_leaderboards(current_league)
    elif page == "Non-Game Points":
        page_non_game_points(current_league)
    elif page == "Highlights":
        page_highlights(current_league)
    elif page == "Display Board":
        page_display_board(current_league)
    elif page == "Admin / Clear Data":
        page_admin()


if __name__ == "__main__":
    main()
