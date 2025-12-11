import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# --------------------------------------------------
# CONFIG
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
        ("15:00 – running clock", 15),
        ("20:00 – running clock", 20),
        ("15:00 – stopped clock", 15),
        ("20:00 – stopped clock", 20),
    ],
    "Softball": [
        ("No timer (innings)", 0),
        ("60:00 – time limit", 60),
    ],
    "Hockey": [
        ("No timer", 0),
        ("30:00 – running clock", 30),
        ("45:00 – running clock", 45),
    ],
    "Soccer": [
        ("No timer", 0),
        ("30:00 – running clock", 30),
        ("40:00 – running clock", 40),
    ],
    "Flag Football": [
        ("No timer", 0),
        ("40:00 – running clock", 40),
    ],
    "Kickball": [
        ("No timer (innings)", 0),
        ("45:00 – limit", 45),
    ],
    "Euro": [
        ("No timer", 0),
        ("30:00 – running clock", 30),
    ],
    "Speedball": [
        ("No timer", 0),
        ("30:00 – running clock", 30),
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
        "team_a",
        "team_b",
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
    cols = ["id", "date", "title", "description", "file_name"]
    return read_csv_safe(p, cols)


def save_highlights(league_key: str, df: pd.DataFrame):
    p = league_paths(league_key)["highlights"]
    write_csv_safe(df, p)


# --------------------------------------------------
# STANDINGS / COMBINED STANDINGS
# --------------------------------------------------


def calc_team_standings(
    league_key: str, include_non_game: bool = True
) -> pd.DataFrame:
    """Return a standings table for one league (no age weighting)."""
    games = get_games(league_key)
    non_game = get_non_game_points(league_key)

    if games.empty:
        teams = sorted(get_roster(league_key)["team_name"].dropna().unique().tolist())
        return pd.DataFrame(
            {
                "Team": teams,
                "Wins": 0,
                "Losses": 0,
                "Ties": 0,
                "Game Points": 0,
                "Non-Game Points": 0,
                "Total Points": 0,
            }
        )

    games["score_a"] = pd.to_numeric(games["score_a"], errors="coerce").fillna(0)
    games["score_b"] = pd.to_numeric(games["score_b"], errors="coerce").fillna(0)
    games["points_a"] = pd.to_numeric(games["points_a"], errors="coerce").fillna(0)
    games["points_b"] = pd.to_numeric(games["points_b"], errors="coerce").fillna(0)

    teams = set(games["team_a"]).union(set(games["team_b"]))
    standings = pd.DataFrame({"Team": sorted(t for t in teams if pd.notna(t))})

    def record_for_team(team):
        rows_a = games[games["team_a"] == team]
        rows_b = games[games["team_b"] == team]

        wins = ((rows_a["score_a"] > rows_a["score_b"]).sum() +
                (rows_b["score_b"] > rows_b["score_a"]).sum())
        losses = ((rows_a["score_a"] < rows_a["score_b"]).sum() +
                  (rows_b["score_b"] < rows_b["score_a"]).sum())
        ties = ((rows_a["score_a"] == rows_a["score_b"]).sum() +
                (rows_b["score_b"] == rows_b["score_a"]).sum())

        game_points = rows_a["points_a"].sum() + rows_b["points_b"].sum()
        return wins, losses, ties, game_points

    wins_list, losses_list, ties_list, game_pts_list = [], [], [], []
    for team in standings["Team"]:
        w, l, t, gp = record_for_team(team)
        wins_list.append(w)
        losses_list.append(l)
        ties_list.append(t)
        game_pts_list.append(gp)

    standings["Wins"] = wins_list
    standings["Losses"] = losses_list
    standings["Ties"] = ties_list
    standings["Game Points"] = game_pts_list

    # Non-game points
    if not non_game.empty:
        non_game["points"] = pd.to_numeric(non_game["points"], errors="coerce").fillna(0)
        non_sum = non_game.groupby("team_name")["points"].sum().reset_index()
        non_sum = non_sum.rename(columns={"team_name": "Team", "points": "Non-Game Points"})
        standings = standings.merge(non_sum, on="Team", how="left")
    else:
        standings["Non-Game Points"] = 0

    standings["Non-Game Points"] = standings["Non-Game Points"].fillna(0)

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

    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Game date", value=datetime.today())
    with col2:
        sport = st.text_input("Sport (e.g., A Hoop, B Softball)")
    with col3:
        level = st.selectbox(
            "Level (A/B/C/D)",
            ["A", "B", "C", "D"],
            key=f"pg_level_{league_key}",
        )

    col4, col5 = st.columns(2)
    with col4:
        team_a = st.selectbox("Team A", teams, key=f"{league_key}_pg_team_a")
        score_a = st.number_input("Score A", min_value=0, step=1, value=0)
    with col5:
        team_b = st.selectbox(
            "Team B", [t for t in teams if t != team_a], key=f"{league_key}_pg_team_b"
        )
        score_b = st.number_input("Score B", min_value=0, step=1, value=0)

    st.caption(
        "For now, game points go in automatically: 2 points for a win, 1 for a tie, 0 for a loss. "
        "We can tweak this table later to match camp rules exactly."
    )

    if st.button("Save Game Result", key=f"save_game_pg_{league_key}"):
        games = get_games(league_key)

        if score_a > score_b:
            points_a, points_b = 2, 0
        elif score_a < score_b:
            points_a, points_b = 0, 2
        else:
            points_a = points_b = 1

        new_id = 1 if games.empty else int(games["game_id"].max()) + 1
        new_row = {
            "game_id": new_id,
            "date": date.isoformat(),
            "league_key": league_key,
            "sport": sport,
            "level": level,
            "team_a": team_a,
            "team_b": team_b,
            "score_a": score_a,
            "score_b": score_b,
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

        col4, col5 = st.columns(2)
        with col4:
            team_a = st.selectbox("Team A", teams, key=f"lg_team_a_{league_key}")
        with col5:
            team_b = st.selectbox(
                "Team B", [t for t in teams if t != team_a], key=f"lg_team_b_{league_key}"
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

        roster_a = roster[roster["team_name"] == team_a].copy()
        roster_b = roster[roster["team_name"] == team_b].copy()

        def player_display(row):
            return f"{row['first_name']} {row['last_name']} ({row['player_id']})"

        roster_a["display"] = roster_a.apply(player_display, axis=1)
        roster_b["display"] = roster_b.apply(player_display, axis=1)

        col_pa, col_pb = st.columns(2)
        with col_pa:
            active_a_display = st.multiselect(
                f"Active players for {team_a}",
                roster_a["display"].tolist(),
                default=roster_a["display"].tolist(),
                key=f"lg_active_a_{league_key}",
            )
        with col_pb:
            active_b_display = st.multiselect(
                f"Active players for {team_b}",
                roster_b["display"].tolist(),
                default=roster_b["display"].tolist(),
                key=f"lg_active_b_{league_key}",
            )

        if st.button("Start Live Game", key=f"lg_start_{league_key}"):
            if not active_a_display or not active_b_display:
                st.error("Please select at least one active player for each team.")
                return

            # Build player list
            stat_fields = SPORT_STATS.get(sport, [])
            players = []

            def add_players(roster_side, active_display, team_name):
                for _, r in roster_side.iterrows():
                    if r["display"] not in active_display:
                        continue
                    player = {
                        "player_id": str(r["player_id"]),
                        "player_name": f"{r['first_name']} {r['last_name']}",
                        "team_name": team_name,
                    }
                    for sf in stat_fields:
                        player[sf] = 0
                    players.append(player)

            add_players(roster_a, active_a_display, team_a)
            add_players(roster_b, active_b_display, team_b)

            # Initialize live state
            live["status"] = "in_progress"
            live["date"] = date.isoformat()
            live["sport"] = sport
            live["level"] = level
            live["team_a"] = team_a
            live["team_b"] = team_b
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
        st.markdown(f"**Teams:** {live.get('team_a', '')} vs {live.get('team_b', '')}")
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
                if live.get("timer_minutes", 0) > 0:
                    # If already running, do nothing; else start from now
                    if not live.get("timer_running", False):
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
    team_a = live.get("team_a", "")
    team_b = live.get("team_b", "")
    score_a = int(live.get("score_a", 0))
    score_b = int(live.get("score_b", 0))
    sport = live.get("sport", "")

    col_sa, col_timer_mid, col_sb = st.columns([3, 2, 3])

    with col_sa:
        st.markdown(f"<div class='clm-team-name'>{team_a}</div>", unsafe_allow_html=True)
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
        # Timer already shown above; just leave empty here or add label
        st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    with col_sb:
        st.markdown(f"<div class='clm-team-name'>{team_b}</div>", unsafe_allow_html=True)
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

        # Show & control players by team
        col_ta, col_tb = st.columns(2)

        with col_ta:
            st.markdown(f"#### {team_a}")
            df_a = df_players[df_players["team_name"] == team_a].copy()
            if df_a.empty:
                st.info("No active players on this team.")
            else:
                for idx, row in df_a.iterrows():
                    p_name = row["player_name"]
                    st.write(f"**{p_name}**")
                    cols_row = st.columns(len(stat_fields))
                    for j, sf in enumerate(stat_fields):
                        label = STAT_LABELS.get(sf, sf.replace("_", " ").title())
                        if cols_row[j].button(
                            f"+1 {label}",
                            key=f"{league_key}_A_{idx}_{sf}_plus",
                        ):
                            # Update in live state
                            for p in players:
                                if p["player_name"] == p_name and p["team_name"] == team_a:
                                    p[sf] = p.get(sf, 0) + 1
                                    break
                            live["players"] = players
                            st.rerun()

        with col_tb:
            st.markdown(f"#### {team_b}")
            df_b = df_players[df_players["team_name"] == team_b].copy()
            if df_b.empty:
                st.info("No active players on this team.")
            else:
                for idx, row in df_b.iterrows():
                    p_name = row["player_name"]
                    st.write(f"**{p_name}**")
                    cols_row = st.columns(len(stat_fields))
                    for j, sf in enumerate(stat_fields):
                        label = STAT_LABELS.get(sf, sf.replace("_", " ").title())
                        if cols_row[j].button(
                            f"+1 {label}",
                            key=f"{league_key}_B_{idx}_{sf}_plus",
                        ):
                            for p in players:
                                if p["player_name"] == p_name and p["team_name"] == team_b:
                                    p[sf] = p.get(sf, 0) + 1
                                    break
                            live["players"] = players
                            st.rerun()

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

    # Finalize or cancel game
    col_end1, col_end2 = st.columns(2)
    with col_end1:
        if st.button("Finalize Game & Save", key=f"lg_finalize_{league_key}"):
            games = get_games(league_key)
            stats_df = get_stats(league_key)

            # Determine new game_id
            new_game_id = 1 if games.empty else int(games["game_id"].max()) + 1

            # Compute game points (simple 2/1/0 rule)
            score_a_final = int(live.get("score_a", 0))
            score_b_final = int(live.get("score_b", 0))
            if score_a_final > score_b_final:
                points_a, points_b = 2, 0
            elif score_a_final < score_b_final:
                points_a, points_b = 0, 2
            else:
                points_a = points_b = 1

            game_row = {
                "game_id": new_game_id,
                "date": live.get("date", datetime.today().isoformat()),
                "league_key": league_key,
                "sport": live.get("sport", ""),
                "level": live.get("level", ""),
                "team_a": live.get("team_a", ""),
                "team_b": live.get("team_b", ""),
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

    st.subheader("Add Highlight (file reference)")
    date = st.date_input("Date", value=datetime.today(), key=f"hl_date_{league_key}")
    title = st.text_input("Title", key=f"hl_title_{league_key}")
    desc = st.text_area("Description", key=f"hl_desc_{league_key}")
    file_name = st.text_input(
        "Video file name / URL (for now, just store a reference)",
        key=f"hl_file_{league_key}",
    )

    if st.button("Save Highlight", key=f"hl_save_{league_key}"):
        new_id = 1 if df_high.empty else int(df_high["id"].max()) + 1
        new_row = {
            "id": new_id,
            "date": date.isoformat(),
            "title": title,
            "description": desc,
            "file_name": file_name,
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
        st.dataframe(df_high, use_container_width=True)


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
