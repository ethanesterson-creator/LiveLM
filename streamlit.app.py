import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Crest League Manager v2",
    layout="wide",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LEAGUES = {
    "Sophomore League": {"key": "soph", "weight": 1.0},
    "Junior League": {"key": "junior", "weight": 1.3},
    "Senior League": {"key": "senior", "weight": 1.6},
}

ADMIN_PASSWORD = "Hyaffa26"

# Non-game point categories used in the Non-Game Points tab
NON_GAME_CATEGORIES = [
    "League Spirit",
    "Sportsmanship",
    "Cleanup / Organization",
    "Participation / Effort",
    "Other",
]

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
# HELPERS: CORE LOGIC
# --------------------------------------------------


def ensure_games_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    return read_csv_safe(league_paths("dummy")["games"], cols) if df is None else df.reindex(
        columns=cols, fill_value=None
    )


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

    # Sum across leagues per team
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

    roster = get_roster(league_key)
    if roster.empty:
        st.warning("Upload a roster first on the Setup page.")
        return

    teams = sorted(roster["team_name"].dropna().unique().tolist())
    if len(teams) < 2:
        st.warning("You need at least two teams in the roster.")
        return

    st.subheader("Game Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Game date", value=datetime.today())
    with col2:
        sport = st.text_input("Sport (e.g., A Hoop, B Softball)")
    with col3:
        level = st.text_input("Level (optional)", value="")

    col4, col5 = st.columns(2)
    with col4:
        team_a = st.selectbox("Team A", teams, key=f"{league_key}_team_a")
        score_a = st.number_input("Score A", min_value=0, step=1, value=0)
    with col5:
        team_b = st.selectbox("Team B", [t for t in teams if t != team_a], key=f"{league_key}_team_b")
        score_b = st.number_input("Score B", min_value=0, step=1, value=0)

    st.caption(
        "For now, game points go in automatically: 2 points for a win, 1 for a tie, 0 for a loss. "
        "We can tweak this table later to match camp rules exactly."
    )

    if st.button("Save Game Result", key=f"save_game_{league_key}"):
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
    st.info("Stat leaderboards still use the existing stats structure from v1. "
            "For Phase 1, this page is unchanged. We can enhance it in a later phase.")


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

    if st.button("Add Non-Game Points", key=f"add_non_{league_key}"):
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
                    st.dataframe(st_df[["Team", "Wins", "Losses", "Ties", "Total Points"]],
                                 use_container_width=True)

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
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")
        return

    st.success("Admin area unlocked.")
    if st.button("Lock Admin Area"):
        st.session_state.admin_ok = False
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Clear Data")

    target = st.selectbox(
        "What do you want to clear?",
        [
            "Nothing",
            "This league – Games only",
            "This league – Non-game points only",
            "This league – Highlights only",
            "This league – ALL data (games, non-game, highlights, roster)",
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
        st.sidebar.image(str(logo_path), use_column_width=True)

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
