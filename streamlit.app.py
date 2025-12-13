import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Crest League Manager – Multi-Game", layout="wide")

ET = pytz.timezone("America/New_York")

LEAGUES = {
    "Sophomore League": {"key": "soph", "weight": 1.0},
    "Junior League": {"key": "junior", "weight": 1.3},
    "Senior League": {"key": "senior", "weight": 1.6},
}

LEVELS = ["A", "B", "C", "D"]

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
SPORTS = list(SPORT_STATS.keys())

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

# Timer presets per sport.
# We model "periods" so basketball can be 2 halves, hockey can be 3 periods, etc.
# timer_mode is label only (running/stopped). We don't enforce stopped clock rules; it's for display/expectations.
SPORT_TIMERS = {
    "Basketball": [
        {"label": "No timer", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "2x 15:00 halves (running)", "periods": 2, "period_min": 15, "mode": "running"},
        {"label": "2x 20:00 halves (running)", "periods": 2, "period_min": 20, "mode": "running"},
        {"label": "2x 15:00 halves (stopped)", "periods": 2, "period_min": 15, "mode": "stopped"},
        {"label": "2x 20:00 halves (stopped)", "periods": 2, "period_min": 20, "mode": "stopped"},
    ],
    "Hockey": [
        {"label": "No timer", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "3x 12:00 periods (running)", "periods": 3, "period_min": 12, "mode": "running"},
        {"label": "3x 15:00 periods (running)", "periods": 3, "period_min": 15, "mode": "running"},
    ],
    "Soccer": [
        {"label": "No timer", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "2x 25:00 halves (running)", "periods": 2, "period_min": 25, "mode": "running"},
        {"label": "2x 30:00 halves (running)", "periods": 2, "period_min": 30, "mode": "running"},
    ],
    "Flag Football": [
        {"label": "No timer", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "2x 20:00 halves (running)", "periods": 2, "period_min": 20, "mode": "running"},
    ],
    "Softball": [
        {"label": "No timer (innings)", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "1x 60:00 time limit", "periods": 1, "period_min": 60, "mode": "running"},
        {"label": "1x 75:00 time limit", "periods": 1, "period_min": 75, "mode": "running"},
    ],
    "Kickball": [
        {"label": "No timer (innings)", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "1x 45:00 time limit", "periods": 1, "period_min": 45, "mode": "running"},
        {"label": "1x 60:00 time limit", "periods": 1, "period_min": 60, "mode": "running"},
    ],
    "Euro": [
        {"label": "No timer", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "2x 20:00 halves (running)", "periods": 2, "period_min": 20, "mode": "running"},
    ],
    "Speedball": [
        {"label": "No timer", "periods": 0, "period_min": 0, "mode": "none"},
        {"label": "2x 20:00 halves (running)", "periods": 2, "period_min": 20, "mode": "running"},
    ],
}

# Game point values (same logic you requested previously)
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
        "Softball": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Flag Football": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Basketball": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Hockey": {"A": 25, "B": 20, "C": 15, "D": 10},
        "Soccer": {"A": 20, "B": 15, "C": 10, "D": 5},
        "Euro": {"A": 20, "B": 15, "C": 10, "D": 5},
        "Speedball": {"A": 20, "B": 15, "C": 10, "D": 5},
        "Kickball": {"A": 15, "B": 10, "C": 5, "D": 5},
    },
}

NON_GAME_CATEGORIES = [
    "League Spirit",
    "Sportsmanship",
    "Cleanup / Organization",
    "Participation / Effort",
    "Other",
]

# =========================
# GOOGLE SHEETS CLIENT
# =========================

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_gspread_client():
    sheet_id = st.secrets["SHEET_ID"]
    sa_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPE)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    return sh

def ws(name: str):
    return get_gspread_client().worksheet(name)

def now_et_iso() -> str:
    return datetime.now(ET).isoformat()

def df_from_ws(name: str) -> pd.DataFrame:
    w = ws(name)
    values = w.get_all_values()
    if not values or len(values) < 1:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    if not rows:
        return pd.DataFrame(columns=headers)
    return pd.DataFrame(rows, columns=headers)

def append_row(name: str, row: List):
    ws(name).append_row(row, value_input_option="USER_ENTERED")

def ensure_headers(name: str, expected: List[str]):
    w = ws(name)
    current = w.row_values(1)
    if current != expected:
        w.clear()
        w.append_row(expected)

# =========================
# UTILS
# =========================

def league_key_from_name(league_name: str) -> str:
    return LEAGUES[league_name]["key"]

def roster_sheet_name(league_key: str) -> str:
    return f"Rosters_{league_key}"

def base_points(league_key: str, sport: str, level: str) -> float:
    return float(GAME_POINT_VALUES.get(league_key, {}).get(sport, {}).get(level, 10))

def points_for_result(league_key: str, sport: str, level: str, score_a: int, score_b: int) -> Tuple[float, float]:
    b = base_points(league_key, sport, level)
    if score_a > score_b:
        return b, 0.0
    if score_b > score_a:
        return 0.0, b
    return b/2.0, b/2.0

def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def make_game_id(league_key: str, sport: str, level: str) -> str:
    # short, readable, unique enough
    stamp = datetime.now(ET).strftime("%Y%m%d-%H%M%S")
    return f"{league_key.upper()}-{sport[:3].upper()}-{level}-{stamp}-{uuid.uuid4().hex[:6]}"

def live_game_row_by_id(game_id: str) -> Optional[Dict[str, str]]:
    df = df_from_ws("LiveGames")
    if df.empty:
        return None
    df = df[df["game_id"] == game_id]
    if df.empty:
        return None
    return df.iloc[0].to_dict()

def update_live_game_cells(game_id: str, updates: Dict[str, str]):
    """
    Update one live game row by game_id. This is the key “multi-user safe-ish” move:
    update only specific cells in the row.
    """
    w = ws("LiveGames")
    data = w.get_all_values()
    if not data or len(data) < 2:
        return
    headers = data[0]
    # find row
    row_idx = None
    for i in range(1, len(data)):
        if len(data[i]) > 0 and data[i][0] == game_id:  # assumes game_id is column A
            row_idx = i + 1  # 1-indexed
            break
    if row_idx is None:
        return

    # batch update cells
    cell_updates = []
    for col_name, value in updates.items():
        if col_name not in headers:
            continue
        col_idx = headers.index(col_name) + 1
        cell_updates.append((row_idx, col_idx, value))
    if not cell_updates:
        return

    # gspread batch_update wants A1 ranges; we’ll do small updates with update_cells
    cells = []
    for r, c, v in cell_updates:
        cells.append(gspread.Cell(r, c, v))
    w.update_cells(cells, value_input_option="USER_ENTERED")

def upsert_live_stat(game_id: str, league_key: str, sport: str, level: str,
                     team_name: str, player_id: str, player_name: str,
                     stat_type: str, delta: int = 1):
    """
    Increment stat row in LiveStats for (game_id, player_id, stat_type).
    """
    w = ws("LiveStats")
    data = w.get_all_values()
    headers = data[0] if data else []
    if not headers:
        return

    # find existing row
    target_row = None
    for i in range(1, len(data)):
        row = data[i]
        if len(row) < 9:
            continue
        if row[0] == game_id and row[5] == player_id and row[7] == stat_type:
            target_row = i + 1
            break

    if target_row is None:
        # append new
        append_row("LiveStats", [
            game_id, league_key, sport, level, team_name, player_id, player_name,
            stat_type, str(delta), now_et_iso()
        ])
    else:
        # update value cell
        value_col = headers.index("value") + 1
        updated_col = headers.index("updated_at") + 1
        current_val = safe_int(w.cell(target_row, value_col).value, 0)
        new_val = max(0, current_val + delta)
        w.update_cell(target_row, value_col, str(new_val))
        w.update_cell(target_row, updated_col, now_et_iso())

def live_stats_for_game(game_id: str) -> pd.DataFrame:
    df = df_from_ws("LiveStats")
    if df.empty:
        return pd.DataFrame()
    df = df[df["game_id"] == game_id].copy()
    if df.empty:
        return df
    df["value"] = df["value"].apply(lambda x: safe_int(x, 0))
    return df

def roster_df(league_key: str) -> pd.DataFrame:
    name = roster_sheet_name(league_key)
    df = df_from_ws(name)
    if df.empty:
        return pd.DataFrame(columns=["player_id", "first_name", "last_name", "team_name", "bunk"])
    return df

def save_roster_to_sheet(league_key: str, df: pd.DataFrame):
    name = roster_sheet_name(league_key)
    w = ws(name)
    w.clear()
    headers = ["player_id", "first_name", "last_name", "team_name", "bunk"]
    w.append_row(headers)
    if df.empty:
        return
    df = df[headers].fillna("")
    w.append_rows(df.values.tolist(), value_input_option="USER_ENTERED")

# =========================
# STANDINGS / LEADERBOARDS
# =========================

def standings_for_league(league_key: str, include_non_game: bool = True) -> pd.DataFrame:
    games = df_from_ws("Games")
    roster = roster_df(league_key)
    teams = sorted(roster["team_name"].dropna().unique().tolist()) if not roster.empty else []

    if games.empty:
        return pd.DataFrame({"Team": teams, "Wins": 0, "Losses": 0, "Ties": 0, "Game Points": 0.0, "Non-Game Points": 0.0, "Total Points": 0.0})

    g = games[games["league_key"] == league_key].copy()
    if g.empty:
        return pd.DataFrame({"Team": teams, "Wins": 0, "Losses": 0, "Ties": 0, "Game Points": 0.0, "Non-Game Points": 0.0, "Total Points": 0.0})

    for c in ["score_a", "score_b", "points_a", "points_b"]:
        g[c] = g[c].apply(lambda x: safe_float(x, 0.0))

    # gather teams from games too
    for col in ["team_a", "team_a2", "team_b", "team_b2"]:
        teams += g[col].dropna().tolist()
    teams = sorted({t for t in teams if str(t).strip() != ""})

    wins = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    ties = {t: 0 for t in teams}
    game_pts = {t: 0.0 for t in teams}

    for _, row in g.iterrows():
        side_a = [row.get("team_a", "")]
        if str(row.get("team_a2", "")).strip():
            side_a.append(row.get("team_a2", ""))
        side_b = [row.get("team_b", "")]
        if str(row.get("team_b2", "")).strip():
            side_b.append(row.get("team_b2", ""))

        sa = safe_int(row.get("score_a", 0), 0)
        sb = safe_int(row.get("score_b", 0), 0)
        pa = safe_float(row.get("points_a", 0.0), 0.0)
        pb = safe_float(row.get("points_b", 0.0), 0.0)

        if sa > sb:
            for t in side_a:
                if t: wins[t] += 1; game_pts[t] += pa
            for t in side_b:
                if t: losses[t] += 1; game_pts[t] += pb
        elif sb > sa:
            for t in side_b:
                if t: wins[t] += 1; game_pts[t] += pb
            for t in side_a:
                if t: losses[t] += 1; game_pts[t] += pa
        else:
            for t in side_a:
                if t: ties[t] += 1; game_pts[t] += pa
            for t in side_b:
                if t: ties[t] += 1; game_pts[t] += pb

    df = pd.DataFrame({
        "Team": teams,
        "Wins": [wins[t] for t in teams],
        "Losses": [losses[t] for t in teams],
        "Ties": [ties[t] for t in teams],
        "Game Points": [game_pts[t] for t in teams],
    })

    # non-game points
    if include_non_game:
        ng = df_from_ws("NonGamePoints")
        if not ng.empty:
            ng = ng[ng["league_key"] == league_key].copy()
            if not ng.empty:
                ng["points"] = ng["points"].apply(lambda x: safe_float(x, 0.0))
                ng_sum = ng.groupby("team_name")["points"].sum().reset_index()
                ng_sum.columns = ["Team", "Non-Game Points"]
                df = df.merge(ng_sum, on="Team", how="left")
        if "Non-Game Points" not in df.columns:
            df["Non-Game Points"] = 0.0
        df["Non-Game Points"] = df["Non-Game Points"].fillna(0.0)
        df["Total Points"] = df["Game Points"] + df["Non-Game Points"]
    else:
        df["Non-Game Points"] = 0.0
        df["Total Points"] = df["Game Points"]

    df = df.sort_values(["Total Points", "Wins"], ascending=[False, False]).reset_index(drop=True)
    return df

def combined_standings(include_non_game: bool = True) -> pd.DataFrame:
    rows = []
    for league_name, cfg in LEAGUES.items():
        lk = cfg["key"]
        w = cfg["weight"]
        st_df = standings_for_league(lk, include_non_game=include_non_game)
        if st_df.empty:
            continue
        for _, r in st_df.iterrows():
            rows.append({
                "Team": r["Team"],
                "League": league_name,
                "League Points": float(r["Total Points"]),
                "Weight": w,
                "Weighted Points": float(r["Total Points"]) * w,
            })
    if not rows:
        return pd.DataFrame(columns=["Team", "Total Raw Points", "Total Weighted Points"])

    df = pd.DataFrame(rows)
    agg = df.groupby("Team").agg(
        Total_Raw=("League Points", "sum"),
        Total_Weighted=("Weighted Points", "sum"),
    ).reset_index()
    agg.columns = ["Team", "Total Raw Points", "Total Weighted Points"]
    return agg.sort_values(["Total Weighted Points", "Total Raw Points"], ascending=[False, False]).reset_index(drop=True)

def leaderboard(league_key: str, sport: Optional[str], stat_type: str) -> pd.DataFrame:
    s = df_from_ws("Stats")
    if s.empty:
        return pd.DataFrame()
    s = s[s["league_key"] == league_key].copy()
    if sport and sport != "All sports":
        s = s[s["sport"] == sport]
    s = s[s["stat_type"] == stat_type]
    if s.empty:
        return pd.DataFrame()
    s["value"] = s["value"].apply(lambda x: safe_int(x, 0))
    agg = s.groupby(["player_id", "player_name", "team_name"])["value"].sum().reset_index()
    agg = agg.rename(columns={"value": "Total"})
    agg = agg.sort_values("Total", ascending=False).reset_index(drop=True)
    agg.index = agg.index + 1
    return agg

# =========================
# UI PAGES
# =========================

def page_setup(league_name: str):
    lk = league_key_from_name(league_name)
    st.header(f"Setup – {league_name}")

    st.subheader("Upload / Replace Roster CSV")
    st.caption("Required columns: player_id, first_name, last_name, team_name, bunk")

    up = st.file_uploader("Upload roster CSV", type="csv", key=f"roster_upload_{lk}")
    if up is not None:
        df = pd.read_csv(up)
        needed = ["player_id", "first_name", "last_name", "team_name", "bunk"]
        for c in needed:
            if c not in df.columns:
                st.error(f"Missing column: {c}")
                return
        save_roster_to_sheet(lk, df[needed].fillna(""))
        st.success("Roster saved to Google Sheets.")

    st.subheader("Current Roster")
    df = roster_df(lk)
    if df.empty:
        st.info("No roster uploaded yet.")
    else:
        st.dataframe(df, use_container_width=True)

def page_live_games_home(league_name: str):
    lk = league_key_from_name(league_name)
    st.header(f"Live Games – {league_name}")

    roster = roster_df(lk)
    if roster.empty:
        st.warning("Upload a roster first (Setup page).")
        return
    teams = sorted(roster["team_name"].dropna().unique().tolist())
    if len(teams) < 2:
        st.warning("Roster needs at least 2 teams.")
        return

    st.subheader("Start a New Live Game (creates a shared Game ID)")
    colA, colB, colC = st.columns(3)
    with colA:
        sport = st.selectbox("Sport", SPORTS, key=f"new_sport_{lk}")
    with colB:
        level = st.selectbox("Level", LEVELS, key=f"new_level_{lk}")
    with colC:
        match_type = st.selectbox("Match Type", ["1v1", "2v2"], key=f"new_match_{lk}")

    if match_type == "1v1":
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Team A", teams, key=f"new_team_a_{lk}")
        with c2:
            team_b = st.selectbox("Team B", [t for t in teams if t != team_a], key=f"new_team_b_{lk}")
        team_a2 = ""
        team_b2 = ""
    else:
        st.caption("Pick 2 teams per side (combined game).")
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Side A – Team 1", teams, key=f"new_team_a1_{lk}")
            team_a2 = st.selectbox("Side A – Team 2", [t for t in teams if t != team_a], key=f"new_team_a2_{lk}")
        with c2:
            remaining = [t for t in teams if t not in [team_a, team_a2]]
            team_b = st.selectbox("Side B – Team 1", remaining, key=f"new_team_b1_{lk}")
            team_b2 = st.selectbox("Side B – Team 2", [t for t in remaining if t != team_b], key=f"new_team_b2_{lk}")

    timer_presets = SPORT_TIMERS.get(sport, [{"label":"No timer","periods":0,"period_min":0,"mode":"none"}])
    timer_label = st.selectbox("Timer Preset", [t["label"] for t in timer_presets], key=f"new_timer_{lk}")
    chosen = next(t for t in timer_presets if t["label"] == timer_label)

    notes = st.text_input("Notes (optional) – e.g., Court 2 / Field 1", key=f"new_notes_{lk}")

    if st.button("Create Live Game", key=f"create_game_{lk}"):
        game_id = make_game_id(lk, sport, level)
        now = now_et_iso()
        ensure_headers("LiveGames", ws("LiveGames").row_values(1))  # no-op but safe

        append_row("LiveGames", [
            game_id,
            "live",
            lk,
            sport,
            level,
            match_type,
            team_a,
            team_a2,
            team_b,
            team_b2,
            "0",
            "0",
            now,
            now,
            chosen["mode"],
            str(chosen["periods"]),
            str(chosen["period_min"] * 60),
            "FALSE",
            "1",
            str(chosen["period_min"] * 60),
            "",
            notes,
        ])
        st.success(f"Live game created: {game_id}")
        st.session_state["active_game_id"] = game_id
        st.rerun()

    st.markdown("---")
    st.subheader("Open an Existing Live Game")
    lg = df_from_ws("LiveGames")
    if lg.empty:
        st.info("No live games yet.")
        return
    lg = lg[(lg["league_key"] == lk) & (lg["status"].isin(["live", "paused"]))].copy()
    if lg.empty:
        st.info("No active live games for this league right now.")
        return

    lg["label"] = lg.apply(lambda r: f"{r['game_id']} – {r['sport']} {r['level']} ({r['match_type']})  {r.get('notes','')}", axis=1)
    pick = st.selectbox("Select a live game to open", lg["label"].tolist(), key=f"open_live_{lk}")
    if st.button("Open Selected Game", key=f"open_btn_{lk}"):
        game_id = pick.split(" – ")[0].strip()
        st.session_state["active_game_id"] = game_id
        st.rerun()

def _timer_tick_if_running(game: Dict[str, str]):
    """
    Tick timer (server-side) for display and for everyone.
    We store last_timer_ts in the sheet; each refresh recalculates and writes back if running.
    """
    if safe_int(game.get("timer_periods", "0"), 0) <= 0:
        return
    if str(game.get("timer_running", "FALSE")).upper() != "TRUE":
        return

    remaining = safe_int(game.get("timer_remaining_sec", "0"), 0)
    last_ts = game.get("last_timer_ts", "")
    now_ts = time.time()

    if last_ts:
        try:
            dt = max(0, int(now_ts - float(last_ts)))
        except Exception:
            dt = 0
    else:
        dt = 0

    if dt <= 0:
        return

    new_remaining = max(0, remaining - dt)
    update_live_game_cells(game["game_id"], {
        "timer_remaining_sec": str(new_remaining),
        "updated_at": now_et_iso(),
        "last_timer_ts": str(now_ts),
        "status": "live" if new_remaining > 0 else "paused",
    })

def fmt_mmss(sec: int) -> str:
    sec = max(0, sec)
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

def page_run_live_game(league_name: str):
    lk = league_key_from_name(league_name)
    st.header(f"Run Live Game – {league_name}")

    game_id = st.session_state.get("active_game_id")
    if not game_id:
        st.info("Go to Live Games page and create/open a game.")
        return

    game = live_game_row_by_id(game_id)
    if not game:
        st.error("That game_id was not found in LiveGames.")
        return

    # Tick timer if needed (so it moves even on refreshes)
    _timer_tick_if_running(game)
    # Reload after tick
    game = live_game_row_by_id(game_id) or game

    sport = game["sport"]
    level = game["level"]
    match_type = game["match_type"]
    side_a_label = game["team_a"] if match_type == "1v1" else f"{game['team_a']} + {game['team_a2']}"
    side_b_label = game["team_b"] if match_type == "1v1" else f"{game['team_b']} + {game['team_b2']}"

    # HEADER
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(f"{sport} {level}  •  {side_a_label} vs {side_b_label}")
        st.caption(f"Game ID: {game_id}  •  Notes: {game.get('notes','')}")
    with c2:
        if st.button("Back to Live Games"):
            st.session_state["active_game_id"] = None
            st.rerun()

    st.markdown("---")

    # TIMER BLOCK
    periods = safe_int(game.get("timer_periods", "0"), 0)
    period_sec = safe_int(game.get("timer_period_sec", "0"), 0)
    period_idx = safe_int(game.get("timer_period_index", "1"), 1)
    remaining = safe_int(game.get("timer_remaining_sec", "0"), 0)
    running = str(game.get("timer_running", "FALSE")).upper() == "TRUE"

    colT1, colT2 = st.columns([2, 3])
    with colT1:
        st.markdown("### Timer")
        if periods <= 0:
            st.info("No timer set for this game.")
        else:
            st.markdown(f"**Period:** {period_idx}/{periods}")
            st.markdown(f"**Time:** `{fmt_mmss(remaining)}`")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Start/Resume", key=f"start_{game_id}"):
                    update_live_game_cells(game_id, {
                        "timer_running": "TRUE",
                        "last_timer_ts": str(time.time()),
                        "status": "live",
                        "updated_at": now_et_iso(),
                    })
                    st.rerun()
            with b2:
                if st.button("Pause", key=f"pause_{game_id}"):
                    update_live_game_cells(game_id, {
                        "timer_running": "FALSE",
                        "status": "paused",
                        "updated_at": now_et_iso(),
                    })
                    st.rerun()
            with b3:
                if st.button("Reset Period", key=f"reset_{game_id}"):
                    update_live_game_cells(game_id, {
                        "timer_running": "FALSE",
                        "timer_remaining_sec": str(period_sec),
                        "last_timer_ts": "",
                        "status": "paused",
                        "updated_at": now_et_iso(),
                    })
                    st.rerun()

            st.caption("Note: 'stopped clock' is not enforced automatically—use Pause as needed.")

    # SCOREBOARD + score buttons
    score_a = safe_int(game.get("score_a", "0"), 0)
    score_b = safe_int(game.get("score_b", "0"), 0)

    with colT2:
        st.markdown("### Scoreboard")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"#### {side_a_label}")
            st.markdown(f"# {score_a}")
            if sport == "Basketball":
                b = st.columns(4)
                if b[0].button("+1", key=f"a1_{game_id}"): update_live_game_cells(game_id, {"score_a": str(score_a+1), "updated_at": now_et_iso()}); st.rerun()
                if b[1].button("+2", key=f"a2_{game_id}"): update_live_game_cells(game_id, {"score_a": str(score_a+2), "updated_at": now_et_iso()}); st.rerun()
                if b[2].button("+3", key=f"a3_{game_id}"): update_live_game_cells(game_id, {"score_a": str(score_a+3), "updated_at": now_et_iso()}); st.rerun()
                if b[3].button("-1", key=f"am_{game_id}"): update_live_game_cells(game_id, {"score_a": str(max(0,score_a-1)), "updated_at": now_et_iso()}); st.rerun()
            else:
                b = st.columns(2)
                if b[0].button("+1", key=f"ap_{game_id}"): update_live_game_cells(game_id, {"score_a": str(score_a+1), "updated_at": now_et_iso()}); st.rerun()
                if b[1].button("-1", key=f"am1_{game_id}"): update_live_game_cells(game_id, {"score_a": str(max(0,score_a-1)), "updated_at": now_et_iso()}); st.rerun()

        with s2:
            st.markdown(f"#### {side_b_label}")
            st.markdown(f"# {score_b}")
            if sport == "Basketball":
                b = st.columns(4)
                if b[0].button("+1", key=f"b1_{game_id}"): update_live_game_cells(game_id, {"score_b": str(score_b+1), "updated_at": now_et_iso()}); st.rerun()
                if b[1].button("+2", key=f"b2_{game_id}"): update_live_game_cells(game_id, {"score_b": str(score_b+2), "updated_at": now_et_iso()}); st.rerun()
                if b[2].button("+3", key=f"b3_{game_id}"): update_live_game_cells(game_id, {"score_b": str(score_b+3), "updated_at": now_et_iso()}); st.rerun()
                if b[3].button("-1", key=f"bm_{game_id}"): update_live_game_cells(game_id, {"score_b": str(max(0,score_b-1)), "updated_at": now_et_iso()}); st.rerun()
            else:
                b = st.columns(2)
                if b[0].button("+1", key=f"bp_{game_id}"): update_live_game_cells(game_id, {"score_b": str(score_b+1), "updated_at": now_et_iso()}); st.rerun()
                if b[1].button("-1", key=f"bm1_{game_id}"): update_live_game_cells(game_id, {"score_b": str(max(0,score_b-1)), "updated_at": now_et_iso()}); st.rerun()

    st.markdown("---")
    st.markdown("### Live Stats")

    # Players = union of roster from involved teams
    roster = roster_df(lk)
    if roster.empty:
        st.warning("Roster missing for this league.")
        return

    teams_in_game = [game["team_a"], game["team_b"]]
    if match_type == "2v2":
        if str(game.get("team_a2","")).strip(): teams_in_game.append(game["team_a2"])
        if str(game.get("team_b2","")).strip(): teams_in_game.append(game["team_b2"])

    roster = roster[roster["team_name"].isin(teams_in_game)].copy()
    roster["player_name"] = roster["first_name"].fillna("") + " " + roster["last_name"].fillna("")
    roster["player_id"] = roster["player_id"].astype(str)

    stat_fields = SPORT_STATS.get(sport, [])
    if not stat_fields:
        st.info("No stats configured for this sport.")
        return

    # Quick feedback message
    if "last_stat_msg" not in st.session_state:
        st.session_state["last_stat_msg"] = ""

    msg = st.session_state.get("last_stat_msg", "")
    if msg:
        st.success(msg)
        # clear after showing once
        st.session_state["last_stat_msg"] = ""

    # Split by side
    sideA_teams = [game["team_a"]] + ([game["team_a2"]] if match_type == "2v2" and str(game.get("team_a2","")).strip() else [])
    sideB_teams = [game["team_b"]] + ([game["team_b2"]] if match_type == "2v2" and str(game.get("team_b2","")).strip() else [])

    left, right = st.columns(2)

    def player_block(df_side: pd.DataFrame, side_title: str):
        st.markdown(f"#### {side_title}")
        if df_side.empty:
            st.info("No players found.")
            return
        # Softball simple order: sort by player_id as stable default; we can enhance later with saved batting order
        df_side = df_side.sort_values(["team_name", "player_name"]).reset_index(drop=True)

        for _, r in df_side.iterrows():
            p_name = r["player_name"].strip()
            team_name = r["team_name"]
            pid = r["player_id"]

            st.write(f"**{p_name}** ({team_name})")
            cols = st.columns(len(stat_fields))
            for i, sf in enumerate(stat_fields):
                lab = STAT_LABELS.get(sf, sf.upper())
                if cols[i].button(f"+1 {lab}", key=f"{game_id}_{pid}_{sf}_{side_title}"):
                    upsert_live_stat(
                        game_id=game_id,
                        league_key=lk,
                        sport=sport,
                        level=level,
                        team_name=team_name,
                        player_id=pid,
                        player_name=p_name,
                        stat_type=sf,
                        delta=1,
                    )
                    st.session_state["last_stat_msg"] = f"{p_name}: +1 {lab}"
                    st.rerun()

    with left:
        player_block(roster[roster["team_name"].isin(sideA_teams)], side_a_label)
    with right:
        player_block(roster[roster["team_name"].isin(sideB_teams)], side_b_label)

    st.markdown("---")
    st.markdown("### Live Stat Summary (updates after every click)")
    ls = live_stats_for_game(game_id)
    if ls.empty:
        st.info("No stats yet.")
    else:
        # pivot into a clean table
        pivot = ls.pivot_table(index=["team_name", "player_name"], columns="stat_type", values="value", aggfunc="sum", fill_value=0).reset_index()
        st.dataframe(pivot, use_container_width=True)

    st.markdown("---")
    st.markdown("### Finish Game")

    cF1, cF2, cF3 = st.columns([2, 2, 3])
    with cF1:
        if st.button("Next Period", key=f"next_period_{game_id}"):
            if periods > 0 and period_idx < periods:
                update_live_game_cells(game_id, {
                    "timer_period_index": str(period_idx + 1),
                    "timer_remaining_sec": str(period_sec),
                    "timer_running": "FALSE",
                    "last_timer_ts": "",
                    "updated_at": now_et_iso(),
                    "status": "paused",
                })
                st.rerun()
            else:
                st.info("Already at last period, or no timer set.")
    with cF2:
        if st.button("Finalize Game & Save", key=f"finalize_{game_id}"):
            # compute points
            pa, pb = points_for_result(lk, sport, level, score_a, score_b)

            # append game to Games
            append_row("Games", [
                game_id,
                now_et_iso(),
                lk,
                sport,
                level,
                match_type,
                game["team_a"],
                game.get("team_a2",""),
                game["team_b"],
                game.get("team_b2",""),
                str(score_a),
                str(score_b),
                str(pa),
                str(pb),
                now_et_iso(),
            ])

            # move live stats -> Stats
            ls2 = live_stats_for_game(game_id)
            if not ls2.empty:
                for _, r in ls2.iterrows():
                    append_row("Stats", [
                        r["game_id"], r["league_key"], r["sport"], r["level"],
                        r["team_name"], r["player_id"], r["player_name"],
                        r["stat_type"], str(r["value"]), now_et_iso()
                    ])

            # mark live game finished
            update_live_game_cells(game_id, {"status": "finished", "timer_running": "FALSE", "updated_at": now_et_iso()})
            st.success("Finalized! Saved to Games + Stats.")
            st.session_state["active_game_id"] = None
            st.rerun()
    with cF3:
        if st.button("Cancel / End Live Game (no save)", key=f"cancel_{game_id}"):
            update_live_game_cells(game_id, {"status": "cancelled", "timer_running": "FALSE", "updated_at": now_et_iso()})
            st.warning("Cancelled. Nothing saved to Games/Stats.")
            st.session_state["active_game_id"] = None
            st.rerun()

def page_standings(league_name: str):
    lk = league_key_from_name(league_name)
    st.header(f"Standings – {league_name}")
    include_non = st.checkbox("Include non-game points", value=True)
    df = standings_for_league(lk, include_non_game=include_non)
    st.dataframe(df, use_container_width=True)

def page_leaderboards(league_name: str):
    lk = league_key_from_name(league_name)
    st.header(f"Leaderboards – {league_name}")

    stats = df_from_ws("Stats")
    if stats.empty:
        st.info("No finalized stats yet.")
        return

    sport_opts = ["All sports"] + sorted(stats[stats["league_key"] == lk]["sport"].dropna().unique().tolist())
    stat_opts = sorted(stats[stats["league_key"] == lk]["stat_type"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        sport = st.selectbox("Sport", sport_opts)
    with col2:
        stat_type = st.selectbox("Stat Category", stat_opts) if stat_opts else st.selectbox("Stat Category", ["(none)"])

    if stat_type == "(none)":
        st.info("No stats categories found yet.")
        return

    lb = leaderboard(lk, sport, stat_type)
    if lb.empty:
        st.info("No data for that filter yet.")
        return
    st.dataframe(lb, use_container_width=True)

def page_non_game_points(league_name: str):
    lk = league_key_from_name(league_name)
    st.header(f"Non-Game Points – {league_name}")

    roster = roster_df(lk)
    if roster.empty:
        st.warning("Upload roster first.")
        return
    teams = sorted(roster["team_name"].dropna().unique().tolist())
    if not teams:
        st.warning("No teams found in roster.")
        return

    df = df_from_ws("NonGamePoints")
    st.subheader("Add Non-Game Points")
    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Date", value=datetime.now(ET).date())
    with col2:
        team = st.selectbox("Team", teams)
    with col3:
        points = st.number_input("Points", step=1, value=1)

    category = st.selectbox("Category", NON_GAME_CATEGORIES)
    reason = st.text_input("Reason/details")

    if st.button("Add Non-Game Points"):
        new_id = int(time.time())
        append_row("NonGamePoints", [
            str(new_id),
            date.isoformat(),
            lk,
            category,
            reason,
            team,
            str(points),
            now_et_iso(),
        ])
        st.success("Added.")
        st.rerun()

    st.markdown("---")
    st.subheader("Existing Non-Game Points")
    if df.empty:
        st.info("None yet.")
    else:
        df2 = df[df["league_key"] == lk].copy()
        st.dataframe(df2, use_container_width=True)

def page_display_board():
    st.header("Display Board")

    include_non = st.checkbox("Include non-game points", value=True)
    mode = st.selectbox("Mode", ["All leagues side-by-side + combined", "Single league"])

    if mode == "Single league":
        league_name = st.selectbox("League", list(LEAGUES.keys()))
        lk = league_key_from_name(league_name)
        st.subheader(f"{league_name} Standings")
        st.dataframe(standings_for_league(lk, include_non_game=include_non), use_container_width=True)
    else:
        cols = st.columns(3)
        for (name, cfg), col in zip(LEAGUES.items(), cols):
            with col:
                st.subheader(name)
                st.dataframe(
                    standings_for_league(cfg["key"], include_non_game=include_non)[["Team", "Wins", "Losses", "Ties", "Total Points"]],
                    use_container_width=True
                )
        st.markdown("---")
        st.subheader("Combined (Age Weighted)")
        st.dataframe(combined_standings(include_non_game=include_non), use_container_width=True)

def page_highlights(league_name: str):
    # IMPORTANT NOTE: file uploads do not persist reliably on Streamlit Community Cloud unless stored elsewhere.
    # Here we store a "link" to the highlight (Drive/YouTube/etc). Upload-to-Drive can be added next.
    lk = league_key_from_name(league_name)
    st.header(f"Highlights – {league_name}")

    st.subheader("Add Highlight (link)")
    date = st.date_input("Date", value=datetime.now(ET).date(), key=f"hdate_{lk}")
    title = st.text_input("Title", key=f"htitle_{lk}")
    desc = st.text_area("Description", key=f"hdesc_{lk}")
    link = st.text_input("Video link (Drive share link / YouTube / etc.)", key=f"hlink_{lk}")

    if st.button("Save Highlight"):
        new_id = int(time.time())
        append_row("Highlights", [str(new_id), date.isoformat(), lk, title, desc, link, now_et_iso()])
        st.success("Saved.")
        st.rerun()

    st.markdown("---")
    st.subheader("Highlights List")
    df = df_from_ws("Highlights")
    if df.empty:
        st.info("No highlights yet.")
        return
    df = df[df["league_key"] == lk].copy()
    st.dataframe(df[["id", "date", "title", "description", "link"]], use_container_width=True)

def main():
    st.sidebar.title("Crest League Manager (Multi-Game)")

    # logo optional
    if "logo-header-2.png" in [p.name for p in []]:
        pass

    current_league = st.sidebar.selectbox("League", list(LEAGUES.keys()))
    page = st.sidebar.radio("Go to", [
        "Setup",
        "Live Games (Create/Open)",
        "Run Live Game",
        "Standings",
        "Leaderboards",
        "Non-Game Points",
        "Highlights",
        "Display Board",
    ])

    # Make sure tabs exist (first run safety)
    # We won't overwrite if you already made them, but we will ensure roster tabs exist.
    # (Assumes you already created all headers exactly.)
    # If a worksheet name is wrong, gspread will error, which helps you catch typos.
    if page == "Setup":
        page_setup(current_league)
    elif page == "Live Games (Create/Open)":
        page_live_games_home(current_league)
    elif page == "Run Live Game":
        page_run_live_game(current_league)
    elif page == "Standings":
        page_standings(current_league)
    elif page == "Leaderboards":
        page_leaderboards(current_league)
    elif page == "Non-Game Points":
        page_non_game_points(current_league)
    elif page == "Highlights":
        page_highlights(current_league)
    elif page == "Display Board":
        page_display_board()

if __name__ == "__main__":
    main()
