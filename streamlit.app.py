import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =========================================================
# Bauercrest League Manager — Phase 2B Live Game Engine
# =========================================================
# Key architecture:
# - Live game state = st.session_state (per user/device)
# - Google Sheets = persistence only (rosters, final games, final stats)
# - NO per-second sheet reads, no sheet reads on stat button presses
# =========================================================

ET = pytz.timezone("America/New_York")

# ---------------------------
# BRANDING / COLORS (easy)
# ---------------------------
BC_PRIMARY = "#0B1E3B"     # Bauercrest-ish navy
BC_ACCENT = "#C9A227"      # gold-ish accent
BC_BG = "#F6F7FB"

LOGO_PATH = "logo-header-2.png"  # put in repo root

# ---------------------------
# Google Sheets setup
# ---------------------------
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SPREADSHEET_ID = st.secrets.get("spreadsheet_id", "")

# Expected sheet headers
HEADERS = {
    "Roster_Sophomore": ["player_id", "first_name", "last_name", "team_name", "bunk", "bat_order"],
    "Roster_Junior":    ["player_id", "first_name", "last_name", "team_name", "bunk", "bat_order"],
    "Roster_Senior":    ["player_id", "first_name", "last_name", "team_name", "bunk", "bat_order"],
    "Games":            ["game_id", "league_key", "sport", "level", "mode",
                         "team_a1", "team_a2", "team_b1", "team_b2",
                         "score_a", "score_b", "points_a", "points_b",
                         "played_at", "saved_at", "notes"],
    "GameStats":        ["game_id", "league_key", "sport", "level",
                         "player_id", "team_name", "stat_key", "value"],
    "NonGamePoints":    ["entry_id", "league_key", "team_name", "category", "reason", "points", "created_at"],
    "Highlights":       ["highlight_id", "league_key", "filename", "uploaded_at"],
}

LEAGUES = {
    "Sophomore League": "sophomore",
    "Junior League": "junior",
    "Senior League": "senior",
}

LEAGUE_TO_ROSTER_SHEET = {
    "sophomore": "Roster_Sophomore",
    "junior": "Roster_Junior",
    "senior": "Roster_Senior",
}

# ---------------------------
# Sports + live stats per sport
# ---------------------------
SPORTS = [
    "Basketball",
    "Softball",
    "Football",
    "Hockey",
    "Soccer",
    "Kickball",
    "Euro",
    "Speedball",
]

# What stat buttons appear in live mode
SPORT_STATS = {
    "Basketball": [("points", "PTS"), ("assists", "AST"), ("rebounds", "REB"), ("steals", "STL"), ("blocks", "BLK")],
    "Hockey":     [("goals", "G"), ("assists", "A"), ("saves", "SV")],
    "Football":   [("tds", "TD"), ("catches", "REC"), ("interceptions", "INT")],
    "Softball":   [("hits", "H"), ("doubles", "2B"), ("triples", "3B"), ("home_runs", "HR"), ("rbis", "RBI"), ("runs", "R")],
    "Soccer":     [("goals", "G"), ("assists", "A"), ("saves", "SV")],
    "Kickball":   [("runs", "R"), ("hits", "H")],
    "Euro":       [("points", "PTS"), ("assists", "AST")],
    "Speedball":  [("points", "PTS"), ("assists", "AST"), ("goals", "G")],
}

# Timer presets by sport
# (You asked for MORE options. Add more anytime.)
TIMER_PRESETS = {
    "Basketball": [
        ("15:00 halves (running)", 30*60, "halves"),
        ("15:00 halves (stop clock)", 30*60, "halves"),
        ("20:00 halves (running)", 40*60, "halves"),
        ("20:00 halves (stop clock)", 40*60, "halves"),
        ("No timer (manual)", 0, "none"),
    ],
    "Hockey": [
        ("12:00 periods (running)", 36*60, "periods"),
        ("15:00 periods (running)", 45*60, "periods"),
        ("10:00 periods (running)", 30*60, "periods"),
        ("No timer (manual)", 0, "none"),
    ],
    "Football": [
        ("20:00 halves (running)", 40*60, "halves"),
        ("15:00 halves (running)", 30*60, "halves"),
        ("No timer (manual)", 0, "none"),
    ],
    "Softball": [
        ("No timer (innings style)", 0, "none"),
    ],
    "Soccer": [
        ("20:00 halves (running)", 40*60, "halves"),
        ("15:00 halves (running)", 30*60, "halves"),
        ("No timer (manual)", 0, "none"),
    ],
    "Kickball": [
        ("No timer (innings style)", 0, "none"),
        ("15:00 halves (running)", 30*60, "halves"),
    ],
    "Euro": [
        ("15:00 halves (running)", 30*60, "halves"),
        ("No timer (manual)", 0, "none"),
    ],
    "Speedball": [
        ("15:00 halves (running)", 30*60, "halves"),
        ("No timer (manual)", 0, "none"),
    ],
}

# Points logic — you can refine later.
# Key: Senior A Softball = 50
POINTS_TABLE = {
    "senior": {
        "Softball": {"A": 50, "B": 45, "C": 40, "D": 35},
        "Football": {"A": 40, "B": 35, "C": 30, "D": 25},
        "Basketball": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Hockey": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Soccer": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Euro": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Speedball": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Kickball": {"A": 25, "B": 20, "C": 15, "D": 10},
    },
    "junior": {
        "Softball": {"A": 42, "B": 37, "C": 32, "D": 27},
        "Football": {"A": 34, "B": 29, "C": 24, "D": 19},
        "Basketball": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Hockey": {"A": 30, "B": 25, "C": 20, "D": 15},
        "Soccer": {"A": 26, "B": 21, "C": 16, "D": 11},
        "Euro": {"A": 26, "B": 21, "C": 16, "D": 11},
        "Speedball": {"A": 26, "B": 21, "C": 16, "D": 11},
        "Kickball": {"A": 20, "B": 15, "C": 10, "D": 8},
    },
    "sophomore": {
        "Softball": {"A": 35, "B": 30, "C": 25, "D": 20},
        "Football": {"A": 28, "B": 23, "C": 18, "D": 13},
        "Basketball": {"A": 24, "B": 19, "C": 14, "D": 10},
        "Hockey": {"A": 24, "B": 19, "C": 14, "D": 10},
        "Soccer": {"A": 20, "B": 16, "C": 12, "D": 8},
        "Euro": {"A": 20, "B": 16, "C": 12, "D": 8},
        "Speedball": {"A": 20, "B": 16, "C": 12, "D": 8},
        "Kickball": {"A": 16, "B": 12, "C": 8, "D": 6},
    },
}

# ---------------------------
# Helpers
# ---------------------------
def now_et() -> datetime:
    return datetime.now(ET)

def now_et_iso() -> str:
    return now_et().strftime("%Y-%m-%d %I:%M:%S %p ET")

def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def points_for_result(league_key: str, sport: str, level: str, score_a: int, score_b: int) -> Tuple[float, float]:
    base = POINTS_TABLE.get(league_key, {}).get(sport, {}).get(level, 0)
    if score_a > score_b:
        return float(base), 0.0
    if score_b > score_a:
        return 0.0, float(base)
    return float(base) / 2.0, float(base) / 2.0

def toast_ok(msg: str):
    # streamlit versions differ; try toast, fallback to success
    if hasattr(st, "toast"):
        st.toast(msg, icon="✅")
    else:
        st.success(msg)

# ---------------------------
# Styling
# ---------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {BC_BG};
        }}
        .bc-title {{
            font-size: 2.1rem;
            font-weight: 800;
            color: {BC_PRIMARY};
            margin: 0.25rem 0 0.75rem 0;
        }}
        .scoreboard {{
            background: white;
            border: 2px solid {BC_PRIMARY};
            border-radius: 16px;
            padding: 18px 18px 10px 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }}
        .sb-time {{
            font-size: 4.0rem;
            font-weight: 900;
            color: {BC_PRIMARY};
            text-align: center;
            letter-spacing: 2px;
            margin: 6px 0 10px 0;
        }}
        .sb-sub {{
            text-align: center;
            color: #666;
            margin-bottom: 14px;
            font-weight: 600;
        }}
        .sb-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
        }}
        .sb-team {{
            flex: 1;
            background: #f7f9ff;
            border: 1px solid #e6e9f5;
            border-radius: 14px;
            padding: 12px;
        }}
        .sb-teamname {{
            font-size: 1.1rem;
            font-weight: 800;
            color: {BC_PRIMARY};
            margin-bottom: 4px;
        }}
        .sb-score {{
            font-size: 3.4rem;
            font-weight: 900;
            color: {BC_PRIMARY};
            line-height: 1;
        }}
        .pill {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: {BC_PRIMARY};
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            margin-top: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Google Sheets (cached reads, controlled writes)
# ---------------------------
@st.cache_resource
def get_gspread_client():
    raw = st.secrets.get("gcp_service_account", "")
    if not raw or not SPREADSHEET_ID:
        raise RuntimeError("Missing Streamlit secrets: spreadsheet_id and/or gcp_service_account.")
    creds = Credentials.from_service_account_info(eval(raw) if raw.strip().startswith("{") else __import__("json").loads(raw), scopes=SCOPE)
    return gspread.authorize(creds).open_by_key(SPREADSHEET_ID)

def ws(name: str):
    return get_gspread_client().worksheet(name)

def ensure_headers(sheet_name: str):
    expected = HEADERS[sheet_name]
    w = ws(sheet_name)
    current = w.row_values(1)
    if current != expected:
        w.clear()
        w.append_row(expected, value_input_option="RAW")

@st.cache_data(ttl=30)
def df_from_ws_cached(sheet_name: str) -> pd.DataFrame:
    ensure_headers(sheet_name)
    w = ws(sheet_name)
    values = w.get_all_values()
    if not values or len(values) == 1:
        return pd.DataFrame(columns=HEADERS[sheet_name])
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

def append_row(sheet_name: str, row: List):
    ensure_headers(sheet_name)
    ws(sheet_name).append_row(row, value_input_option="USER_ENTERED")

# ---------------------------
# Session state: live game engine
# ---------------------------
def init_state():
    if "live_games" not in st.session_state:
        st.session_state.live_games = {}  # game_id -> dict
    if "active_game_id" not in st.session_state:
        st.session_state.active_game_id = None
    if "tick" not in st.session_state:
        st.session_state.tick = 0

def make_game_id(league_key: str, sport: str, level: str) -> str:
    stamp = now_et().strftime("%Y%m%d-%H%M%S")
    return f"{league_key.upper()}-{sport[:3].upper()}-{level}-{stamp}-{uuid.uuid4().hex[:6]}"

def format_mm_ss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"

def get_timer_remaining(game: Dict) -> int:
    # Timer is derived, not decremented.
    if game["duration_seconds"] <= 0:
        return 0
    if not game["running"]:
        return int(game["remaining_at_pause"])
    now_ts = time.time()
    elapsed = now_ts - game["start_ts"]
    remaining = int(max(0, game["duration_seconds"] - elapsed))
    return remaining

def pause_timer(game: Dict):
    if not game["running"]:
        return
    remaining = get_timer_remaining(game)
    game["running"] = False
    game["remaining_at_pause"] = remaining

def start_or_resume_timer(game: Dict):
    # When starting/resuming, we set start_ts so that duration - elapsed = remaining_at_pause
    if game["duration_seconds"] <= 0:
        game["running"] = False
        return
    if game["running"]:
        return
    remaining = int(game.get("remaining_at_pause", game["duration_seconds"]))
    game["running"] = True
    game["start_ts"] = time.time() - (game["duration_seconds"] - remaining)

def reset_timer(game: Dict):
    game["running"] = False
    game["remaining_at_pause"] = int(game["duration_seconds"])
    game["start_ts"] = time.time()

def bump_stat(game: Dict, player_id: str, stat_key: str, delta: int = 1):
    stats = game["player_stats"]  # player_id -> stat_key -> value
    if player_id not in stats:
        stats[player_id] = {}
    stats[player_id][stat_key] = int(stats[player_id].get(stat_key, 0)) + int(delta)

def bump_score(game: Dict, side: str, delta: int):
    if side == "A":
        game["score_a"] = max(0, int(game["score_a"]) + int(delta))
    else:
        game["score_b"] = max(0, int(game["score_b"]) + int(delta))

def roster_for_league(league_key: str) -> pd.DataFrame:
    sheet = LEAGUE_TO_ROSTER_SHEET[league_key]
    df = df_from_ws_cached(sheet).copy()
    if df.empty:
        return df
    # normalize
    for c in ["player_id", "first_name", "last_name", "team_name", "bunk", "bat_order"]:
        if c not in df.columns:
            df[c] = ""
    # bat_order optional
    df["bat_order"] = df["bat_order"].apply(lambda x: safe_int(x, 0))
    return df

# ---------------------------
# Pages
# ---------------------------
def page_setup(current_league_key: str):
    st.markdown('<div class="bc-title">Setup</div>', unsafe_allow_html=True)
    st.write("Upload and confirm rosters (Sophomore / Junior / Senior).")

    st.info("This app stores rosters in Google Sheets. Live games do NOT hit Sheets every second (prevents quota crashes).")

    # roster display
    df = roster_for_league(current_league_key)
    if df.empty:
        st.warning("No roster found for this league yet.")
    else:
        st.subheader("Current Roster")
        st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Upload roster CSV for this league")

    st.caption("CSV must contain: player_id, first_name, last_name, team_name, bunk (bat_order optional)")
    uploaded = st.file_uploader("Upload roster CSV", type=["csv"], key=f"roster_upload_{current_league_key}")

    if uploaded:
        df_new = pd.read_csv(uploaded)
        required = ["player_id", "first_name", "last_name", "team_name", "bunk"]
        missing = [c for c in required if c not in df_new.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return
        if "bat_order" not in df_new.columns:
            df_new["bat_order"] = 0

        # Write roster to sheet (clear + header + rows)
        sheet_name = LEAGUE_TO_ROSTER_SHEET[current_league_key]
        ensure_headers(sheet_name)
        w = ws(sheet_name)
        w.clear()
        w.append_row(HEADERS[sheet_name], value_input_option="RAW")
        rows = df_new[HEADERS[sheet_name]].fillna("").astype(str).values.tolist()
        if rows:
            w.append_rows(rows, value_input_option="USER_ENTERED")

        st.success("Roster uploaded and saved.")
        st.cache_data.clear()

def page_live_games_home(current_league_key: str):
    st.markdown('<div class="bc-title">Live Games (Create / Open)</div>', unsafe_allow_html=True)

    # Create game
    st.subheader("Create a new live game")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sport = st.selectbox("Sport", SPORTS, key="create_sport")
    with c2:
        level = st.selectbox("Level", ["A", "B", "C", "D"], key="create_level")
    with c3:
        mode = st.selectbox("Mode", ["1 Team vs 1 Team", "2 Teams vs 2 Teams"], key="create_mode")

    roster = roster_for_league(current_league_key)
    teams = sorted([t for t in roster["team_name"].dropna().unique().tolist() if str(t).strip()]) if not roster.empty else []

    if not teams:
        st.warning("No teams found yet. Upload a roster first.")
        return

    if mode == "1 Team vs 1 Team":
        t1, t2 = st.columns(2)
        with t1:
            team_a1 = st.selectbox("Team A", teams, key="team_a1")
        with t2:
            team_b1 = st.selectbox("Team B", [t for t in teams if t != team_a1], key="team_b1")
        team_a2 = ""
        team_b2 = ""
    else:
        st.caption("Pick two teams per side. Points will be split evenly across the two teams when saved.")
        a1, a2, b1, b2 = st.columns(4)
        with a1:
            team_a1 = st.selectbox("A1", teams, key="team_a1_2v2")
        with a2:
            team_a2 = st.selectbox("A2", [t for t in teams if t != team_a1], key="team_a2_2v2")
        with b1:
            team_b1 = st.selectbox("B1", [t for t in teams if t not in [team_a1, team_a2]], key="team_b1_2v2")
        with b2:
            team_b2 = st.selectbox("B2", [t for t in teams if t not in [team_a1, team_a2, team_b1]], key="team_b2_2v2")

    presets = TIMER_PRESETS.get(sport, [("No timer (manual)", 0, "none")])
    preset_label = st.selectbox("Timer preset", [p[0] for p in presets], key="timer_preset")
    preset = [p for p in presets if p[0] == preset_label][0]
    duration_seconds = int(preset[1])
    structure = preset[2]

    notes = st.text_input("Notes (optional)", key="create_notes")

    if st.button("Create Live Game", type="primary"):
        game_id = make_game_id(current_league_key, sport, level)
        # Prepare players for this game (active lineup default = only teams involved)
        if mode == "1 Team vs 1 Team":
            eligible = roster[roster["team_name"].isin([team_a1, team_b1])].copy()
        else:
            eligible = roster[roster["team_name"].isin([team_a1, team_a2, team_b1, team_b2])].copy()

        eligible["full_name"] = eligible["first_name"].astype(str) + " " + eligible["last_name"].astype(str)

        st.session_state.live_games[game_id] = {
            "game_id": game_id,
            "league_key": current_league_key,
            "sport": sport,
            "level": level,
            "mode": "2v2" if mode != "1 Team vs 1 Team" else "1v1",
            "team_a1": team_a1,
            "team_a2": team_a2,
            "team_b1": team_b1,
            "team_b2": team_b2,
            "notes": notes,
            "created_at": now_et_iso(),
            # scoreboard
            "score_a": 0,
            "score_b": 0,
            # timer
            "duration_seconds": duration_seconds,
            "structure": structure,
            "running": False,
            "start_ts": time.time(),
            "remaining_at_pause": duration_seconds,
            # lineup & stats
            "eligible_players_df": eligible[["player_id", "full_name", "team_name", "bat_order"]].copy(),
            "active_player_ids": eligible["player_id"].tolist(),  # default all eligible
            "player_stats": {},  # player_id -> stat_key -> value
        }
        st.session_state.active_game_id = game_id
        toast_ok(f"Live game created: {game_id}")
        st.experimental_rerun()

    st.divider()
    st.subheader("Open an existing live game (this device/session)")

    if not st.session_state.live_games:
        st.info("No live games created on this device yet.")
        return

    game_ids = list(st.session_state.live_games.keys())
    pick = st.selectbox("Select game", game_ids, key="open_game_id")
    if st.button("Open Live Game"):
        st.session_state.active_game_id = pick
        st.experimental_rerun()

def page_run_live_game():
    gid = st.session_state.active_game_id
    if not gid or gid not in st.session_state.live_games:
        st.warning("No active live game selected.")
        return

    game = st.session_state.live_games[gid]
    sport = game["sport"]

    st.markdown('<div class="bc-title">Run Live Game</div>', unsafe_allow_html=True)
    st.caption(f"Game ID: {gid} • League: {game['league_key']} • Sport: {game['sport']} • Level: {game['level']}")

    # AUTO-RERUN every second ONLY while timer is running
    # NOTE: this triggers a rerun; we MUST NOT hit Sheets in this page.
    if game["running"] and game["duration_seconds"] > 0:
        time.sleep(1)
        st.session_state.tick += 1
        st.experimental_rerun()

    # Scoreboard block
    remaining = get_timer_remaining(game)
    time_str = "—" if game["duration_seconds"] <= 0 else format_mm_ss(remaining)

    st.markdown('<div class="scoreboard">', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-time">{time_str}</div>', unsafe_allow_html=True)

    sub = f"{game['sport']} • {game['level']} • {('2 Teams vs 2 Teams' if game['mode']=='2v2' else '1 vs 1')}"
    st.markdown(f'<div class="sb-sub">{sub}</div>', unsafe_allow_html=True)

    team_a_label = game["team_a1"] + (f" + {game['team_a2']}" if game["mode"] == "2v2" and game["team_a2"] else "")
    team_b_label = game["team_b1"] + (f" + {game['team_b2']}" if game["mode"] == "2v2" and game["team_b2"] else "")

    st.markdown(
        f"""
        <div class="sb-row">
          <div class="sb-team">
            <div class="sb-teamname">{team_a_label}</div>
            <div class="sb-score">{game["score_a"]}</div>
            <div class="pill">TEAM A</div>
          </div>
          <div style="width:12px;"></div>
          <div class="sb-team">
            <div class="sb-teamname">{team_b_label}</div>
            <div class="sb-score">{game["score_b"]}</div>
            <div class="pill">TEAM B</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("▶ Start / Resume", use_container_width=True):
            start_or_resume_timer(game)
            toast_ok("Timer started.")
            st.experimental_rerun()
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            pause_timer(game)
            toast_ok("Timer paused.")
            st.experimental_rerun()
    with c3:
        if st.button("⟲ Reset Timer", use_container_width=True):
            reset_timer(game)
            toast_ok("Timer reset.")
            st.experimental_rerun()
    with c4:
        if st.button("🧹 Reset Scores", use_container_width=True):
            game["score_a"] = 0
            game["score_b"] = 0
            toast_ok("Scores reset.")
            st.experimental_rerun()

    st.divider()

    st.subheader("Score Buttons")
    sb1, sb2 = st.columns(2)

    # Basketball common scoring buttons; others use +/-1
    if sport == "Basketball":
        with sb1:
            st.write("Team A")
            r1, r2, r3, r4 = st.columns(4)
            if r1.button("+1", use_container_width=True): bump_score(game, "A", 1); toast_ok("Team A +1"); st.experimental_rerun()
            if r2.button("+2", use_container_width=True): bump_score(game, "A", 2); toast_ok("Team A +2"); st.experimental_rerun()
            if r3.button("+3", use_container_width=True): bump_score(game, "A", 3); toast_ok("Team A +3"); st.experimental_rerun()
            if r4.button("-1", use_container_width=True): bump_score(game, "A", -1); toast_ok("Team A -1"); st.experimental_rerun()
        with sb2:
            st.write("Team B")
            r1, r2, r3, r4 = st.columns(4)
            if r1.button("+1 ", use_container_width=True): bump_score(game, "B", 1); toast_ok("Team B +1"); st.experimental_rerun()
            if r2.button("+2 ", use_container_width=True): bump_score(game, "B", 2); toast_ok("Team B +2"); st.experimental_rerun()
            if r3.button("+3 ", use_container_width=True): bump_score(game, "B", 3); toast_ok("Team B +3"); st.experimental_rerun()
            if r4.button("-1 ", use_container_width=True): bump_score(game, "B", -1); toast_ok("Team B -1"); st.experimental_rerun()
    else:
        with sb1:
            st.write("Team A")
            r1, r2 = st.columns(2)
            if r1.button("+1", use_container_width=True): bump_score(game, "A", 1); toast_ok("Team A +1"); st.experimental_rerun()
            if r2.button("-1", use_container_width=True): bump_score(game, "A", -1); toast_ok("Team A -1"); st.experimental_rerun()
        with sb2:
            st.write("Team B")
            r1, r2 = st.columns(2)
            if r1.button("+1 ", use_container_width=True): bump_score(game, "B", 1); toast_ok("Team B +1"); st.experimental_rerun()
            if r2.button("-1 ", use_container_width=True): bump_score(game, "B", -1); toast_ok("Team B -1"); st.experimental_rerun()

    st.divider()

    # Lineup control (remove players not in this game)
    st.subheader("Active Lineup (who should appear for stats)")
    eligible_df = game["eligible_players_df"].copy()
    eligible_df = eligible_df.sort_values(["team_name", "full_name"])
    options = eligible_df["player_id"].tolist()
    labels = {
        r["player_id"]: f"{r['full_name']} — {r['team_name']}"
        for _, r in eligible_df.iterrows()
    }
    current_active = game["active_player_ids"]
    picked = st.multiselect(
        "Pick the players who are actually playing in this game",
        options=options,
        default=current_active,
        format_func=lambda pid: labels.get(pid, pid),
    )
    game["active_player_ids"] = picked

    st.divider()

    # Live stats
    st.subheader("Live Player Stats (tap buttons — no saving each click)")
    stat_buttons = SPORT_STATS.get(sport, [("points", "PTS")])

    # Build active players table (respect bat_order for softball if present)
    active_df = eligible_df[eligible_df["player_id"].isin(game["active_player_ids"])].copy()
    if sport == "Softball":
        # If bat_order is set, sort by it; else keep team/name
        active_df = active_df.sort_values(["bat_order", "team_name", "full_name"])
    else:
        active_df = active_df.sort_values(["team_name", "full_name"])

    if active_df.empty:
        st.warning("No active players selected.")
    else:
        # Display each player with stat buttons
        for _, row in active_df.iterrows():
            pid = row["player_id"]
            name = row["full_name"]
            team = row["team_name"]

            cols = st.columns([2.5] + [1]*len(stat_buttons))
            cols[0].markdown(f"**{name}**  \n_{team}_")

            for i, (stat_key, label) in enumerate(stat_buttons):
                if cols[i+1].button(label, key=f"{gid}_{pid}_{stat_key}"):
                    bump_stat(game, pid, stat_key, 1)
                    toast_ok(f"{name}: +1 {label}")
                    st.experimental_rerun()

    st.divider()

    # In-game stats table (updates every button press)
    st.subheader("Current Game Totals")
    rows = []
    for _, row in active_df.iterrows():
        pid = row["player_id"]
        name = row["full_name"]
        team = row["team_name"]
        stat_map = game["player_stats"].get(pid, {})
        out = {"player": name, "team": team}
        for sk, lbl in stat_buttons:
            out[lbl] = int(stat_map.get(sk, 0))
        rows.append(out)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # Finalize & Save (ONE write burst)
    st.subheader("Finalize Game (Save to Google Sheets)")
    st.warning("This is the ONLY time this live game touches Google Sheets. This prevents API quota crashes.")

    notes = st.text_area("Final notes (optional)", value=game.get("notes", ""), key=f"final_notes_{gid}")

    if st.button("✅ Finalize & Save Game", type="primary"):
        # stop timer
        pause_timer(game)

        league_key = game["league_key"]
        level = game["level"]
        sport = game["sport"]
        score_a = int(game["score_a"])
        score_b = int(game["score_b"])

        pts_a, pts_b = points_for_result(league_key, sport, level, score_a, score_b)

        played_at = now_et_iso()
        saved_at = now_et_iso()

        # For 2v2: split points across the two teams on a side
        # We store the total points for the side in Games,
        # then standings will compute per-team splits.
        append_row("Games", [
            gid, league_key, sport, level, game["mode"],
            game["team_a1"], game.get("team_a2",""),
            game["team_b1"], game.get("team_b2",""),
            score_a, score_b,
            pts_a, pts_b,
            played_at, saved_at,
            notes
        ])

        # Save stats (only active players)
        # We append one row per (player, stat_key)
        for _, row in active_df.iterrows():
            pid = row["player_id"]
            team = row["team_name"]
            stat_map = game["player_stats"].get(pid, {})
            for stat_key, _lbl in stat_buttons:
                val = int(stat_map.get(stat_key, 0))
                if val != 0:
                    append_row("GameStats", [gid, league_key, sport, level, pid, team, stat_key, val])

        toast_ok("Game saved to Google Sheets.")
        st.cache_data.clear()

def page_standings(current_league_key: str):
    st.markdown('<div class="bc-title">Standings</div>', unsafe_allow_html=True)

    # Read-only and cached
    games = df_from_ws_cached("Games")
    ng = df_from_ws_cached("NonGamePoints")
    roster = roster_for_league(current_league_key)

    if roster.empty:
        st.warning("Upload roster first.")
        return

    teams = sorted([t for t in roster["team_name"].dropna().unique().tolist() if str(t).strip()])

    # Compute points by team (game points + optional non-game)
    # For 2v2, split side points evenly between the two teams on that side.
    team_points_game = {t: 0.0 for t in teams}
    team_w = {t: 0 for t in teams}
    team_l = {t: 0 for t in teams}
    team_t = {t: 0 for t in teams}

    if not games.empty:
        g = games[games["league_key"] == current_league_key].copy()
        for col in ["score_a","score_b","points_a","points_b"]:
            if col in g.columns:
                g[col] = g[col].apply(lambda x: float(x) if str(x).strip() else 0.0)

        for _, r in g.iterrows():
            mode = (r.get("mode","1v1") or "1v1").strip()
            a1 = str(r.get("team_a1","")).strip()
            a2 = str(r.get("team_a2","")).strip()
            b1 = str(r.get("team_b1","")).strip()
            b2 = str(r.get("team_b2","")).strip()
            pa = float(r.get("points_a",0))
            pb = float(r.get("points_b",0))
            sa = int(float(r.get("score_a",0)))
            sb = int(float(r.get("score_b",0)))

            a_teams = [t for t in [a1,a2] if t]
            b_teams = [t for t in [b1,b2] if t]
            if not a_teams or not b_teams:
                continue

            # split points across teams on each side
            for t in a_teams:
                if t in team_points_game:
                    team_points_game[t] += pa / max(1, len(a_teams))
            for t in b_teams:
                if t in team_points_game:
                    team_points_game[t] += pb / max(1, len(b_teams))

            # W/L/T counts (give each involved team the same W/L/T)
            if sa > sb:
                for t in a_teams:
                    if t in team_w: team_w[t] += 1
                for t in b_teams:
                    if t in team_l: team_l[t] += 1
            elif sb > sa:
                for t in b_teams:
                    if t in team_w: team_w[t] += 1
                for t in a_teams:
                    if t in team_l: team_l[t] += 1
            else:
                for t in a_teams:
                    if t in team_t: team_t[t] += 1
                for t in b_teams:
                    if t in team_t: team_t[t] += 1

    team_points_nongame = {t: 0.0 for t in teams}
    if not ng.empty:
        d = ng[ng["league_key"] == current_league_key].copy()
        if "points" in d.columns:
            d["points"] = d["points"].apply(lambda x: float(x) if str(x).strip() else 0.0)
        for _, r in d.iterrows():
            team = str(r.get("team_name","")).strip()
            pts = float(r.get("points",0))
            if team in team_points_nongame:
                team_points_nongame[team] += pts

    include_nongame = st.toggle("Include non-game points in totals", value=True)

    rows = []
    for t in teams:
        gp = team_points_game[t]
        ngp = team_points_nongame[t]
        total = gp + ngp if include_nongame else gp
        rows.append({
            "Team": t,
            "W": team_w[t],
            "L": team_l[t],
            "T": team_t[t],
            "Game Pts": round(gp, 1),
            "Non-Game Pts": round(ngp, 1),
            "Total Pts": round(total, 1),
        })

    df_out = pd.DataFrame(rows).sort_values(["Total Pts","W"], ascending=False)
    st.dataframe(df_out, use_container_width=True)

def page_leaderboards(current_league_key: str):
    st.markdown('<div class="bc-title">Leaderboards</div>', unsafe_allow_html=True)
    roster = roster_for_league(current_league_key)
    if roster.empty:
        st.warning("Upload roster first.")
        return

    stats = df_from_ws_cached("GameStats")
    if stats.empty:
        st.info("No saved stats yet.")
        return

    stats = stats[stats["league_key"] == current_league_key].copy()
    if stats.empty:
        st.info("No saved stats yet for this league.")
        return

    # Merge player names
    roster["player_id"] = roster["player_id"].astype(str)
    stats["player_id"] = stats["player_id"].astype(str)

    stats["value"] = stats["value"].apply(lambda x: safe_int(x, 0))
    merged = stats.merge(roster[["player_id","first_name","last_name","team_name"]], on="player_id", how="left")
    merged["player"] = merged["first_name"].fillna("").astype(str) + " " + merged["last_name"].fillna("").astype(str)

    sport = st.selectbox("Sport", sorted(merged["sport"].dropna().unique().tolist()))
    m = merged[merged["sport"] == sport].copy()

    if m.empty:
        st.info("No stats for that sport yet.")
        return

    # pick stat category
    stat_keys = sorted(m["stat_key"].dropna().unique().tolist())
    stat_key = st.selectbox("Stat category", stat_keys)

    m2 = m[m["stat_key"] == stat_key].copy()
    if m2.empty:
        st.info("No entries for that stat category yet.")
        return

    board = m2.groupby(["player","team_name"], as_index=False)["value"].sum()
    board = board.sort_values("value", ascending=False).head(20)
    st.dataframe(board.rename(columns={"team_name":"Team","value":"Total"}), use_container_width=True)

def page_non_game_points(current_league_key: str):
    st.markdown('<div class="bc-title">Non-Game Points</div>', unsafe_allow_html=True)
    roster = roster_for_league(current_league_key)
    if roster.empty:
        st.warning("Upload roster first.")
        return
    teams = sorted([t for t in roster["team_name"].dropna().unique().tolist() if str(t).strip()])

    cat = st.selectbox("Category", [
        "League Spirit",
        "Sportsmanship",
        "Cleanup / Organization",
        "Participation / Effort",
        "Other",
    ])
    team = st.selectbox("Team", teams)
    reason = st.text_input("Reason (or details)", value="" if cat != "Other" else "")
    pts = st.number_input("Points to award", min_value=-100, max_value=100, value=1, step=1)

    if st.button("Add Non-Game Points", type="primary"):
        entry_id = uuid.uuid4().hex[:10]
        append_row("NonGamePoints", [entry_id, current_league_key, team, cat, reason, int(pts), now_et_iso()])
        toast_ok("Non-game points added.")
        st.cache_data.clear()

    st.divider()
    df = df_from_ws_cached("NonGamePoints")
    df = df[df["league_key"] == current_league_key].copy() if not df.empty else df
    if df.empty:
        st.info("No non-game points yet.")
    else:
        st.dataframe(df, use_container_width=True)

def page_admin():
    st.markdown('<div class="bc-title">Admin / Clear Data</div>', unsafe_allow_html=True)

    pw = st.text_input("Admin password", type="password")
    if pw != "Hyaffa26":
        st.info("Enter password to access admin tools.")
        return

    st.success("Admin unlocked.")

    st.warning("These actions affect Google Sheets data. Use carefully.")

    action = st.selectbox("Action", [
        "Clear roster for a league",
        "Clear all games",
        "Clear all game stats",
        "Clear non-game points",
        "Clear highlights log",
    ])

    if action == "Clear roster for a league":
        league = st.selectbox("League", list(LEAGUES.keys()))
        lk = LEAGUES[league]
        sheet_name = LEAGUE_TO_ROSTER_SHEET[lk]
        if st.button(f"Clear {league} roster", type="primary"):
            ensure_headers(sheet_name)
            w = ws(sheet_name)
            w.clear()
            w.append_row(HEADERS[sheet_name], value_input_option="RAW")
            st.cache_data.clear()
            st.success("Roster cleared.")

    if action == "Clear all games":
        if st.button("Clear Games sheet", type="primary"):
            ensure_headers("Games")
            w = ws("Games")
            w.clear()
            w.append_row(HEADERS["Games"], value_input_option="RAW")
            st.cache_data.clear()
            st.success("Games cleared.")

    if action == "Clear all game stats":
        if st.button("Clear GameStats sheet", type="primary"):
            ensure_headers("GameStats")
            w = ws("GameStats")
            w.clear()
            w.append_row(HEADERS["GameStats"], value_input_option="RAW")
            st.cache_data.clear()
            st.success("GameStats cleared.")

    if action == "Clear non-game points":
        if st.button("Clear NonGamePoints sheet", type="primary"):
            ensure_headers("NonGamePoints")
            w = ws("NonGamePoints")
            w.clear()
            w.append_row(HEADERS["NonGamePoints"], value_input_option="RAW")
            st.cache_data.clear()
            st.success("Non-game points cleared.")

    if action == "Clear highlights log":
        if st.button("Clear Highlights sheet", type="primary"):
            ensure_headers("Highlights")
            w = ws("Highlights")
            w.clear()
            w.append_row(HEADERS["Highlights"], value_input_option="RAW")
            st.cache_data.clear()
            st.success("Highlights log cleared.")

# ---------------------------
# Main
# ---------------------------
def main():
    st.set_page_config(page_title="Bauercrest League Manager", layout="wide")
    inject_css()
    init_state()

    # Sidebar
    with st.sidebar:
        st.title("Manager")

        # logo
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except Exception:
            pass

        league_label = st.selectbox("League (Setup / Live / Stats)", list(LEAGUES.keys()))
        current_league = LEAGUES[league_label]

        st.caption(f"Managing league data for: **{league_label}**")

        page = st.radio("Go to", [
            "Setup",
            "Live Games (Create/Open)",
            "Run Live Game",
            "Standings",
            "Leaderboards",
            "Non-Game Points",
            "Admin / Clear Data",
        ])

    # Pages
    if page == "Setup":
        page_setup(current_league)
    elif page == "Live Games (Create/Open)":
        page_live_games_home(current_league)
    elif page == "Run Live Game":
        page_run_live_game()
    elif page == "Standings":
        page_standings(current_league)
    elif page == "Leaderboards":
        page_leaderboards(current_league)
    elif page == "Non-Game Points":
        page_non_game_points(current_league)
    elif page == "Admin / Clear Data":
        page_admin()

if __name__ == "__main__":
    main()
