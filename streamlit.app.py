import time
import uuid
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pytz
import streamlit as st

# Supabase
try:
    from supabase import create_client
except Exception:
    create_client = None

# Google Sheets (optional)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

# =========================
# CAMP SETTINGS / BRANDING
# =========================
ET = pytz.timezone("America/New_York")
APP_TITLE = "Bauercrest League Manager"

LOGO_PATH = "logo-header-2.png"  # optional
BC_BG = "#000000"
BC_PRIMARY = "#0B1E3B"
BC_ACCENT = "#C9A227"

# =========================
# LEAGUES / SHEET TAB NAMES
# =========================
LEAGUES = {
    "Sophomore League": "sophmore",
    "Junior League": "junior",
    "Senior League": "seniors",
}

ROSTER_SHEETS = {
    "sophmore": "rosters_sophmore",
    "junior": "rosters_junior",
    "seniors": "rosters_seniors",
}

SHEET_TABS = {
    "games": "games",
    "stats": "stats",
    "nongamepoints": "nongamepoints",
    "highlights": "highlights",
}

# =========================
# SPORTS / STATS / TIMERS
# =========================
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

SPORT_STATS = {
    "Basketball": [("PTS", "+1 PTS"), ("AST", "+1 AST"), ("REB", "+1 REB"), ("STL", "+1 STL"), ("BLK", "+1 BLK")],
    "Hockey":     [("G", "+1 G"), ("A", "+1 A"), ("SV", "+1 SV")],
    "Football":   [("TD", "+1 TD"), ("REC", "+1 REC"), ("INT", "+1 INT")],
    "Softball":   [("H", "+1 H"), ("2B", "+1 2B"), ("3B", "+1 3B"), ("HR", "+1 HR"), ("RBI", "+1 RBI"), ("R", "+1 R")],
    "Soccer":     [("G", "+1 G"), ("A", "+1 A"), ("SV", "+1 SV")],
    "Kickball":   [("R", "+1 R"), ("H", "+1 H")],
    "Euro":       [("PTS", "+1 PTS"), ("AST", "+1 AST")],
    "Speedball":  [("PTS", "+1 PTS"), ("AST", "+1 AST"), ("G", "+1 G")],
}

# More timer presets per sport (you asked for more, not less)
TIMER_PRESETS = {
    "Basketball": [
        ("15:00 halves (running)", 30*60),
        ("15:00 halves (stop clock)", 30*60),
        ("20:00 halves (running)", 40*60),
        ("20:00 halves (stop clock)", 40*60),
        ("10:00 halves (running)", 20*60),
        ("No timer (manual)", 0),
    ],
    "Hockey": [
        ("12:00 periods (running)", 36*60),
        ("15:00 periods (running)", 45*60),
        ("10:00 periods (running)", 30*60),
        ("No timer (manual)", 0),
    ],
    "Football": [
        ("20:00 halves (running)", 40*60),
        ("15:00 halves (running)", 30*60),
        ("10:00 halves (running)", 20*60),
        ("No timer (manual)", 0),
    ],
    "Soccer": [
        ("20:00 halves (running)", 40*60),
        ("15:00 halves (running)", 30*60),
        ("10:00 halves (running)", 20*60),
        ("No timer (manual)", 0),
    ],
    "Softball": [
        ("No timer (innings style)", 0),
        ("60:00 hard cap (running)", 60*60),
    ],
    "Kickball": [
        ("No timer (innings style)", 0),
        ("45:00 hard cap (running)", 45*60),
        ("30:00 hard cap (running)", 30*60),
    ],
    "Euro": [
        ("15:00 halves (running)", 30*60),
        ("20:00 halves (running)", 40*60),
        ("No timer (manual)", 0),
    ],
    "Speedball": [
        ("15:00 halves (running)", 30*60),
        ("20:00 halves (running)", 40*60),
        ("No timer (manual)", 0),
    ],
}

# Points table (camp-approximate). You can tweak later in Admin.
POINTS_TABLE = {
    "seniors": {
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
    "sophmore": {
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

# =========================
# UTILS
# =========================
def now_et() -> datetime:
    return datetime.now(ET)

def now_et_str() -> str:
    return now_et().strftime("%Y-%m-%d %I:%M:%S %p ET")

def format_mm_ss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"

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
    if hasattr(st, "toast"):
        st.toast(msg, icon="✅")
    else:
        st.success(msg)

def toast_err(msg: str):
    if hasattr(st, "toast"):
        st.toast(msg, icon="⚠️")
    else:
        st.error(msg)

# =========================
# STYLE
# =========================
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {BC_BG}; }}
        .title {{
            font-size: 2rem; font-weight: 900; color: {BC_PRIMARY}; margin: 0.25rem 0 0.75rem 0;
        }}
        .card {{
            background: white; border: 2px solid {BC_PRIMARY}; border-radius: 16px;
            padding: 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }}
        .sb-time {{
            font-size: 4.2rem; font-weight: 950; color: {BC_PRIMARY};
            text-align: center; letter-spacing: 2px; margin: 6px 0 4px 0;
        }}
        .sb-sub {{
            text-align: center; color: #666; font-weight: 700; margin-bottom: 14px;
        }}
        .sb-row {{
            display: flex; gap: 14px; justify-content: space-between; align-items: stretch;
        }}
        .sb-team {{
            flex: 1; background: #f7f9ff; border: 1px solid #e6e9f5;
            border-radius: 14px; padding: 12px;
        }}
        .sb-teamname {{ font-size: 1.1rem; font-weight: 900; color: {BC_PRIMARY}; margin-bottom: 6px; }}
        .sb-score {{ font-size: 3.6rem; font-weight: 950; color: {BC_PRIMARY}; line-height: 1; }}
        .pill {{
            display: inline-block; padding: 6px 10px; border-radius: 999px;
            background: {BC_PRIMARY}; color: white; font-weight: 800; font-size: 0.9rem; margin-top: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# SECRETS + CLIENTS
# =========================
def get_secret(*keys: str) -> Optional[str]:
    for k in keys:
        v = st.secrets.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return None

@st.cache_resource
def supabase_client():
    if create_client is None:
        return None
    url = get_secret("supabase_url", "SUPABASE_URL")
    key = get_secret("supabase_anon_key", "SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

@st.cache_resource
def gspread_client():
    sid = get_secret("sheet_id", "SHEET_ID")
    sa = st.secrets.get("gcp_service_account")
    if not sid or not sa or gspread is None or Credentials is None:
        return None
    # sa may be dict, json string, or toml multiline string
    try:
        if isinstance(sa, dict):
            info = sa
        else:
            info = json.loads(sa)
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
        )
        gc = gspread.authorize(creds)
        return gc.open_by_key(sid)
    except Exception:
        return None

# =========================
# ROSTERS (Sheets-first, CSV fallback)
# =========================
@st.cache_data(ttl=60)
def load_roster_from_sheets(league_key: str) -> pd.DataFrame:
    book = gspread_client()
    if book is None:
        return pd.DataFrame()
    tab = ROSTER_SHEETS[league_key]
    ws = book.worksheet(tab)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(columns=["player_id", "first_name", "last_name", "team_name", "bunk"])
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    # normalize
    for c in ["player_id", "first_name", "last_name", "team_name", "bunk"]:
        if c not in df.columns:
            df[c] = ""
    df = df[["player_id", "first_name", "last_name", "team_name", "bunk"]].copy()
    df["player_id"] = df["player_id"].astype(str)
    return df

def load_roster(league_key: str) -> pd.DataFrame:
    # 1) session override (CSV upload)
    key = f"roster_df_{league_key}"
    if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame) and not st.session_state[key].empty:
        return st.session_state[key].copy()
    # 2) sheets
    df = load_roster_from_sheets(league_key)
    if not df.empty:
        return df
    # 3) empty
    return pd.DataFrame(columns=["player_id", "first_name", "last_name", "team_name", "bunk"])

# =========================
# LIVE GAME ENGINE (local session)
# =========================
def init_state():
    st.session_state.setdefault("live_games", {})         # game_id -> game dict
    st.session_state.setdefault("active_game_id", None)   # currently open game in this session
    st.session_state.setdefault("tick", 0)

def new_game_id() -> str:
    return uuid.uuid4().hex

def compute_remaining(game: Dict) -> int:
    dur = int(game.get("duration_seconds", 0))
    if dur <= 0:
        return 0
    if not game.get("timer_running", False):
        return int(game.get("remaining_at_pause", dur))
    anchor_ts = float(game.get("anchor_ts", time.time()))
    remaining_at_anchor = int(game.get("remaining_at_anchor", dur))
    elapsed = time.time() - anchor_ts
    return max(0, int(remaining_at_anchor - elapsed))

def timer_start(game: Dict):
    dur = int(game.get("duration_seconds", 0))
    if dur <= 0:
        return
    if game.get("timer_running", False):
        return
    remaining = int(game.get("remaining_at_pause", dur))
    game["timer_running"] = True
    game["anchor_ts"] = time.time()
    game["remaining_at_anchor"] = remaining

def timer_pause(game: Dict):
    if not game.get("timer_running", False):
        return
    remaining = compute_remaining(game)
    game["timer_running"] = False
    game["remaining_at_pause"] = remaining

def timer_reset(game: Dict):
    dur = int(game.get("duration_seconds", 0))
    game["timer_running"] = False
    game["remaining_at_pause"] = dur
    game["anchor_ts"] = time.time()
    game["remaining_at_anchor"] = dur

def bump_score(game: Dict, side: str, delta: int):
    if side == "A":
        game["score_a"] = max(0, int(game.get("score_a", 0)) + int(delta))
    else:
        game["score_b"] = max(0, int(game.get("score_b", 0)) + int(delta))

def bump_player_stat(game: Dict, player_id: str, stat_key: str, delta: int = 1):
    stats = game.setdefault("player_stats", {})  # pid -> {stat_key:value}
    if player_id not in stats:
        stats[player_id] = {}
    stats[player_id][stat_key] = int(stats[player_id].get(stat_key, 0)) + int(delta)

# =========================
# SUPABASE SAVE (Finalize)
# =========================
def supabase_ok() -> bool:
    return supabase_client() is not None

def finalize_to_supabase(game: Dict, roster_df: pd.DataFrame) -> Tuple[bool, str]:
    sb = supabase_client()
    if sb is None:
        return False, "Supabase not configured."

    # Prepare game record
    gid = game["game_id"]
    league_key = game["league_key"]
    sport = game["sport"]
    level = game["level"]
    mode = game["mode"]
    team_a1 = game["team_a1"]
    team_a2 = game.get("team_a2", "")
    team_b1 = game["team_b1"]
    team_b2 = game.get("team_b2", "")
    score_a = int(game.get("score_a", 0))
    score_b = int(game.get("score_b", 0))
    notes = game.get("notes", "")

    pts_a, pts_b = points_for_result(league_key, sport, level, score_a, score_b)

    # 1) Insert into games
    try:
        sb.table("games").insert({
            "id": gid,
            "league_key": league_key,
            "sport": sport,
            "level": level,
            "mode": mode,
            "team_a1": team_a1,
            "team_a2": team_a2 if team_a2 else None,
            "team_b1": team_b1,
            "team_b2": team_b2 if team_b2 else None,
            "score_a": score_a,
            "score_b": score_b,
            "points_a": pts_a,
            "points_b": pts_b,
            "notes": notes
        }).execute()
    except Exception as e:
        return False, f"Failed to save game: {e}"

    # 2) Insert stats totals
    active_ids = set(game.get("active_player_ids", []))
    stats_map = game.get("player_stats", {})
    name_map = {}
    team_map = {}
    if not roster_df.empty:
        roster_df = roster_df.copy()
        roster_df["player_id"] = roster_df["player_id"].astype(str)
        roster_df["player_name"] = roster_df["first_name"].astype(str) + " " + roster_df["last_name"].astype(str)
        for _, r in roster_df.iterrows():
            pid = str(r["player_id"])
            name_map[pid] = str(r["player_name"]).strip()
            team_map[pid] = str(r["team_name"]).strip()

    rows = []
    for pid, m in stats_map.items():
        if active_ids and pid not in active_ids:
            continue
        for stat_key, val in m.items():
            val = int(val)
            if val == 0:
                continue
            rows.append({
                "game_id": gid,
                "league_key": league_key,
                "sport": sport,
                "level": level,
                "player_id": str(pid),
                "player_name": name_map.get(str(pid), str(pid)),
                "team_name": team_map.get(str(pid), ""),
                "stat_key": stat_key,
                "value": val
            })

    try:
        if rows:
            sb.table("stats").insert(rows).execute()
    except Exception as e:
        return False, f"Game saved, but stats failed: {e}"

    return True, "Saved."

# =========================
# UI PAGES
# =========================
def sidebar_nav() -> Tuple[str, str]:
    with st.sidebar:
        st.title("League Manager")
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except Exception:
            pass

        league_label = st.selectbox("League", list(LEAGUES.keys()))
        league_key = LEAGUES[league_label]

        page = st.radio("Menu", [
            "Live Games",
            "Run Live Game",
            "Standings",
            "Leaderboards",
            "Non-Game Points",
            "Highlights",
            "Roster Tools",
            "Admin",
        ])

        # Connection status (useful, not “foundation”)
        st.divider()
        st.caption("Connections")
        st.write("Supabase:", "✅" if supabase_ok() else "❌")
        st.write("Google Sheets:", "✅" if gspread_client() is not None else "⚠️ (optional)")

    return page, league_key

def page_live_games(league_key: str):
    st.markdown(f'<div class="title">Live Games</div>', unsafe_allow_html=True)
    roster = load_roster(league_key)
    if roster.empty:
        st.warning("No roster loaded for this league yet. Go to **Roster Tools** and upload a CSV, or connect Google Sheets roster tabs.")
        return

    teams = sorted([t for t in roster["team_name"].dropna().unique().tolist() if str(t).strip()])

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sport = st.selectbox("Sport", SPORTS)
    with col2:
        level = st.selectbox("Level", ["A", "B", "C", "D"])
    with col3:
        mode_ui = st.selectbox("Mode", ["1 Team vs 1 Team", "2 Teams vs 2 Teams"])

    presets = TIMER_PRESETS.get(sport, [("No timer (manual)", 0)])
    preset_label = st.selectbox("Timer Preset", [p[0] for p in presets])
    duration_seconds = int([p[1] for p in presets if p[0] == preset_label][0])

    if mode_ui == "1 Team vs 1 Team":
        a, b = st.columns(2)
        with a:
            team_a1 = st.selectbox("Team A", teams, key="t_a1")
        with b:
            team_b1 = st.selectbox("Team B", [t for t in teams if t != team_a1], key="t_b1")
        team_a2 = ""
        team_b2 = ""
        mode = "1v1"
    else:
        st.caption("Pick two teams per side.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            team_a1 = st.selectbox("A1", teams, key="t_a1_2")
        with c2:
            team_a2 = st.selectbox("A2", [t for t in teams if t != team_a1], key="t_a2_2")
        with c3:
            team_b1 = st.selectbox("B1", [t for t in teams if t not in [team_a1, team_a2]], key="t_b1_2")
        with c4:
            team_b2 = st.selectbox("B2", [t for t in teams if t not in [team_a1, team_a2, team_b1]], key="t_b2_2")
        mode = "2v2"

    notes = st.text_input("Notes (optional)")

    if st.button("Create Live Game", type="primary"):
        gid = new_game_id()
        # Eligible players = teams involved
        involved = [team_a1, team_b1] if mode == "1v1" else [team_a1, team_a2, team_b1, team_b2]
        eligible = roster[roster["team_name"].isin(involved)].copy()
        eligible["player_id"] = eligible["player_id"].astype(str)
        eligible["player_name"] = eligible["first_name"].astype(str) + " " + eligible["last_name"].astype(str)

        game = {
            "game_id": gid,
            "league_key": league_key,
            "sport": sport,
            "level": level,
            "mode": mode,
            "team_a1": team_a1,
            "team_a2": team_a2,
            "team_b1": team_b1,
            "team_b2": team_b2,
            "created_at": now_et_str(),
            "notes": notes,

            # scoreboard
            "score_a": 0,
            "score_b": 0,

            # timer
            "duration_seconds": duration_seconds,
            "timer_running": False,
            "remaining_at_pause": duration_seconds,
            "anchor_ts": time.time(),
            "remaining_at_anchor": duration_seconds,

            # lineup & stats
            "eligible_df": eligible[["player_id", "player_name", "team_name"]].copy(),
            "active_player_ids": eligible["player_id"].tolist(),
            "player_stats": {},
        }

        st.session_state.live_games[gid] = game
        st.session_state.active_game_id = gid
        toast_ok("Live game created.")
        st.experimental_rerun()

    st.divider()
    st.subheader("Your Live Games (this device)")
    if not st.session_state.live_games:
        st.info("No live games created on this device yet.")
        return

    rows = []
    for gid, g in st.session_state.live_games.items():
        rows.append({
            "Game ID": gid,
            "League": g["league_key"],
            "Sport": g["sport"],
            "Level": g["level"],
            "Mode": g["mode"],
            "A": g["team_a1"] + (f"+{g['team_a2']}" if g["mode"] == "2v2" and g.get("team_a2") else ""),
            "B": g["team_b1"] + (f"+{g['team_b2']}" if g["mode"] == "2v2" and g.get("team_b2") else ""),
            "Score": f"{g['score_a']} - {g['score_b']}",
            "Created": g["created_at"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    pick = st.selectbox("Open a live game", list(st.session_state.live_games.keys()))
    if st.button("Open Selected Game"):
        st.session_state.active_game_id = pick
        st.experimental_rerun()

def page_run_live_game(league_key: str):
    st.markdown(f'<div class="title">Run Live Game</div>', unsafe_allow_html=True)

    gid = st.session_state.active_game_id
    if not gid or gid not in st.session_state.live_games:
        st.info("Go to **Live Games** to create or open a game.")
        return

    game = st.session_state.live_games[gid]
    roster = load_roster(league_key)

    # Auto tick every second only while timer is running
    if game.get("timer_running", False) and int(game.get("duration_seconds", 0)) > 0:
        time.sleep(1)
        st.session_state.tick += 1
        st.experimental_rerun()

    remaining = compute_remaining(game)
    time_str = "—" if int(game.get("duration_seconds", 0)) <= 0 else format_mm_ss(remaining)

    team_a_label = game["team_a1"] + (f" + {game['team_a2']}" if game["mode"] == "2v2" and game.get("team_a2") else "")
    team_b_label = game["team_b1"] + (f" + {game['team_b2']}" if game["mode"] == "2v2" and game.get("team_b2") else "")

    # Scoreboard card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-time">{time_str}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-sub">{game["sport"]} • {game["level"]} • {("2v2" if game["mode"]=="2v2" else "1v1")}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sb-row">
          <div class="sb-team">
            <div class="sb-teamname">{team_a_label}</div>
            <div class="sb-score">{game["score_a"]}</div>
            <div class="pill">TEAM A</div>
          </div>
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

    # Timer controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("▶ Start / Resume", use_container_width=True):
            timer_start(game)
            toast_ok("Timer started.")
            st.experimental_rerun()
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            timer_pause(game)
            toast_ok("Timer paused.")
            st.experimental_rerun()
    with c3:
        if st.button("⟲ Reset Timer", use_container_width=True):
            timer_reset(game)
            toast_ok("Timer reset.")
            st.experimental_rerun()
    with c4:
        if st.button("🧹 Reset Score", use_container_width=True):
            game["score_a"] = 0
            game["score_b"] = 0
            toast_ok("Score reset.")
            st.experimental_rerun()

    st.divider()

    # Score buttons
    st.subheader("Score Controls")
    if game["sport"] == "Basketball":
        left, right = st.columns(2)
        with left:
            st.write("Team A")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("+1", use_container_width=True): bump_score(game, "A", 1); toast_ok("Team A +1"); st.experimental_rerun()
            if b2.button("+2", use_container_width=True): bump_score(game, "A", 2); toast_ok("Team A +2"); st.experimental_rerun()
            if b3.button("+3", use_container_width=True): bump_score(game, "A", 3); toast_ok("Team A +3"); st.experimental_rerun()
            if b4.button("-1", use_container_width=True): bump_score(game, "A", -1); toast_ok("Team A -1"); st.experimental_rerun()
        with right:
            st.write("Team B")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("+1 ", use_container_width=True): bump_score(game, "B", 1); toast_ok("Team B +1"); st.experimental_rerun()
            if b2.button("+2 ", use_container_width=True): bump_score(game, "B", 2); toast_ok("Team B +2"); st.experimental_rerun()
            if b3.button("+3 ", use_container_width=True): bump_score(game, "B", 3); toast_ok("Team B +3"); st.experimental_rerun()
            if b4.button("-1 ", use_container_width=True): bump_score(game, "B", -1); toast_ok("Team B -1"); st.experimental_rerun()
    else:
        left, right = st.columns(2)
        with left:
            st.write("Team A")
            b1, b2 = st.columns(2)
            if b1.button("+1", use_container_width=True): bump_score(game, "A", 1); toast_ok("Team A +1"); st.experimental_rerun()
            if b2.button("-1", use_container_width=True): bump_score(game, "A", -1); toast_ok("Team A -1"); st.experimental_rerun()
        with right:
            st.write("Team B")
            b1, b2 = st.columns(2)
            if b1.button("+1 ", use_container_width=True): bump_score(game, "B", 1); toast_ok("Team B +1"); st.experimental_rerun()
            if b2.button("-1 ", use_container_width=True): bump_score(game, "B", -1); toast_ok("Team B -1"); st.experimental_rerun()

    st.divider()

    # Active lineup
    st.subheader("Active Lineup")
    eligible_df = game["eligible_df"].copy()
    pid_to_label = {r["player_id"]: f"{r['player_name']} — {r['team_name']}" for _, r in eligible_df.iterrows()}
    picked = st.multiselect(
        "Select the players actually playing (removes everyone else from the stat buttons).",
        options=eligible_df["player_id"].tolist(),
        default=game.get("active_player_ids", []),
        format_func=lambda pid: pid_to_label.get(pid, pid),
    )
    game["active_player_ids"] = picked

    st.divider()

    # Player stat buttons + running totals
    st.subheader("Player Stats")
    stat_buttons = SPORT_STATS.get(game["sport"], [("PTS", "+1 PTS")])

    active_df = eligible_df[eligible_df["player_id"].isin(game.get("active_player_ids", []))].copy()
    active_df = active_df.sort_values(["team_name", "player_name"])

    if active_df.empty:
        st.warning("No active players selected.")
    else:
        for _, r in active_df.iterrows():
            pid = r["player_id"]
            name = r["player_name"]
            team = r["team_name"]

            cols = st.columns([2.5] + [1] * len(stat_buttons))
            cols[0].markdown(f"**{name}**  \n_{team}_")

            for i, (stat_key, label) in enumerate(stat_buttons):
                if cols[i+1].button(label, key=f"{gid}_{pid}_{stat_key}"):
                    bump_player_stat(game, pid, stat_key, 1)
                    # green confirmation
                    toast_ok(f"{name}: +1 {stat_key}")
                    st.experimental_rerun()

    st.divider()

    st.subheader("This Game Totals (updates instantly)")
    rows = []
    stats_map = game.get("player_stats", {})
    for _, r in active_df.iterrows():
        pid = r["player_id"]
        row = {"Player": r["player_name"], "Team": r["team_name"]}
        for sk, _lbl in stat_buttons:
            row[sk] = int(stats_map.get(pid, {}).get(sk, 0))
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.subheader("Finalize & Save")
    st.caption("This saves the completed game and all stats to Supabase. Your live scoring stays fast and reliable.")

    game["notes"] = st.text_area("Notes (optional)", value=game.get("notes", ""))

    if st.button("✅ Finalize & Save Game", type="primary"):
        # pause timer to lock time
        timer_pause(game)
        ok, msg = finalize_to_supabase(game, roster)
        if ok:
            toast_ok("Game saved successfully.")
        else:
            toast_err(msg)

def page_standings(league_key: str):
    st.markdown(f'<div class="title">Standings</div>', unsafe_allow_html=True)
    roster = load_roster(league_key)
    if roster.empty:
        st.warning("No roster loaded for this league yet.")
        return

    teams = sorted([t for t in roster["team_name"].dropna().unique().tolist() if str(t).strip()])
    include_nongame = st.toggle("Include non-game points in totals", value=True)

    # Pull from Supabase for reliability
    sb = supabase_client()
    if sb is None:
        st.error("Supabase not configured.")
        return

    try:
        games = sb.table("games").select("*").eq("league_key", league_key).execute().data
        ng = sb.table("nongamepoints").select("*").eq("league_key", league_key).execute().data
    except Exception as e:
        st.error(f"Supabase read error: {e}")
        return

    # Compute points + W/L/T
    pts_game = {t: 0.0 for t in teams}
    w = {t: 0 for t in teams}
    l = {t: 0 for t in teams}
    t_ = {t: 0 for t in teams}
    pts_ng = {t: 0.0 for t in teams}

    for g in games or []:
        mode = g.get("mode", "1v1")
        a_teams = [g.get("team_a1")] + ([g.get("team_a2")] if mode == "2v2" else [])
        b_teams = [g.get("team_b1")] + ([g.get("team_b2")] if mode == "2v2" else [])
        a_teams = [x for x in a_teams if x]
        b_teams = [x for x in b_teams if x]
        pa = float(g.get("points_a", 0))
        pb = float(g.get("points_b", 0))
        sa = int(g.get("score_a", 0))
        sb_ = int(g.get("score_b", 0))

        for team in a_teams:
            if team in pts_game:
                pts_game[team] += pa / max(1, len(a_teams))
        for team in b_teams:
            if team in pts_game:
                pts_game[team] += pb / max(1, len(b_teams))

        if sa > sb_:
            for team in a_teams:
                if team in w: w[team] += 1
            for team in b_teams:
                if team in l: l[team] += 1
        elif sb_ > sa:
            for team in b_teams:
                if team in w: w[team] += 1
            for team in a_teams:
                if team in l: l[team] += 1
        else:
            for team in a_teams + b_teams:
                if team in t_: t_[team] += 1

    for row in ng or []:
        team = str(row.get("team_name", "")).strip()
        pts = float(row.get("points", 0))
        if team in pts_ng:
            pts_ng[team] += pts

    rows = []
    for team in teams:
        total = pts_game[team] + (pts_ng[team] if include_nongame else 0.0)
        rows.append({
            "Team": team,
            "W": w[team],
            "L": l[team],
            "T": t_[team],
            "Game Pts": round(pts_game[team], 1),
            "Non-Game Pts": round(pts_ng[team], 1),
            "Total Pts": round(total, 1),
        })
    df = pd.DataFrame(rows).sort_values(["Total Pts", "W"], ascending=False)
    st.dataframe(df, use_container_width=True)

def page_leaderboards(league_key: str):
    st.markdown(f'<div class="title">Leaderboards</div>', unsafe_allow_html=True)
    sb = supabase_client()
    if sb is None:
        st.error("Supabase not configured.")
        return

    try:
        stats = sb.table("stats").select("*").eq("league_key", league_key).execute().data
    except Exception as e:
        st.error(f"Supabase read error: {e}")
        return

    if not stats:
        st.info("No stats saved yet.")
        return

    df = pd.DataFrame(stats)
    sport = st.selectbox("Sport", sorted(df["sport"].unique().tolist()))
    df = df[df["sport"] == sport].copy()
    stat_key = st.selectbox("Category", sorted(df["stat_key"].unique().tolist()))
    df = df[df["stat_key"] == stat_key].copy()

    board = df.groupby(["player_name", "team_name"], as_index=False)["value"].sum()
    board = board.sort_values("value", ascending=False).head(25)
    board = board.rename(columns={"player_name": "Player", "team_name": "Team", "value": "Total"})
    st.dataframe(board, use_container_width=True)

def page_non_game_points(league_key: str):
    st.markdown(f'<div class="title">Non-Game Points</div>', unsafe_allow_html=True)
    roster = load_roster(league_key)
    if roster.empty:
        st.warning("No roster loaded for this league yet.")
        return
    teams = sorted([t for t in roster["team_name"].dropna().unique().tolist() if str(t).strip()])

    sb = supabase_client()
    if sb is None:
        st.error("Supabase not configured.")
        return

    category = st.selectbox("Category", [
        "League Spirit",
        "Sportsmanship",
        "Cleanup / Organization",
        "Participation / Effort",
        "Other",
    ])
    team = st.selectbox("Team", teams)
    reason = st.text_input("Reason / Details")
    points = st.number_input("Points", min_value=-100, max_value=100, value=1, step=1)

    if st.button("Add Points", type="primary"):
        try:
            sb.table("nongamepoints").insert({
                "league_key": league_key,
                "team_name": team,
                "category": category,
                "reason": reason,
                "points": int(points),
            }).execute()
            toast_ok("Non-game points added.")
        except Exception as e:
            toast_err(f"Save failed: {e}")

    st.divider()
    try:
        rows = sb.table("nongamepoints").select("*").eq("league_key", league_key).order("created_at", desc=True).execute().data
    except Exception as e:
        st.error(f"Supabase read error: {e}")
        return
    if not rows:
        st.info("No entries yet.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

def page_highlights(league_key: str):
    st.markdown(f'<div class="title">Highlights</div>', unsafe_allow_html=True)
    sb = supabase_client()
    if sb is None:
        st.error("Supabase not configured.")
        return

    uploaded = st.file_uploader("Upload a highlight video file", type=["mp4", "mov", "m4v"])
    st.caption("This version logs filenames for the day. If you want cloud storage + playback, we can add Supabase Storage next.")

    if uploaded is not None:
        # For now, store filename in DB. (Storage can come next.)
        if st.button("Save Highlight Entry", type="primary"):
            try:
                sb.table("highlights").insert({
                    "league_key": league_key,
                    "filename": uploaded.name,
                }).execute()
                toast_ok("Highlight entry saved.")
            except Exception as e:
                toast_err(f"Save failed: {e}")

    st.divider()
    try:
        rows = sb.table("highlights").select("*").eq("league_key", league_key).order("created_at", desc=True).execute().data
    except Exception as e:
        st.error(f"Supabase read error: {e}")
        return
    if not rows:
        st.info("No highlights logged yet.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

def page_roster_tools(league_key: str):
    st.markdown(f'<div class="title">Roster Tools</div>', unsafe_allow_html=True)
    st.caption("You can pull rosters from Google Sheets or upload a CSV as a backup.")

    df = load_roster_from_sheets(league_key)
    if not df.empty:
        st.success("Roster loaded from Google Sheets.")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Google Sheets roster not available for this league (or not configured).")

    st.divider()
    st.subheader("Upload roster CSV (backup)")
    st.caption("Required columns: player_id, first_name, last_name, team_name, bunk")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df_new = pd.read_csv(file)
            needed = ["player_id", "first_name", "last_name", "team_name", "bunk"]
            missing = [c for c in needed if c not in df_new.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                return
            df_new = df_new[needed].copy()
            df_new["player_id"] = df_new["player_id"].astype(str)
            st.session_state[f"roster_df_{league_key}"] = df_new
            toast_ok("Roster loaded from CSV for this session.")
            st.dataframe(df_new, use_container_width=True)
        except Exception as e:
            st.error(f"CSV error: {e}")

def page_admin():
    st.markdown(f'<div class="title">Admin</div>', unsafe_allow_html=True)
    pw = st.text_input("Admin password", type="password")
    if pw != "Hyaffa26":
        st.info("Enter password to access admin tools.")
        return

    sb = supabase_client()
    if sb is None:
        st.error("Supabase not configured.")
        return

    st.success("Admin unlocked.")
    action = st.selectbox("Action", [
        "Delete a saved game (Supabase)",
        "Clear ALL saved games (Supabase)",
        "Clear ALL saved stats (Supabase)",
        "Clear non-game points (Supabase)",
        "Clear highlights log (Supabase)",
    ])

    if action == "Delete a saved game (Supabase)":
        game_id = st.text_input("Game ID (uuid hex from your session)")
        if st.button("Delete Game", type="primary"):
            try:
                sb.table("games").delete().eq("id", game_id).execute()
                sb.table("stats").delete().eq("game_id", game_id).execute()
                toast_ok("Deleted.")
            except Exception as e:
                toast_err(f"Delete failed: {e}")

    if action == "Clear ALL saved games (Supabase)":
        if st.button("Clear games table", type="primary"):
            try:
                sb.table("games").delete().neq("league_key", "___nope___").execute()
                toast_ok("Games cleared.")
            except Exception as e:
                toast_err(f"Clear failed: {e}")

    if action == "Clear ALL saved stats (Supabase)":
        if st.button("Clear stats table", type="primary"):
            try:
                sb.table("stats").delete().neq("league_key", "___nope___").execute()
                toast_ok("Stats cleared.")
            except Exception as e:
                toast_err(f"Clear failed: {e}")

    if action == "Clear non-game points (Supabase)":
        if st.button("Clear nongamepoints table", type="primary"):
            try:
                sb.table("nongamepoints").delete().neq("league_key", "___nope___").execute()
                toast_ok("Non-game points cleared.")
            except Exception as e:
                toast_err(f"Clear failed: {e}")

    if action == "Clear highlights log (Supabase)":
        if st.button("Clear highlights table", type="primary"):
            try:
                sb.table("highlights").delete().neq("league_key", "___nope___").execute()
                toast_ok("Highlights cleared.")
            except Exception as e:
                toast_err(f"Clear failed: {e}")

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()
    init_state()

    page, league_key = sidebar_nav()

    if page == "Live Games":
        page_live_games(league_key)
    elif page == "Run Live Game":
        page_run_live_game(league_key)
    elif page == "Standings":
        page_standings(league_key)
    elif page == "Leaderboards":
        page_leaderboards(league_key)
    elif page == "Non-Game Points":
        page_non_game_points(league_key)
    elif page == "Highlights":
        page_highlights(league_key)
    elif page == "Roster Tools":
        page_roster_tools(league_key)
    elif page == "Admin":
        page_admin()

if __name__ == "__main__":
    main()
