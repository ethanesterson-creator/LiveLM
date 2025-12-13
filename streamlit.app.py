import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import io
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import gspread
from google.oauth2.service_account import Credentials

# =========================
# Bauercrest League Manager (Live + Multi-Game safe)
# Google Sheets backend w/ caching to avoid 429 quota
# =========================

# ---------- Brand ----------
BAUERCREST_NAVY = "#0b1f3a"
BAUERCREST_GOLD = "#f5c542"

LOGO_PATH = "logo-header-2.png"  # put in repo root, or change name here

# ---------- Leagues ----------
LEAGUES = ["Sophomore League", "Junior League", "Senior League"]
LEAGUE_KEYS = {
    "Sophomore League": "SOPH",
    "Junior League": "JUNIOR",
    "Senior League": "SENIOR",
}

# ---------- Sports + Timer Presets ----------
# Keep this list big and realistic; add more whenever you want.
SPORTS = [
    "Basketball", "Hockey", "Softball", "Football", "Kickball",
    "Euro", "Speedball", "Soccer", "Lacrosse", "Volleyball", "Street Hockey"
]

TIMER_PRESETS = {
    "Basketball": [
        ("15 min halves (stop clock)", {"mode": "countdown", "segments": [15*60, 15*60], "running": False}),
        ("15 min halves (running)", {"mode": "countdown", "segments": [15*60, 15*60], "running": True}),
        ("20 min halves (stop clock)", {"mode": "countdown", "segments": [20*60, 20*60], "running": False}),
        ("20 min halves (running)", {"mode": "countdown", "segments": [20*60, 20*60], "running": True}),
        ("10 min quarters (stop clock)", {"mode": "countdown", "segments": [10*60]*4, "running": False}),
        ("10 min quarters (running)", {"mode": "countdown", "segments": [10*60]*4, "running": True}),
    ],
    "Hockey": [
        ("12 min periods (stop clock)", {"mode": "countdown", "segments": [12*60]*3, "running": False}),
        ("12 min periods (running)", {"mode": "countdown", "segments": [12*60]*3, "running": True}),
        ("15 min periods (stop clock)", {"mode": "countdown", "segments": [15*60]*3, "running": False}),
        ("15 min periods (running)", {"mode": "countdown", "segments": [15*60]*3, "running": True}),
    ],
    "Football": [
        ("20 min halves (running)", {"mode": "countdown", "segments": [20*60, 20*60], "running": True}),
        ("20 min halves (stop clock)", {"mode": "countdown", "segments": [20*60, 20*60], "running": False}),
        ("15 min halves (running)", {"mode": "countdown", "segments": [15*60, 15*60], "running": True}),
    ],
    "Softball": [
        ("6 innings (no timer)", {"mode": "none", "segments": [], "running": True}),
        ("7 innings (no timer)", {"mode": "none", "segments": [], "running": True}),
        ("45 min (running)", {"mode": "countdown", "segments": [45*60], "running": True}),
        ("60 min (running)", {"mode": "countdown", "segments": [60*60], "running": True}),
    ],
}

DEFAULT_TIMER = ("No Timer", {"mode": "none", "segments": [], "running": True})

# ---------- Stat Categories by Sport ----------
SPORT_STAT_FIELDS = {
    "Basketball": ["PTS", "AST", "REB", "BLK", "STL"],
    "Hockey": ["G", "A", "SOG"],
    "Softball": ["H", "2B", "3B", "HR", "RBI"],
    "Football": ["TD", "FG", "XP", "SACK"],
    "Kickball": ["R", "K", "OUTS"],
    "Euro": ["PTS", "AST"],
    "Speedball": ["G", "A"],
    "Soccer": ["G", "A"],
    "Lacrosse": ["G", "A"],
    "Volleyball": ["K", "A", "D"],
    "Street Hockey": ["G", "A"],
}

# ---------- Points / Weighting ----------
# You told me: Senior A Softball = 50. Use sensible ladders.
# These are "league points awarded for winning" (ties optional).
# Adjust any time.
BASE_SENIOR_A = {
    "Softball": 50,
    "Football": 40,
    "Basketball": 35,
    "Hockey": 35,
    "Kickball": 30,
    "Euro": 28,
    "Speedball": 28,
    "Soccer": 32,
    "Lacrosse": 32,
    "Volleyball": 28,
    "Street Hockey": 30,
}
LEVEL_DELTAS = {"A": 0, "B": -5, "C": -10, "D": -15}
LEAGUE_MULT = {"SOPH": 0.70, "JUNIOR": 0.85, "SENIOR": 1.00}  # makes combined standings proportional

# ---------- Sheet Structure ----------
# One Google Sheet. Separate worksheets per league and data type.
WS = {
    "ROSTERS": "Rosters_{LK}",
    "GAMES": "Games_{LK}",
    "STATS": "Stats_{LK}",
    "NON_GAME": "NonGame_{LK}",
    "HIGHLIGHTS": "Highlights_{LK}",
}

# ---------- Timezone ----------
ET = ZoneInfo("America/New_York")

# =========================
# Google Sheets Helpers (cached)
# =========================

def _require_secret(key: str):
    if key not in st.secrets:
        st.error(f"Missing Streamlit secret: {key}")
        st.stop()

@st.cache_resource(show_spinner=False)
def get_gspread_client():
    _require_secret("gcp_service_account")
    _require_secret("SHEET_ID")
    sa = st.secrets["gcp_service_account"]
    # Streamlit secrets might store this as dict already or as a JSON string.
    if isinstance(sa, str):
        sa = json.loads(sa)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)

def sheet_id() -> str:
    _require_secret("SHEET_ID")
    return st.secrets["SHEET_ID"]

def open_spreadsheet():
    client = get_gspread_client()
    return client.open_by_key(sheet_id())

def ws_name(kind: str, league_key: str) -> str:
    return WS[kind].format(LK=league_key)

def ensure_worksheet(name: str, headers: list[str]):
    sh = open_spreadsheet()
    try:
        w = sh.worksheet(name)
    except Exception:
        w = sh.add_worksheet(title=name, rows=2000, cols=max(10, len(headers) + 2))
        w.append_row(headers)
        return w

    # Ensure header row is correct (and unique)
    first = w.row_values(1)
    if not first:
        w.append_row(headers)
    else:
        # If duplicates / wrong headers, rewrite row 1 safely:
        if len(set(first)) != len(first) or first[:len(headers)] != headers:
            w.update("1:1", [headers])
    return w

def ensure_all_sheets():
    # Create all worksheets once.
    for league, lk in LEAGUE_KEYS.items():
        ensure_worksheet(ws_name("ROSTERS", lk), ["player_id", "first_name", "last_name", "team_name", "bunk"])
        ensure_worksheet(ws_name("GAMES", lk), [
            "game_id", "timestamp_et", "sport", "level", "team1", "team2",
            "team1_score", "team2_score", "winner", "points_awarded"
        ])
        ensure_worksheet(ws_name("STATS", lk), [
            "game_id", "timestamp_et", "sport", "level", "player_id",
            "first_name", "last_name", "team_name", "stat", "value"
        ])
        ensure_worksheet(ws_name("NON_GAME", lk), [
            "ng_id", "timestamp_et", "team_name", "reason", "other_reason", "points"
        ])
        ensure_worksheet(ws_name("HIGHLIGHTS", lk), [
            "hl_id", "timestamp_et", "filename", "mime", "bytes_b64"
        ])

# ---------- Cached READS ----------
@st.cache_data(ttl=20, show_spinner=False)
def read_ws_df(ws_title: str) -> pd.DataFrame:
    sh = open_spreadsheet()
    w = sh.worksheet(ws_title)
    values = w.get_all_values()
    if not values or len(values) < 1:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df

def clear_cache_after_write():
    st.cache_data.clear()

# ---------- Writes ----------
def append_row(ws_title: str, row: list):
    sh = open_spreadsheet()
    w = sh.worksheet(ws_title)
    w.append_row(row, value_input_option="USER_ENTERED")
    clear_cache_after_write()

def overwrite_ws_from_df(ws_title: str, df: pd.DataFrame):
    sh = open_spreadsheet()
    w = sh.worksheet(ws_title)
    w.clear()
    w.append_row(list(df.columns))
    if len(df) > 0:
        w.append_rows(df.astype(str).values.tolist(), value_input_option="USER_ENTERED")
    clear_cache_after_write()

# =========================
# Scoring Logic
# =========================

def game_points_award(league_key: str, sport: str, level: str) -> int:
    # Determine base senior A for sport
    base = BASE_SENIOR_A.get(sport, 30)
    delta = LEVEL_DELTAS.get(level, 0)
    raw = base + delta
    mult = LEAGUE_MULT.get(league_key, 1.0)
    pts = int(round(raw * mult))
    # Never below 5
    return max(5, pts)

def now_et_str():
    return datetime.now(ET).strftime("%Y-%m-%d %I:%M:%S %p")

def new_id(prefix: str):
    return f"{prefix}-{int(time.time()*1000)}-{np.random.randint(1000,9999)}"

# =========================
# UI Helpers
# =========================

def inject_css():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #0b1f3a 0%, #071529 40%, #071529 100%);
                color: white;
            }}
            h1, h2, h3, h4, h5, h6, p, div, span, label {{
                color: white !important;
            }}
            .block-container {{
                padding-top: 1.5rem;
            }}
            .scoreboard {{
                border: 2px solid {BAUERCREST_GOLD};
                border-radius: 16px;
                padding: 18px;
                background: rgba(255,255,255,0.06);
            }}
            .big-timer {{
                font-size: 56px;
                font-weight: 800;
                letter-spacing: 1px;
            }}
            .big-score {{
                font-size: 64px;
                font-weight: 900;
            }}
            .pill {{
                display:inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                background: rgba(245,197,66,0.18);
                border: 1px solid rgba(245,197,66,0.45);
                margin-right: 6px;
                font-weight: 700;
            }}
            .small-muted {{
                opacity: 0.85;
                font-size: 0.95rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def sidebar_header():
    st.sidebar.markdown("## Manager")
    st.sidebar.markdown("<div class='small-muted'>Bauercrest Crest League Tools</div>", unsafe_allow_html=True)
    try:
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    except Exception:
        pass

def select_league_sidebar():
    league = st.sidebar.selectbox("League (for Setup / Scoring / Stats)", LEAGUES, index=2)
    lk = LEAGUE_KEYS[league]
    st.sidebar.markdown(f"<div class='small-muted'>Managing league data for: <b>{league}</b></div>", unsafe_allow_html=True)
    return league, lk

def nav_sidebar():
    return st.sidebar.radio(
        "Go to",
        [
            "Setup",
            "Live Games (Create/Open)",
            "Run Live Game",
            "Standings",
            "Leaderboards",
            "Highlights",
            "Non-Game Points",
            "Display Board",
            "Admin / Clear Data",
        ]
    )

# =========================
# Data Accessors
# =========================

def roster_df(lk: str) -> pd.DataFrame:
    df = read_ws_df(ws_name("ROSTERS", lk))
    if df.empty:
        return df
    # ensure types
    for c in ["player_id", "first_name", "last_name", "team_name", "bunk"]:
        if c not in df.columns:
            return pd.DataFrame(columns=["player_id", "first_name", "last_name", "team_name", "bunk"])
    return df

def games_df(lk: str) -> pd.DataFrame:
    df = read_ws_df(ws_name("GAMES", lk))
    return df

def stats_df(lk: str) -> pd.DataFrame:
    df = read_ws_df(ws_name("STATS", lk))
    return df

def non_game_df(lk: str) -> pd.DataFrame:
    df = read_ws_df(ws_name("NON_GAME", lk))
    return df

def highlights_df(lk: str) -> pd.DataFrame:
    df = read_ws_df(ws_name("HIGHLIGHTS", lk))
    return df

# =========================
# Live Game Session State
# =========================

def init_live_state():
    if "live_game" not in st.session_state:
        st.session_state.live_game = None
    if "timer" not in st.session_state:
        st.session_state.timer = {}
    if "stat_log" not in st.session_state:
        st.session_state.stat_log = pd.DataFrame(columns=["player_id", "name", "team", "stat", "delta", "t"])

def reset_live_state():
    st.session_state.live_game = None
    st.session_state.timer = {}
    st.session_state.stat_log = pd.DataFrame(columns=["player_id", "name", "team", "stat", "delta", "t"])

# =========================
# Pages
# =========================

def page_setup(lk: str, league_name: str):
    st.title("Setup")
    st.markdown(f"<span class='pill'>Upload roster for {league_name}</span>", unsafe_allow_html=True)
    st.divider()

    df = roster_df(lk)
    st.subheader("Current Roster")
    if df.empty:
        st.info("No roster uploaded yet for this league.")
    else:
        st.dataframe(df, use_container_width=True)

    st.subheader("Upload New Roster CSV")
    st.caption("CSV must contain columns: player_id, first_name, last_name, team_name, bunk")

    up = st.file_uploader("Upload roster CSV", type=["csv"], key=f"roster_upload_{lk}")
    if up is not None:
        new_df = pd.read_csv(up)
        required = ["player_id", "first_name", "last_name", "team_name", "bunk"]
        missing = [c for c in required if c not in new_df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            new_df = new_df[required].copy()
            overwrite_ws_from_df(ws_name("ROSTERS", lk), new_df)
            st.success("Roster uploaded.")
            st.rerun()

def page_live_games_home(lk: str, league_name: str):
    st.title("Live Games (Create / Open)")
    st.markdown(f"<span class='pill'>{league_name}</span>", unsafe_allow_html=True)
    st.divider()

    r = roster_df(lk)
    if r.empty:
        st.warning("Upload a roster first in Setup.")
        return

    teams = sorted(r["team_name"].dropna().unique().tolist())
    c1, c2 = st.columns([1,1])

    with c1:
        st.subheader("Create a New Live Game")
        sport = st.selectbox("Sport", SPORTS, index=0, key=f"sport_{lk}")
        level = st.selectbox("Level", ["A","B","C","D"], index=1, key=f"level_{lk}")

        presets = TIMER_PRESETS.get(sport, [DEFAULT_TIMER])
        preset_label = st.selectbox("Timer preset", [p[0] for p in presets], index=0, key=f"preset_{lk}")
        preset_cfg = dict(presets[[p[0] for p in presets].index(preset_label)][1])

        mode = st.radio("Game type", ["1 team vs 1 team", "2 teams vs 2 teams"], horizontal=True, key=f"gt_{lk}")

        if mode == "1 team vs 1 team":
            t1 = st.selectbox("Team 1", teams, key=f"t1_{lk}")
            t2 = st.selectbox("Team 2", [t for t in teams if t != t1], key=f"t2_{lk}")
            team1_label = t1
            team2_label = t2
            team1_teams = [t1]
            team2_teams = [t2]
        else:
            t1a = st.selectbox("Side A - Team 1", teams, key=f"t1a_{lk}")
            t1b = st.selectbox("Side A - Team 2", [t for t in teams if t != t1a], key=f"t1b_{lk}")
            remaining = [t for t in teams if t not in [t1a, t1b]]
            t2a = st.selectbox("Side B - Team 1", remaining, key=f"t2a_{lk}")
            t2b = st.selectbox("Side B - Team 2", [t for t in remaining if t != t2a], key=f"t2b_{lk}")
            team1_label = f"{t1a}+{t1b}"
            team2_label = f"{t2a}+{t2b}"
            team1_teams = [t1a, t1b]
            team2_teams = [t2a, t2b]

        st.caption("Lineup selector: choose only the players actually playing in this game.")
        # Active lineup: filter roster by participating teams
        allowed_teams = set(team1_teams + team2_teams)
        candidates = r[r["team_name"].isin(list(allowed_teams))].copy()
        candidates["name"] = candidates["first_name"].astype(str) + " " + candidates["last_name"].astype(str)

        default_players = candidates["name"].tolist()
        lineup = st.multiselect(
            "Active players (in this game)",
            options=default_players,
            default=default_players,
            key=f"lineup_{lk}"
        )

        if st.button("Create Live Game", type="primary", use_container_width=True):
            game_id = new_id(f"LIVE-{LEAGUE_KEYS[league_name]}")
            live = {
                "game_id": game_id,
                "created_et": now_et_str(),
                "lk": lk,
                "league_name": league_name,
                "sport": sport,
                "level": level,
                "preset": preset_label,
                "preset_cfg": preset_cfg,
                "team1_label": team1_label,
                "team2_label": team2_label,
                "team1_teams": team1_teams,
                "team2_teams": team2_teams,
                "score1": 0,
                "score2": 0,
                "active_players": lineup,
                "status": "IN_PROGRESS",
            }
            st.session_state.live_game = live
            init_timer_from_preset(preset_cfg)
            st.success("Live game created. Go to 'Run Live Game'.")
            # No sheet writes here. This is per-user session.
            st.rerun()

    with c2:
        st.subheader("Open an Existing Live Game")
        st.caption("Live games are per-device/session (so multiple games can run at once).")
        if st.session_state.get("live_game"):
            lg = st.session_state.live_game
            st.info(f"Currently loaded: {lg['sport']} ({lg['level']}) {lg['team1_label']} vs {lg['team2_label']}")
            if st.button("Clear current live game"):
                reset_live_state()
                st.rerun()
        else:
            st.info("No live game loaded on this device yet. Create one on the left.")

def init_timer_from_preset(cfg: dict):
    # cfg: mode, segments, running
    st.session_state.timer = {
        "mode": cfg.get("mode", "none"),
        "segments": cfg.get("segments", []),
        "segment_index": 0,
        "running_clock": cfg.get("running", True),
        "running": False,
        "last_tick": None,
        "remaining": cfg.get("segments", [0])[0] if cfg.get("segments") else 0,
        "elapsed": 0,
    }

def format_mmss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def timer_tick():
    t = st.session_state.timer
    if not t.get("running"):
        return
    now = time.time()
    last = t.get("last_tick")
    if last is None:
        t["last_tick"] = now
        return
    dt = now - last
    t["last_tick"] = now

    if t["mode"] == "countdown":
        t["remaining"] = max(0, t["remaining"] - dt)
    elif t["mode"] == "countup":
        t["elapsed"] = t["elapsed"] + dt

def timer_display_str():
    t = st.session_state.timer
    if t.get("mode") == "none":
        return "—"
    if t.get("mode") == "countdown":
        return format_mmss(int(t.get("remaining", 0)))
    if t.get("mode") == "countup":
        return format_mmss(int(t.get("elapsed", 0)))
    return "—"

def advance_segment_if_needed():
    t = st.session_state.timer
    if t.get("mode") != "countdown":
        return
    if not t.get("segments"):
        return
    if int(t.get("remaining", 0)) > 0:
        return
    # move to next segment
    idx = t.get("segment_index", 0)
    if idx < len(t["segments"]) - 1:
        t["segment_index"] = idx + 1
        t["remaining"] = t["segments"][t["segment_index"]]
        t["running"] = False
        t["last_tick"] = None

def page_run_live_game(lk: str, league_name: str):
    init_live_state()
    st.title("Run Live Game")
    st.markdown(f"<span class='pill'>{league_name}</span>", unsafe_allow_html=True)
    st.divider()

    lg = st.session_state.get("live_game")
    if not lg or lg.get("lk") != lk:
        st.warning("No live game loaded for this league on this device. Go to Live Games (Create/Open).")
        return

    # Tick timer locally (no Sheets reads)
    timer_tick()
    advance_segment_if_needed()

    # SCOREBOARD
    st.markdown("<div class='scoreboard'>", unsafe_allow_html=True)
    cA, cB, cC = st.columns([2.2, 1.2, 2.2])

    with cA:
        st.markdown(f"### {lg['team1_label']}")
        st.markdown(f"<div class='big-score'>{lg['score1']}</div>", unsafe_allow_html=True)

    with cB:
        seg = st.session_state.timer.get("segment_index", 0) + 1
        total_seg = len(st.session_state.timer.get("segments", []))
        seg_txt = f"Segment {seg}/{total_seg}" if total_seg else "Timer"
        st.markdown(f"<div class='small-muted'>{lg['sport']} • {lg['level']} • {seg_txt}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-timer'>{timer_display_str()}</div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶ Start", use_container_width=True):
                st.session_state.timer["running"] = True
                st.session_state.timer["last_tick"] = None
                st.rerun()
        with b2:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.timer["running"] = False
                st.session_state.timer["last_tick"] = None
                st.rerun()
        with b3:
            if st.button("⏭ Next", use_container_width=True):
                # force next segment
                t = st.session_state.timer
                if t.get("segments"):
                    idx = t.get("segment_index", 0)
                    if idx < len(t["segments"]) - 1:
                        t["segment_index"] = idx + 1
                        t["remaining"] = t["segments"][t["segment_index"]]
                        t["running"] = False
                        t["last_tick"] = None
                st.rerun()

    with cC:
        st.markdown(f"### {lg['team2_label']}")
        st.markdown(f"<div class='big-score'>{lg['score2']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    # SCORE BUTTONS (big + simple)
    s1, s2 = st.columns(2)

    def score_buttons(side: int):
        # side 1 => team1, side 2 => team2
        col = s1 if side == 1 else s2
        with col:
            st.subheader("Score Controls")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("➕ +1", use_container_width=True, key=f"p1_{side}"):
                    if side == 1: lg["score1"] += 1
                    else: lg["score2"] += 1
                    st.rerun()
            with c2:
                if st.button("➕ +2", use_container_width=True, key=f"p2_{side}"):
                    if side == 1: lg["score1"] += 2
                    else: lg["score2"] += 2
                    st.rerun()
            with c3:
                if st.button("➕ +3", use_container_width=True, key=f"p3_{side}"):
                    if side == 1: lg["score1"] += 3
                    else: lg["score2"] += 3
                    st.rerun()
            d1, d2, d3 = st.columns(3)
            with d1:
                if st.button("➖ -1", use_container_width=True, key=f"m1_{side}"):
                    if side == 1: lg["score1"] = max(0, lg["score1"] - 1)
                    else: lg["score2"] = max(0, lg["score2"] - 1)
                    st.rerun()
            with d2:
                if st.button("Reset", use_container_width=True, key=f"rst_{side}"):
                    if side == 1: lg["score1"] = 0
                    else: lg["score2"] = 0
                    st.rerun()
            with d3:
                st.write("")

    score_buttons(1)
    score_buttons(2)

    st.divider()

    # STATS (per sport) - no sheet writes; only local tallies
    r = roster_df(lk)
    r["name"] = r["first_name"].astype(str) + " " + r["last_name"].astype(str)

    # Only show active players
    active_names = set(lg["active_players"])
    r_active = r[r["name"].isin(list(active_names))].copy()

    # Softball: allow batting order
    if lg["sport"] == "Softball":
        st.subheader("Softball Batting Order (optional)")
        st.caption("Drag/drop not native in Streamlit. Use the order number column and sort.")
        if "batting_order" not in st.session_state:
            st.session_state.batting_order = {n: i+1 for i, n in enumerate(r_active["name"].tolist())}
        order_df = pd.DataFrame({
            "name": r_active["name"],
            "order": [st.session_state.batting_order.get(n, 99) for n in r_active["name"]],
        })
        edited = st.data_editor(order_df, use_container_width=True, num_rows="fixed", key="bat_order_editor")
        # store order
        for _, row in edited.iterrows():
            st.session_state.batting_order[row["name"]] = int(row["order"])
        # apply sort
        r_active["order"] = r_active["name"].map(lambda n: st.session_state.batting_order.get(n, 999))
        r_active = r_active.sort_values(["order", "team_name", "last_name"], ascending=True)

    stat_fields = SPORT_STAT_FIELDS.get(lg["sport"], ["PTS", "AST"])

    st.subheader("Live Stats")
    st.caption("Tap buttons during the game. Nothing writes to Google Sheets until you press SAVE at the end.")
    st.markdown("<div class='small-muted'>Green flash confirms the tap worked.</div>", unsafe_allow_html=True)

    # initialize per-player stat totals
    if "player_stats" not in st.session_state or st.session_state.get("player_stats_game_id") != lg["game_id"]:
        st.session_state.player_stats_game_id = lg["game_id"]
        st.session_state.player_stats = {}
        for _, row in r_active.iterrows():
            pid = row["player_id"]
            st.session_state.player_stats[pid] = {f: 0 for f in stat_fields}

    # UI: player cards with buttons
    for team, group in r_active.groupby("team_name"):
        st.markdown(f"### {team}")
        for _, row in group.iterrows():
            pid = row["player_id"]
            pname = row["name"]
            cols = st.columns([2.2] + [1]*len(stat_fields))
            cols[0].markdown(f"**{pname}**")
            for i, stat in enumerate(stat_fields):
                key = f"btn_{lg['game_id']}_{pid}_{stat}"
                if cols[i+1].button(f"+{stat}", key=key, use_container_width=True):
                    st.session_state.player_stats[pid][stat] += 1
                    # confirmation log
                    st.session_state.stat_log = pd.concat([
                        st.session_state.stat_log,
                        pd.DataFrame([{
                            "player_id": pid,
                            "name": pname,
                            "team": row["team_name"],
                            "stat": stat,
                            "delta": 1,
                            "t": datetime.now(ET).strftime("%I:%M:%S %p")
                        }])
                    ], ignore_index=True)
                    # tiny confirmation flash
                    cols[0].success(f"+{stat}", icon="✅")
                    st.rerun()

    st.divider()
    st.subheader("This Game Stat Log (updates every tap)")
    if len(st.session_state.stat_log) == 0:
        st.info("No live stats logged yet.")
    else:
        st.dataframe(st.session_state.stat_log.tail(200), use_container_width=True, height=260)

    st.divider()

    # SAVE GAME (single write to Sheets at end)
    st.subheader("Finish Game")
    c_end1, c_end2 = st.columns([1, 1])

    with c_end1:
        if st.button("✅ Save Game + Stats to League", type="primary", use_container_width=True):
            # Determine winner
            if lg["score1"] > lg["score2"]:
                winner = lg["team1_label"]
            elif lg["score2"] > lg["score1"]:
                winner = lg["team2_label"]
            else:
                winner = "TIE"

            pts = game_points_award(lk, lg["sport"], lg["level"])
            game_id = new_id(f"GAME-{lk}")

            # Write Games row
            append_row(ws_name("GAMES", lk), [
                game_id,
                now_et_str(),
                lg["sport"],
                lg["level"],
                lg["team1_label"],
                lg["team2_label"],
                lg["score1"],
                lg["score2"],
                winner,
                pts if winner != "TIE" else 0
            ])

            # Write Stats rows
            # Convert local stats dict to rows
            for _, row in r_active.iterrows():
                pid = row["player_id"]
                totals = st.session_state.player_stats.get(pid, {})
                for stat, val in totals.items():
                    if int(val) != 0:
                        append_row(ws_name("STATS", lk), [
                            game_id,
                            now_et_str(),
                            lg["sport"],
                            lg["level"],
                            pid,
                            row["first_name"],
                            row["last_name"],
                            row["team_name"],
                            stat,
                            int(val)
                        ])

            st.success("Saved! Standings + leaderboards will update for everyone.")
            reset_live_state()
            st.rerun()

    with c_end2:
        if st.button("🗑️ Discard This Live Game (no save)", use_container_width=True):
            reset_live_state()
            st.warning("Discarded.")
            st.rerun()

def page_standings(lk: str, league_name: str):
    st.title("Standings")
    st.markdown(f"<span class='pill'>{league_name}</span>", unsafe_allow_html=True)
    st.divider()

    r = roster_df(lk)
    g = games_df(lk)
    ng = non_game_df(lk)

    if r.empty:
        st.warning("No roster uploaded yet.")
        return

    teams = sorted(r["team_name"].dropna().unique().tolist())

    # in-game points
    in_points = {t: 0 for t in teams}
    wins = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}

    if not g.empty:
        # numeric
        for col in ["team1_score", "team2_score", "points_awarded"]:
            if col in g.columns:
                g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0).astype(int)

        for _, row in g.iterrows():
            t1 = row.get("team1")
            t2 = row.get("team2")
            winner = row.get("winner")
            pts = int(row.get("points_awarded", 0))

            if winner and winner != "TIE":
                # winner label might be combined (A+B). Award to that label doesn't map clean.
                # We only award if it matches an actual team name.
                if winner in in_points:
                    in_points[winner] += pts
                    wins[winner] += 1
                    # loser
                    loser = t1 if winner == t2 else t2
                    if loser in losses:
                        losses[loser] += 1

    # non-game points
    non_points = {t: 0 for t in teams}
    if not ng.empty and "team_name" in ng.columns and "points" in ng.columns:
        ng["points"] = pd.to_numeric(ng["points"], errors="coerce").fillna(0).astype(int)
        for _, row in ng.iterrows():
            t = row["team_name"]
            if t in non_points:
                non_points[t] += int(row["points"])

    show_total = st.toggle("Include non-game points in Total Points", value=True)

    rows = []
    for t in teams:
        total = in_points[t] + (non_points[t] if show_total else 0)
        rows.append({
            "Team": t,
            "Wins": wins[t],
            "Losses": losses[t],
            "In-Game Points": in_points[t],
            "Non-Game Points": non_points[t],
            "Total Points": total
        })
    df = pd.DataFrame(rows).sort_values(["Total Points", "Wins"], ascending=False)
    st.dataframe(df, use_container_width=True)

def page_leaderboards(lk: str, league_name: str):
    st.title("Leaderboards")
    st.markdown(f"<span class='pill'>{league_name}</span>", unsafe_allow_html=True)
    st.divider()

    s = stats_df(lk)
    if s.empty:
        st.info("No stats yet.")
        return

    # numeric
    s["value"] = pd.to_numeric(s.get("value", 0), errors="coerce").fillna(0).astype(int)

    sport = st.selectbox("Sport", ["All"] + sorted(s["sport"].dropna().unique().tolist()))
    if sport != "All":
        s = s[s["sport"] == sport]

    stat = st.selectbox("Stat Category", sorted(s["stat"].dropna().unique().tolist()))
    s2 = s[s["stat"] == stat].copy()
    if s2.empty:
        st.info("No entries for that stat.")
        return

    s2["name"] = s2["first_name"].astype(str) + " " + s2["last_name"].astype(str)
    lb = (
        s2.groupby(["player_id", "name", "team_name"], as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .head(25)
    )
    st.dataframe(lb, use_container_width=True)

def page_highlights(lk: str, league_name: str):
    st.title("Highlights")
    st.markdown(f"<span class='pill'>{league_name}</span>", unsafe_allow_html=True)
    st.divider()

    st.subheader("Upload highlight videos (files)")
    st.caption("These store in Google Sheets as base64. Keep files small (best: < 15–25MB).")

    up = st.file_uploader("Upload a video file", type=["mp4", "mov", "m4v"], accept_multiple_files=True)
    if up:
        for f in up:
            raw = f.read()
            # basic size guard
            if len(raw) > 25 * 1024 * 1024:
                st.warning(f"{f.name} skipped (too large for Sheets storage).")
                continue
            b64 = io.BytesIO(raw).getvalue()
            import base64
            b64s = base64.b64encode(b64).decode("utf-8")
            append_row(ws_name("HIGHLIGHTS", lk), [
                new_id("HL"),
                now_et_str(),
                f.name,
                f.type,
                b64s
            ])
        st.success("Uploaded.")
        st.rerun()

    st.subheader("Today’s highlights (playback)")
    df = highlights_df(lk)
    if df.empty:
        st.info("No highlights uploaded yet.")
        return

    # show most recent first
    df = df.tail(12).iloc[::-1].copy()
    for _, row in df.iterrows():
        st.markdown(f"**{row.get('filename','(file)')}**  •  {row.get('timestamp_et','')}")
        try:
            import base64
            data = base64.b64decode(row.get("bytes_b64", "").encode("utf-8"))
            st.video(data)
        except Exception:
            st.warning("Could not play this file (corrupt or too large).")

def page_non_game_points(lk: str, league_name: str):
    st.title("Non-Game Points")
    st.markdown(f"<span class='pill'>{league_name}</span>", unsafe_allow_html=True)
    st.divider()

    r = roster_df(lk)
    if r.empty:
        st.warning("Upload a roster first.")
        return
    teams = sorted(r["team_name"].dropna().unique().tolist())

    # Categories from Crest proposal ideas + camp reality
    categories = [
        "Sportsmanship",
        "Spirit / Energy",
        "Attendance / Punctuality",
        "Clean Field / Respect Equipment",
        "Helping another team",
        "Leadership moment",
        "Community service / camp pride",
        "Other (type it)"
    ]

    team = st.selectbox("Team", teams)
    cat = st.selectbox("Reason", categories)
    other = ""
    if cat == "Other (type it)":
        other = st.text_input("Other reason")
    pts = st.number_input("Points to award", min_value=-50, max_value=200, value=5, step=1)

    if st.button("Award Non-Game Points", type="primary"):
        append_row(ws_name("NON_GAME", lk), [
            new_id("NG"),
            now_et_str(),
            team,
            cat,
            other,
            int(pts)
        ])
        st.success("Awarded.")
        st.rerun()

    st.subheader("History")
    df = non_game_df(lk)
    if df.empty:
        st.info("No non-game points recorded yet.")
    else:
        df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
        st.dataframe(df.sort_values("timestamp_et", ascending=False), use_container_width=True)

def page_display_board():
    st.title("Display Board")
    st.caption("Pick what you want to show, then click 'Open Full Screen Board' for TV mode.")
    st.divider()

    league = st.selectbox("League to Display", LEAGUES, index=2)
    lk = LEAGUE_KEYS[league]

    mode = st.selectbox("Board Mode", ["Standings", "Leaderboards", "Highlights (loop)"])

    # Build a query-string-ish state using session_state
    st.session_state.display_mode = mode
    st.session_state.display_lk = lk
    st.session_state.display_league = league

    if st.button("Open Full Screen Board (TV)", type="primary"):
        st.session_state.fullscreen = True
        st.rerun()

    # inline preview
    st.subheader("Preview")
    render_board(fullscreen=False)

def render_board(fullscreen: bool):
    lk = st.session_state.get("display_lk", LEAGUE_KEYS["Senior League"])
    league = st.session_state.get("display_league", "Senior League")
    mode = st.session_state.get("display_mode", "Standings")

    if fullscreen:
        st.markdown(
            """
            <style>
                header, footer, [data-testid="stSidebar"] {display:none !important;}
                .block-container {padding-top: 0.5rem !important;}
            </style>
            """,
            unsafe_allow_html=True
        )

    st.markdown(f"# {league} — {mode}")

    if mode == "Standings":
        r = roster_df(lk)
        if r.empty:
            st.info("No roster.")
            return
        teams = sorted(r["team_name"].dropna().unique().tolist())
        g = games_df(lk)
        ng = non_game_df(lk)

        in_points = {t: 0 for t in teams}
        wins = {t: 0 for t in teams}
        losses = {t: 0 for t in teams}
        if not g.empty:
            g["points_awarded"] = pd.to_numeric(g.get("points_awarded", 0), errors="coerce").fillna(0).astype(int)
            for _, row in g.iterrows():
                winner = row.get("winner")
                pts = int(row.get("points_awarded", 0))
                if winner in in_points:
                    in_points[winner] += pts
                    wins[winner] += 1

        non_points = {t: 0 for t in teams}
        if not ng.empty:
            ng["points"] = pd.to_numeric(ng.get("points", 0), errors="coerce").fillna(0).astype(int)
            for _, row in ng.iterrows():
                t = row.get("team_name")
                if t in non_points:
                    non_points[t] += int(row.get("points", 0))

        rows = []
        for t in teams:
            rows.append({
                "Team": t,
                "W": wins[t],
                "In-Game": in_points[t],
                "Non-Game": non_points[t],
                "Total": in_points[t] + non_points[t]
            })
        df = pd.DataFrame(rows).sort_values(["Total", "W"], ascending=False)
        st.dataframe(df, use_container_width=True, height=700)

    elif mode == "Leaderboards":
        s = stats_df(lk)
        if s.empty:
            st.info("No stats yet.")
            return
        s["value"] = pd.to_numeric(s.get("value", 0), errors="coerce").fillna(0).astype(int)
        # cycle through sports automatically in fullscreen (simple)
        sports = sorted(s["sport"].dropna().unique().tolist())
        if not sports:
            st.info("No sport rows.")
            return
        if "lb_index" not in st.session_state:
            st.session_state.lb_index = 0
        sport = sports[st.session_state.lb_index % len(sports)]
        st.session_state.lb_index += 1

        st.markdown(f"## {sport} Leaders")
        s2 = s[s["sport"] == sport].copy()
        if s2.empty:
            return
        stat = st.selectbox("Stat", sorted(s2["stat"].dropna().unique().tolist()), key="board_stat")
        s3 = s2[s2["stat"] == stat].copy()
        if s3.empty:
            st.info("No rows for that stat.")
            return

        s3["name"] = s3["first_name"].astype(str) + " " + s3["last_name"].astype(str)
        lb = (
            s3.groupby(["player_id", "name", "team_name"], as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
            .head(15)
        )
        st.dataframe(lb, use_container_width=True, height=700)

        if fullscreen:
            # refresh view without hitting Sheets too often (cached ttl handles reads)
            time.sleep(3)
            st.rerun()

    elif mode == "Highlights (loop)":
        df = highlights_df(lk)
        if df.empty:
            st.info("No highlights uploaded.")
            return
        df = df.tail(10).copy()

        # loop index in session
        if "hl_idx" not in st.session_state:
            st.session_state.hl_idx = 0
        row = df.iloc[st.session_state.hl_idx % len(df)]
        st.session_state.hl_idx += 1

        st.markdown(f"## {row.get('filename','Highlight')}")
        try:
            import base64
            data = base64.b64decode(row.get("bytes_b64", "").encode("utf-8"))
            st.video(data)
        except Exception:
            st.warning("Could not play highlight.")
        if fullscreen:
            time.sleep(6)
            st.rerun()

def page_admin_global():
    st.title("Admin / Clear Data")
    st.divider()

    # Password protect admin
    ADMIN_PASS = "Hyaffa26"
    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False

    if not st.session_state.admin_ok:
        pw = st.text_input("Enter admin password", type="password")
        if st.button("Unlock Admin"):
            if pw == ADMIN_PASS:
                st.session_state.admin_ok = True
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Wrong password.")
        return

    if st.button("Lock Admin"):
        st.session_state.admin_ok = False
        st.rerun()

    st.subheader("Clear Options")
    league = st.selectbox("Which league?", LEAGUES, index=2)
    lk = LEAGUE_KEYS[league]

    target = st.selectbox("What do you want to delete?", [
        "Roster",
        "Games",
        "Stats",
        "Non-Game Points",
        "Highlights",
        "ALL league data (danger)"
    ])

    if target == "ALL league data (danger)":
        st.warning("This will wipe everything for this league.")
    else:
        st.info(f"This will wipe: {target} for {league}")

    if st.button("DELETE selected data", type="primary"):
        if target == "Roster":
            overwrite_ws_from_df(ws_name("ROSTERS", lk), pd.DataFrame(columns=["player_id","first_name","last_name","team_name","bunk"]))
        elif target == "Games":
            overwrite_ws_from_df(ws_name("GAMES", lk), pd.DataFrame(columns=[
                "game_id","timestamp_et","sport","level","team1","team2","team1_score","team2_score","winner","points_awarded"
            ]))
        elif target == "Stats":
            overwrite_ws_from_df(ws_name("STATS", lk), pd.DataFrame(columns=[
                "game_id","timestamp_et","sport","level","player_id","first_name","last_name","team_name","stat","value"
            ]))
        elif target == "Non-Game Points":
            overwrite_ws_from_df(ws_name("NON_GAME", lk), pd.DataFrame(columns=[
                "ng_id","timestamp_et","team_name","reason","other_reason","points"
            ]))
        elif target == "Highlights":
            overwrite_ws_from_df(ws_name("HIGHLIGHTS", lk), pd.DataFrame(columns=[
                "hl_id","timestamp_et","filename","mime","bytes_b64"
            ]))
        else:
            overwrite_ws_from_df(ws_name("ROSTERS", lk), pd.DataFrame(columns=["player_id","first_name","last_name","team_name","bunk"]))
            overwrite_ws_from_df(ws_name("GAMES", lk), pd.DataFrame(columns=[
                "game_id","timestamp_et","sport","level","team1","team2","team1_score","team2_score","winner","points_awarded"
            ]))
            overwrite_ws_from_df(ws_name("STATS", lk), pd.DataFrame(columns=[
                "game_id","timestamp_et","sport","level","player_id","first_name","last_name","team_name","stat","value"
            ]))
            overwrite_ws_from_df(ws_name("NON_GAME", lk), pd.DataFrame(columns=[
                "ng_id","timestamp_et","team_name","reason","other_reason","points"
            ]))
            overwrite_ws_from_df(ws_name("HIGHLIGHTS", lk), pd.DataFrame(columns=[
                "hl_id","timestamp_et","filename","mime","bytes_b64"
            ]))
        st.success("Deleted.")
        st.rerun()

# =========================
# Main
# =========================

def main():
    st.set_page_config(page_title="Bauercrest League Manager", layout="wide")
    inject_css()
    sidebar_header()

    # Make sure sheets exist (creates once; reads are cached anyway)
    try:
        ensure_all_sheets()
    except Exception as e:
        st.error("Google Sheets connection failed. Check secrets + API enabled + sharing.")
        st.exception(e)
        st.stop()

    league_name, lk = select_league_sidebar()
    page = nav_sidebar()

    if "fullscreen" in st.session_state and st.session_state.fullscreen:
        # Fullscreen board mode
        render_board(fullscreen=True)
        if st.button("Exit Full Screen"):
            st.session_state.fullscreen = False
            st.rerun()
        return

    if page == "Setup":
        page_setup(lk, league_name)
    elif page == "Live Games (Create/Open)":
        page_live_games_home(lk, league_name)
    elif page == "Run Live Game":
        page_run_live_game(lk, league_name)
    elif page == "Standings":
        page_standings(lk, league_name)
    elif page == "Leaderboards":
        page_leaderboards(lk, league_name)
    elif page == "Highlights":
        page_highlights(lk, league_name)
    elif page == "Non-Game Points":
        page_non_game_points(lk, league_name)
    elif page == "Display Board":
        page_display_board()
    elif page == "Admin / Clear Data":
        page_admin_global()
    else:
        st.info("Select a page from the sidebar.")

if __name__ == "__main__":
    main()
