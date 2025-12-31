# crest_live_league_manager_WORKING.py
# Camp Bauercrest - Live League Manager (Streamlit + Supabase + Google Sheets)
# Focus: fast, reliable live scoring + stats for multiple simultaneous games.

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Google Sheets deps
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:  # pragma: no cover
    gspread = None
    Credentials = None

# Supabase (supabase-py)
try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Bauercrest Live League Manager"
LOGO_PATH = "logo-header-2.png"

LEAGUES = [
    ("sophomore", "Sophomores"),
    ("junior", "Juniors"),
    ("seniors", "Seniors"),
]
LEAGUE_KEYS = [k for k, _ in LEAGUES]

LEVELS = ["A", "B", "C", "D"]
SPORTS = ["Basketball", "Softball"]  # extend later

# Default timer lengths
DEFAULT_DURATIONS = {
    "Basketball": 20 * 60,  # 20:00
    "Softball": 0,          # no clock by default
}

# Stat buttons by sport (you asked for tap-fast counselor workflow)
STAT_BUTTONS = {
    "Basketball": [
        ("PTS_1", "+1"),
        ("PTS_2", "+2"),
        ("PTS_3", "+3"),
        ("AST", "+Ast"),
        ("REB", "+Reb"),
        ("STL", "+Stl"),
        ("BLK", "+Blk"),
        ("TO", "+TO"),
    ],
    "Softball": [
        ("RUN", "+Run"),     # also bumps team score by 1 if enabled
        ("HIT", "+Hit"),
        ("RBI", "+RBI"),
    ],
}

# -----------------------------
# Helpers
# -----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_utc(dt: Optional[datetime] = None) -> str:
    return (dt or now_utc()).isoformat()

def epoch_now() -> float:
    return time.time()

def secrets_get_first(*keys: str, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default

@st.cache_resource
def get_supabase():
    if create_client is None:
        raise RuntimeError("supabase-py is not installed in this environment.")
    url = secrets_get_first("supabase_url", "SUPABASE_URL")
    key = secrets_get_first("supabase_service_role_key", "SUPABASE_SERVICE_ROLE_KEY",
                            "supabase_anon_key", "SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing Supabase secrets: supabase_url and (service_role_key or anon_key).")
    return create_client(url, key)

def sb_select(table: str, match: Optional[Dict] = None, order: Optional[str] = None, limit: Optional[int] = None):
    sb = get_supabase()
    q = sb.table(table).select("*")
    if match:
        for k, v in match.items():
            q = q.eq(k, v)
    if order:
        # order like "created_at.desc"
        col, direction = order.split(".")
        q = q.order(col, desc=(direction.lower() == "desc"))
    if limit:
        q = q.limit(limit)
    res = q.execute()
    if getattr(res, "data", None) is None:
        raise RuntimeError(f"Supabase select failed: {getattr(res, 'error', None)}")
    return res.data

def sb_insert(table: str, payload: Dict):
    sb = get_supabase()
    res = sb.table(table).insert(payload).execute()
    if getattr(res, "data", None) is None:
        raise RuntimeError(f"Supabase insert failed: {getattr(res, 'error', None)}")
    return res.data

def sb_update(table: str, match: Dict, payload: Dict):
    sb = get_supabase()
    q = sb.table(table).update(payload)
    for k, v in match.items():
        q = q.eq(k, v)
    res = q.execute()
    if getattr(res, "data", None) is None:
        raise RuntimeError(f"Supabase update failed: {getattr(res, 'error', None)}")
    return res.data

def sb_rpc(fn: str, params: Dict):
    sb = get_supabase()
    res = sb.rpc(fn, params).execute()
    # supabase-py returns .data even if empty; errors in .error
    if getattr(res, "error", None):
        raise RuntimeError(f"Supabase rpc failed: {res.error}")
    return getattr(res, "data", None)

def normalize_notes(notes_text: str, players_a: List[str], players_b: List[str]) -> str:
    # store as JSON so selection persists across counselors
    payload = {
        "notes_text": notes_text.strip(),
        "players_a": players_a,
        "players_b": players_b,
    }
    return json.dumps(payload)

def parse_notes(notes: Optional[str]) -> Tuple[str, List[str], List[str]]:
    if not notes:
        return "", [], []
    try:
        obj = json.loads(notes)
        if isinstance(obj, dict) and ("players_a" in obj or "players_b" in obj):
            return str(obj.get("notes_text", "")).strip(), list(obj.get("players_a", []) or []), list(obj.get("players_b", []) or [])
    except Exception:
        pass
    return str(notes), [], []

# Google Sheets
def _gs_client():
    if gspread is None or Credentials is None:
        raise RuntimeError("gspread/google-auth not installed.")
    sa = st.secrets.get("gcp_service_account")
    if not sa:
        raise RuntimeError("Missing [gcp_service_account] in secrets.")
    creds = Credentials.from_service_account_info(
        dict(sa),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    return gspread.authorize(creds)

def sheet_id() -> str:
    sid = secrets_get_first("sheet_id", "SPREADSHEET_ID", "spreadsheet_id", "SPREADSHEET")
    if not sid:
        raise RuntimeError("Missing Sheet secret: sheet_id or SPREADSHEET_ID")
    return sid

@st.cache_data(ttl=300)
def df_from_ws(tab_name: str) -> pd.DataFrame:
    gc = _gs_client()
    sh = gc.open_by_key(sheet_id())
    ws = sh.worksheet(tab_name)
    rows = ws.get_all_records()
    return pd.DataFrame(rows)

def roster_tabs_for_league(league_key: str) -> List[str]:
    # your sheet tabs are rosters_sophomore / rosters_junior / rosters_senior
    if league_key == "sophomore":
        return ["rosters_sophomore"]
    if league_key == "junior":
        return ["rosters_junior"]
    return ["rosters_senior"]

def roster_df(league_key: str) -> pd.DataFrame:
    tabs = roster_tabs_for_league(league_key)
    dfs = []
    for t in tabs:
        df = df_from_ws(t).copy()
        if df.empty:
            continue
        # enforce columns
        for c in ["player_id", "first_name", "last_name", "team_name", "bunk"]:
            if c not in df.columns:
                df[c] = ""
        df["full_name"] = (df["first_name"].astype(str).str.strip() + " " + df["last_name"].astype(str).str.strip()).str.strip()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["player_id","first_name","last_name","team_name","bunk","full_name"])

def team_roster_names(df: pd.DataFrame, team_name: str) -> List[str]:
    sub = df[df["team_name"].astype(str).str.strip() == str(team_name).strip()].copy()
    names = [n for n in sub["full_name"].astype(str).tolist() if n.strip()]
    # de-dupe while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def full_name_to_player_id(df: pd.DataFrame) -> Dict[str, str]:
    m = {}
    for _, r in df.iterrows():
        nm = str(r.get("full_name","")).strip()
        pid = str(r.get("player_id","")).strip()
        if nm and pid and nm not in m:
            m[nm] = pid
    return m

# -----------------------------
# Live game engine
# -----------------------------
def get_active_games(league_key: str) -> List[Dict]:
    games = sb_select("live_games", match={"league_key": league_key, "status": "active"}, order="created_at.desc", limit=25)
    return games

def create_live_game(payload: Dict) -> str:
    # Guard: league_key cannot be null (your error screenshot)
    if not payload.get("league_key"):
        raise ValueError("league_key is required")
    if not payload.get("id"):
        payload["id"] = str(uuid.uuid4())
    sb_insert("live_games", payload)
    return payload["id"]

def compute_remaining(game: Dict) -> int:
    # Supports both timer_remaining_seconds and timer_remaining_at_anchor
    dur = int(game.get("duration_seconds") or 0)
    running = bool(game.get("timer_running"))
    anchor_ts = game.get("timer_anchor_ts")
    remaining_anchor = game.get("timer_remaining_at_anchor", None)
    if remaining_anchor is None:
        remaining_anchor = game.get("timer_remaining_seconds", dur)
    remaining_anchor = int(remaining_anchor or 0)
    if not running or not anchor_ts:
        return max(0, remaining_anchor)
    elapsed = int(max(0, epoch_now() - float(anchor_ts)))
    return max(0, remaining_anchor - elapsed)

def timer_start(game_id: str, current_remaining: int):
    sb_update("live_games", {"id": game_id}, {
        "timer_running": True,
        "timer_anchor_ts": float(epoch_now()),
        "timer_remaining_at_anchor": int(current_remaining),
        "timer_remaining_seconds": int(current_remaining),
        "clock_style": "running",
        "updated_at": iso_utc(),
    })

def timer_pause(game_id: str, current_remaining: int):
    sb_update("live_games", {"id": game_id}, {
        "timer_running": False,
        "timer_anchor_ts": None,
        "timer_remaining_at_anchor": int(current_remaining),
        "timer_remaining_seconds": int(current_remaining),
        "clock_style": "nonrunning",
        "updated_at": iso_utc(),
    })

def timer_reset(game_id: str, duration_seconds: int):
    sb_update("live_games", {"id": game_id}, {
        "timer_running": False,
        "timer_anchor_ts": None,
        "timer_remaining_at_anchor": int(duration_seconds),
        "timer_remaining_seconds": int(duration_seconds),
        "clock_style": "nonrunning",
        "duration_seconds": int(duration_seconds),
        "updated_at": iso_utc(),
    })

def add_score(game: Dict, side: str, delta: int):
    # Prefer RPC if you add apply_live_event; else direct update
    game_id = game["id"]
    if side not in ("A","B"):
        return
    field = "score_a" if side == "A" else "score_b"
    new_val = int(game.get(field) or 0) + int(delta)
    sb_update("live_games", {"id": game_id}, {field: int(new_val), "updated_at": iso_utc()})
    # log event
    try:
        sb_insert("live_events", {
            "game_id": game_id,
            "created_at": iso_utc(),
            "event_type": "score",
            "side": side,
            "delta": int(delta),
            "player_id": None,
            "team_name": None,
            "stat_key": None,
        })
    except Exception:
        pass

def add_player_stat(game: Dict, player_id: str, team_name: str, stat_key: str, delta: int):
    sb_insert("live_events", {
        "game_id": game["id"],
        "created_at": iso_utc(),
        "event_type": "stat",
        "side": None,
        "delta": int(delta),
        "player_id": player_id,
        "team_name": team_name,
        "stat_key": stat_key,
    })

# -----------------------------
# UI
# -----------------------------
def sidebar_brand():
    st.sidebar.title("Bauercrest Live LM")
    try:
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    except Exception:
        pass

def league_picker() -> str:
    # Persist chosen league
    if "league_key" not in st.session_state:
        st.session_state["league_key"] = LEAGUE_KEYS[0]
    label_map = {k: v for k, v in LEAGUES}
    current = st.sidebar.selectbox("League", options=LEAGUE_KEYS, format_func=lambda k: label_map.get(k, k), key="league_key")
    return current

def page_setup(league_key: str):
    st.header("Setup")
    st.caption("Sanity check: roster + sheet connection (cached).")
    df = roster_df(league_key)
    if df.empty:
        st.error("Roster sheet is empty or not readable.")
        return
    teams = sorted([t for t in df["team_name"].astype(str).unique().tolist() if t.strip()])
    st.write(f"Teams found: **{len(teams)}** | Players found: **{len(df)}**")
    st.dataframe(df[["full_name","team_name","bunk"]].head(30), use_container_width=True)

def page_games_lobby(league_key: str):
    st.header("Games")
    st.caption("Create a game, or open an active game already running.")
    games = get_active_games(league_key)
    if games:
        st.subheader("Active games")
        for g in games[:10]:
            title = f"{g.get('sport','?')} {g.get('level','?')} — {g.get('team_a1','A')} vs {g.get('team_b1','B')} (Score {g.get('score_a',0)}-{g.get('score_b',0)})"
            cols = st.columns([4,1])
            cols[0].write(title)
            if cols[1].button("Open", key=f"open_{g['id']}"):
                st.session_state["active_game_id"] = g["id"]
                st.session_state["page"] = "Run Game"
                st.rerun()
    else:
        st.info("No active games yet.")

    st.divider()
    st.subheader("Create a new live game")

    df = roster_df(league_key)
    teams = sorted([t for t in df["team_name"].astype(str).unique().tolist() if t.strip()])
    if not teams:
        st.error("No teams in roster sheet.")
        return

    c1, c2, c3 = st.columns([2,1,1])
    sport = c1.selectbox("Sport", options=SPORTS, key="new_sport")
    level = c2.selectbox("Level", options=LEVELS, key="new_level")  # A/B/C/D (your requirement)
    mode = c3.selectbox("Mode", options=["1v1","2v2"], key="new_mode")

    duration_default = DEFAULT_DURATIONS.get(sport, 0)
    duration_seconds = st.number_input("Clock (seconds)", min_value=0, max_value=7200, value=int(duration_default), step=30)

    # Team pickers
    if mode == "1v1":
        tA = st.selectbox("Left team", teams, key="new_team_a")
        tB = st.selectbox("Right team", [t for t in teams if t != tA] or teams, key="new_team_b")
        team_a1, team_a2, team_b1, team_b2 = tA, None, tB, None
    else:
        col = st.columns(4)
        team_a1 = col[0].selectbox("A1", teams, key="new_a1")
        team_a2 = col[1].selectbox("A2", [t for t in teams if t != team_a1] or teams, key="new_a2")
        team_b1 = col[2].selectbox("B1", [t for t in teams if t not in {team_a1, team_a2}] or teams, key="new_b1")
        team_b2 = col[3].selectbox("B2", [t for t in teams if t not in {team_a1, team_a2, team_b1}] or teams, key="new_b2")

    # Preload players from roster into left/right areas
    left_team_label = team_a1
    right_team_label = team_b1
    left_default = team_roster_names(df, left_team_label)
    right_default = team_roster_names(df, right_team_label)

    st.write("### Who’s playing (edit roster for this game)")
    colL, colR = st.columns(2)
    with colL:
        players_a = st.multiselect(f"{left_team_label} players", options=left_default, default=left_default, key="players_a")
    with colR:
        players_b = st.multiselect(f"{right_team_label} players", options=right_default, default=right_default, key="players_b")

    notes_text = st.text_input("Notes (optional)", key="new_notes")

    if st.button("Create game & open", type="primary"):
        payload = {
            "id": str(uuid.uuid4()),
            "created_at": iso_utc(),
            "updated_at": iso_utc(),
            "league_key": league_key,
            "sport": sport,
            "level": level,
            "mode": mode,
            "team_a1": team_a1,
            "team_a2": team_a2,
            "team_b1": team_b1,
            "team_b2": team_b2,
            "score_a": 0,
            "score_b": 0,
            "duration_seconds": int(duration_seconds),
            "timer_running": False,
            "timer_anchor_ts": None,
            "timer_remaining_at_anchor": int(duration_seconds),
            "timer_remaining_seconds": int(duration_seconds),
            "clock_style": "nonrunning",
            "status": "active",
            "notes": normalize_notes(notes_text, players_a, players_b),
        }
        gid = create_live_game(payload)
        st.session_state["active_game_id"] = gid
        st.session_state["page"] = "Run Game"
        st.success("Game created.")
        st.rerun()

def page_run_game(league_key: str):
    st.header("Run Live Game")

    game_id = st.session_state.get("active_game_id")
    if not game_id:
        st.info("Open a game from the Games page.")
        return

    # Pull game fresh
    rows = sb_select("live_games", match={"id": game_id}, limit=1)
    if not rows:
        st.error("Game not found.")
        return
    game = rows[0]
    if game.get("league_key") != league_key:
        st.warning("This game belongs to a different league. Switching league view.")
        st.session_state["league_key"] = game.get("league_key", league_key)
        st.rerun()

    sport = game.get("sport","Basketball")
    level = game.get("level","A")
    left_team = game.get("team_a1","Left")
    right_team = game.get("team_b1","Right")

    notes_text, players_a_names, players_b_names = parse_notes(game.get("notes"))

    top = st.columns([3,2,2,2])
    top[0].subheader(f"{sport} — Level {level}")
    top[1].metric("Left", int(game.get("score_a") or 0))
    top[2].metric("Right", int(game.get("score_b") or 0))
    if top[3].button("End Game", type="secondary"):
        sb_update("live_games", {"id": game_id}, {"status":"ended","updated_at": iso_utc()})
        st.success("Game ended.")
        st.session_state["active_game_id"] = None
        st.session_state["page"] = "Games"
        st.rerun()

    # Timer
    st.divider()
    st.subheader("Clock")
    remaining = compute_remaining(game)
    mins = remaining // 60
    secs = remaining % 60
    st.write(f"**{mins:02d}:{secs:02d}**")
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    if c1.button("Start", disabled=bool(game.get("timer_running")) or int(game.get("duration_seconds") or 0) == 0):
        timer_start(game_id, remaining)
        st.rerun()
    if c2.button("Pause", disabled=not bool(game.get("timer_running"))):
        timer_pause(game_id, remaining)
        st.rerun()
    if c3.button("Reset"):
        timer_reset(game_id, int(game.get("duration_seconds") or 0))
        st.rerun()
    auto_refresh = c4.toggle("Live refresh (recommended)", value=True)
    if auto_refresh and bool(game.get("timer_running")):
        # gentle polling; prevents freezing, keeps clock moving
        time.sleep(0.8)
        st.rerun()

    # Score buttons
    st.divider()
    st.subheader("Scoreboard")
    colA, colB = st.columns(2)
    with colA:
        st.write(f"### {left_team}")
        s1, s2, s3 = st.columns(3)
        if s1.button("+1", key="scoreA1"): add_score(game, "A", 1); st.rerun()
        if s2.button("+2", key="scoreA2"): add_score(game, "A", 2); st.rerun()
        if s3.button("+3", key="scoreA3"): add_score(game, "A", 3); st.rerun()
    with colB:
        st.write(f"### {right_team}")
        s1, s2, s3 = st.columns(3)
        if s1.button("+1", key="scoreB1"): add_score(game, "B", 1); st.rerun()
        if s2.button("+2", key="scoreB2"): add_score(game, "B", 2); st.rerun()
        if s3.button("+3", key="scoreB3"): add_score(game, "B", 3); st.rerun()

    # Player stat pad
    st.divider()
    st.subheader("Player Stats (tap-fast)")

    df = roster_df(league_key)
    name_to_pid = full_name_to_player_id(df)

    # If notes didn't have roster selections (older games), derive from teams
    if not players_a_names:
        players_a_names = team_roster_names(df, left_team)
    if not players_b_names:
        players_b_names = team_roster_names(df, right_team)

    auto_team_score = st.toggle("Auto-add to team score when adding player points/runs", value=True)

    def stat_panel(team_label: str, side: str, names: List[str]):
        st.write(f"### {team_label}")
        if not names:
            st.info("No players set for this side.")
            return
        # choose active player to keep UI clean
        active = st.selectbox("Who are you recording a stat for?", options=names, key=f"active_{side}")
        pid = name_to_pid.get(active, "")
        if not pid:
            st.warning("Missing player_id for selected player in roster sheet.")
            return

        btns = STAT_BUTTONS.get(sport, STAT_BUTTONS["Basketball"])
        cols = st.columns(len(btns))
        for i, (k, label) in enumerate(btns):
            if cols[i].button(label, key=f"{side}_{pid}_{k}"):
                if sport == "Basketball" and k.startswith("PTS_"):
                    pts = int(k.split("_")[1])
                    if auto_team_score:
                        add_score(game, side, pts)
                    add_player_stat(game, pid, team_label, "PTS", pts)
                elif sport == "Softball" and k == "RUN":
                    if auto_team_score:
                        add_score(game, side, 1)
                    add_player_stat(game, pid, team_label, "RUN", 1)
                else:
                    add_player_stat(game, pid, team_label, k, 1)
                st.rerun()

    left_col, right_col = st.columns(2)
    with left_col:
        stat_panel(left_team, "A", players_a_names)
    with right_col:
        stat_panel(right_team, "B", players_b_names)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    sidebar_brand()
    league_key = league_picker()

    if "page" not in st.session_state:
        st.session_state["page"] = "Games"

    page = st.sidebar.radio("Go to", ["Setup","Games","Run Game"], index=["Setup","Games","Run Game"].index(st.session_state["page"]))
    st.session_state["page"] = page

    try:
        if page == "Setup":
            page_setup(league_key)
        elif page == "Games":
            page_games_lobby(league_key)
        else:
            page_run_game(league_key)
    except Exception as e:
        st.error(str(e))
        st.exception(e)

if __name__ == "__main__":
    main()
