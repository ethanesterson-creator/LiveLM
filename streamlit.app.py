# streamlit.app.py
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---- Optional deps (installed via requirements.txt) ----
# gspread + google-auth for Google Sheets (rosters, games, stats, non-game points, highlights)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:  # pragma: no cover
    gspread = None
    Credentials = None

# Supabase for live-game engine + concurrency
try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


# =========================
# Branding
# =========================
BAUERCREST_NAVY = "#ffffff"
BAUERCREST_GOLD = "#D4AF37"

APP_TITLE = "Bauercrest Crest League Manager (Live)"

# If you have a local logo file in your repo, keep this path
LOGO_PATH = "logo-header-2.png"  # adjust if needed


# =========================
# League configuration
# =========================
LEAGUES = [
    ("sophmore", "Sophmore League"),
    ("junior", "Junior League"),
    ("seniors", "Senior League"),
]

LEAGUE_KEYS = [k for k, _ in LEAGUES]
LEAGUE_LABELS = {k: v for k, v in LEAGUES}

# Sports list (extend any time)
SPORTS = [
    "Basketball",
    "Football",
    "Softball",
    "Kickball",
    "Hockey",
    "Soccer",
    "Euro",
    "Speedball",
    "Volleyball",
    "Dodgeball",
    "Tennis",
    "Wiffleball",
    "Handball",
]

LEVELS = ["A", "B", "C", "D"]

GAME_MODES = [
    ("1v1", "1 team vs 1 team"),
    ("2v2", "2 teams vs 2 teams (combined)"),
]

# Stat keys per sport for the live stat-buttons UI
SPORT_STATS: Dict[str, List[str]] = {
    "Basketball": ["PTS", "AST", "REB", "STL", "BLK"],
    "Football": ["TD", "REC", "INT", "SACK"],
    "Softball": ["H", "RBI", "R", "BB", "SO"],
    "Kickball": ["K", "R", "RBI", "OUT"],
    "Hockey": ["G", "A", "SOG"],
    "Soccer": ["G", "A", "SOG"],
    "Euro": ["G", "A"],
    "Speedball": ["G", "A"],
    "Volleyball": ["K", "A", "D"],
    "Dodgeball": ["OUT", "CATCH"],
    "Tennis": ["GAME_W"],
    "Wiffleball": ["H", "RBI", "R"],
    "Handball": ["G", "A"],
}

# Score increment presets per sport (buttons on scoreboard)
SCORE_BUTTONS: Dict[str, List[int]] = {
    "Basketball": [1, 2, 3, -1],
    "Football": [1, 2, 3, 6, -1],
    "Softball": [1, -1],
    "Kickball": [1, -1],
    "Hockey": [1, -1],
    "Soccer": [1, -1],
    "Euro": [1, -1],
    "Speedball": [1, -1],
    "Volleyball": [1, -1],
    "Dodgeball": [1, -1],
    "Tennis": [1, -1],
    "Wiffleball": [1, -1],
    "Handball": [1, -1],
}

# Timer preset options per sport (duration_seconds, label)
SPORT_TIMER_PRESETS: Dict[str, List[Tuple[int, str]]] = {
    "Basketball": [
        (15 * 60 * 2, "15 min halves (total 30:00)"),
        (20 * 60 * 2, "20 min halves (total 40:00)"),
        (12 * 60 * 4, "12 min quarters (total 48:00)"),
    ],
    "Hockey": [
        (15 * 60 * 3, "15 min periods (total 45:00)"),
        (12 * 60 * 3, "12 min periods (total 36:00)"),
    ],
    "Soccer": [
        (20 * 60 * 2, "20 min halves (total 40:00)"),
        (25 * 60 * 2, "25 min halves (total 50:00)"),
    ],
    "Football": [
        (12 * 60 * 2, "12 min halves (total 24:00)"),
        (15 * 60 * 2, "15 min halves (total 30:00)"),
    ],
    "Softball": [
        (60 * 60, "1 hour time-cap"),
        (45 * 60, "45 min time-cap"),
    ],
    "Kickball": [
        (45 * 60, "45 min time-cap"),
        (35 * 60, "35 min time-cap"),
    ],
    "Euro": [
        (15 * 60 * 2, "15 min halves (total 30:00)"),
        (20 * 60 * 2, "20 min halves (total 40:00)"),
    ],
    "Speedball": [
        (15 * 60 * 2, "15 min halves (total 30:00)"),
    ],
    "Volleyball": [
        (30 * 60, "30 min time-cap"),
    ],
    "Dodgeball": [
        (25 * 60, "25 min time-cap"),
    ],
    "Tennis": [
        (45 * 60, "45 min time-cap"),
    ],
    "Wiffleball": [
        (45 * 60, "45 min time-cap"),
    ],
    "Handball": [
        (15 * 60 * 2, "15 min halves (total 30:00)"),
    ],
}

# Points mapping (editable). These are "game win points" example values.
# Structure: POINTS[league_key][sport][level] = points_for_win
POINTS: Dict[str, Dict[str, Dict[str, int]]] = {lk: {} for lk in LEAGUE_KEYS}
for lk in LEAGUE_KEYS:
    for s in SPORTS:
        POINTS[lk][s] = {lvl: 10 for lvl in LEVELS}  # default

# Seed reasonable examples based on your guidance:
# Seniors: A Softball 50, B 45, C 40, D 35
POINTS["seniors"]["Softball"] = {"A": 50, "B": 45, "C": 40, "D": 35}
# Seniors: A Football 40, B 35, C 30, D 25
POINTS["seniors"]["Football"] = {"A": 40, "B": 35, "C": 30, "D": 25}
# Juniors: scale down ~10
POINTS["junior"]["Softball"] = {"A": 40, "B": 35, "C": 30, "D": 25}
POINTS["junior"]["Football"] = {"A": 32, "B": 28, "C": 24, "D": 20}
# Sophmores: A Softball ~35 down by 5
POINTS["sophmore"]["Softball"] = {"A": 35, "B": 30, "C": 25, "D": 20}
POINTS["sophmore"]["Football"] = {"A": 28, "B": 24, "C": 20, "D": 16}


# =========================
# Helpers: time + formatting
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_notes(notes: str) -> dict:
    try:
        obj = json.loads(notes) if notes else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def build_notes(notes: str, lineup_player_ids: List[str]) -> str:
    obj = parse_notes(notes)
    obj["lineup_player_ids"] = lineup_player_ids
    return json.dumps(obj)


def fmt_clock(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .bc-title {{
                font-size: 2.0rem;
                font-weight: 800;
                color: {BAUERCREST_NAVY};
                margin-bottom: 0.1rem;
            }}
            .bc-subtitle {{
                color: #5b6470;
                margin-top: 0;
            }}
            .scoreboard {{
                border: 2px solid {BAUERCREST_NAVY};
                border-radius: 14px;
                padding: 14px;
                background: #0f172a;
                color: white;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            }}
            .scoreline {{
                display:flex;
                align-items:center;
                justify-content:space-between;
                gap: 12px;
            }}
            .teamname {{
                font-size: 1.1rem;
                font-weight: 700;
                width: 46%;
                overflow:hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .score {{
                font-size: 2.2rem;
                font-weight: 900;
                width: 10%;
                text-align:center;
            }}
            .clock {{
                text-align:center;
                font-size: 3.2rem;
                font-weight: 900;
                letter-spacing: 1px;
                margin: 10px 0;
            }}
            .pill {{
                display:inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(255,255,255,0.1);
                font-size: 0.85rem;
            }}
            .tiny {{
                font-size: 0.85rem;
                opacity: 0.85;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Google Sheets (non-live data)
# =========================
@st.cache_resource
def get_gspread_client():
    """Google Sheets client (optional)."""
    if gspread is None or Credentials is None:
        return None

    # Support either a JSON string or a dict in secrets
    svc = None
    if "gcp_service_account" in st.secrets:
        svc = st.secrets["gcp_service_account"]
    elif "gcp_service_account_json" in st.secrets:
        svc = st.secrets["gcp_service_account_json"]

    if svc is None:
        return None

    try:
        info = json.loads(svc) if isinstance(svc, str) else dict(svc)
    except Exception:
        st.error("Google service account secret is not valid JSON.")
        return None

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def sheets_ready() -> bool:
    return get_gspread_client() is not None and "sheet_id" in st.secrets


def canonical_ws_name(name: str) -> str:
    # Central place to map “code names” to the actual tab names in your Sheet.
    mapping = {
        "Highlights": "Highlights",
        "highlights": "Highlights",
        "NonGamePoints": "NonGamePoints",
        "nongamepoints": "NonGamePoints",
        "rosters_sophomore": "rosters_sophomore",
        "rosters_junior": "rosters_junior",
        "rosters_senior": "rosters_senior",
        "games": "games",
        "stats": "stats",
    }
    return mapping.get(name, name)


def ws(name: str):
    gc = get_gspread_client()
    if gc is None:
        raise RuntimeError("Google Sheets not configured.")
    sh = gc.open_by_key(st.secrets["sheet_id"])
    cname = canonical_ws_name(name)
    try:
        return sh.worksheet(cname)
    except Exception:
        # try a couple of common fallbacks (case differences)
        for alt in {cname.title(), cname.upper(), cname.lower()}:
            try:
                return sh.worksheet(alt)
            except Exception:
                pass
        raise



@st.cache_data(ttl=300, show_spinner=False)
def df_from_ws(name: str) -> pd.DataFrame:
    """Read a worksheet into a DataFrame.

    Cached to avoid hammering the Google Sheets API (prevents 429 quota errors).
    TTL=5 minutes is plenty for rosters/standings during league.
    """
    w = ws(name)
    values = w.get_all_values()
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)


def overwrite_ws(name: str, df: pd.DataFrame) -> None:
    w = ws(name)
    w.clear()
    w.update([df.columns.tolist()] + df.astype(str).fillna("").values.tolist())
    try:
        df_from_ws.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def append_ws(name: str, row: List) -> None:
    w = ws(name)
    w.append_row([str(x) if x is not None else "" for x in row])
    try:
        df_from_ws.clear()  # type: ignore[attr-defined]
    except Exception:
        pass



# =========================
# Supabase (live game engine)
# =========================
@st.cache_resource
def get_supabase():
    if create_client is None:
        return None
    url = st.secrets.get("supabase_url")
    key = st.secrets.get("supabase_anon_key")
    if not url or not key:
        return None
    return create_client(url, key)


def supabase_ready() -> bool:
    return get_supabase() is not None



def sb_upsert_live_game(game_id: str, payload: dict) -> None:
    """Upsert a live_games row. This must EXECUTE (older versions of this file were incomplete)."""
    sb = get_supabase()
    if sb is None:
        raise RuntimeError("Supabase not configured.")
    payload = dict(payload or {})
    payload["id"] = game_id
    sb.table("live_games").upsert(payload).execute()


def sb_update_live_game(game_id: str, payload: dict):
    """Update ONLY. If the row doesn't exist, Supabase will return empty; we treat that as an error."""
    sb = get_supabase()
    if sb is None:
        raise RuntimeError("Supabase not configured.")
    payload = dict(payload or {})
    resp = sb.table("live_games").update(payload).eq("id", game_id).execute()
    try:
        data = resp.data
    except Exception:
        data = None
    if not data:
        raise RuntimeError("Update failed (game not found).")
    return data[0]

def sb_create_live_game(payload: dict) -> str:
    sb = get_supabase()
    if sb is None:
        raise RuntimeError("Supabase not configured.")
    game_id = str(uuid.uuid4())
    payload = dict(payload)
    payload["id"] = game_id
    sb.table("live_games").insert(payload).execute()
    return game_id


def sb_get_live_game(game_id: str) -> Optional[dict]:
    sb = get_supabase()
    if sb is None:
        return None
    r = sb.table("live_games").select("*").eq("id", game_id).limit(1).execute()
    data = getattr(r, "data", None) or []
    return data[0] if data else None


def sb_list_live_games(status: str = "active", limit: int = 50) -> List[dict]:
    sb = get_supabase()
    if sb is None:
        return []
    r = (
        sb.table("live_games")
        .select("*")
        .eq("status", status)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return getattr(r, "data", None) or []


def sb_insert_event(payload: dict) -> None:
    sb = get_supabase()
    if sb is None:
        raise RuntimeError("Supabase not configured.")
    sb.table("live_events").insert(payload).execute()


def sb_list_events(game_id: str, limit: int = 500) -> List[dict]:
    sb = get_supabase()
    if sb is None:
        return []
    r = (
        sb.table("live_events")
        .select("*")
        .eq("game_id", game_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return getattr(r, "data", None) or []


# =========================
# Rosters
# =========================
def normalize_league_key(k: str) -> str:
    k = (k or "").strip().lower()
    # accept common misspellings / labels
    if k in {"sophmore", "sophomore", "soph"}:
        return "sophomore"
    if k in {"jr", "junior"}:
        return "junior"
    if k in {"sr", "senior", "seniors"}:
        return "senior"
    return k


def roster_sheet_name(league_key: str) -> str:
    lk = normalize_league_key(league_key)
    if lk == "sophomore":
        return "rosters_sophomore"
    if lk == "junior":
        return "rosters_junior"
    return "rosters_senior"


def roster_df(league_key: str) -> pd.DataFrame:
    if not sheets_ready():
        return pd.DataFrame()
    return df_from_ws(roster_sheet_name(league_key))


def teams_in_roster(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    if "team_name" not in df.columns:
        return []
    return sorted([t for t in df["team_name"].dropna().unique().tolist() if str(t).strip() != ""])


def players_for_team(df: pd.DataFrame, team: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["team_name"] == team].copy()


def player_label(row: pd.Series) -> str:
    fn = str(row.get("first_name", "")).strip()
    ln = str(row.get("last_name", "")).strip()
    pid = str(row.get("player_id", "")).strip()
    name = (fn + " " + ln).strip()
    return f"{name} ({pid})" if pid else name


# =========================
# Live game timer model (server state)
# =========================
def compute_remaining(game: dict) -> int:
    """Compute remaining seconds based on game.timer_anchor_ts and timer_running."""
    dur = int(game.get("duration_seconds") or 0)
    remaining = int(game.get("timer_remaining_seconds") or dur)
    running = bool(game.get("timer_running"))
    anchor = game.get("timer_anchor_ts")
    if not running:
        return max(0, remaining)

    # anchor_ts stored as ISO string or epoch
    anchor_epoch = None
    if isinstance(anchor, (int, float)):
        anchor_epoch = float(anchor)
    elif isinstance(anchor, str) and anchor:
        try:
            anchor_epoch = datetime.fromisoformat(anchor.replace("Z", "+00:00")).timestamp()
        except Exception:
            anchor_epoch = None

    if anchor_epoch is None:
        return max(0, remaining)

    elapsed = int(time.time() - anchor_epoch)
    return max(0, remaining - elapsed)


def set_timer_running(game_id: str, running: bool) -> None:
    game = sb_get_live_game(game_id)
    if not game:
        return
    cur_remaining = compute_remaining(game)
    payload = {
        "updated_at": now_utc().isoformat(),
        "timer_running": bool(running),
        "timer_remaining_seconds": int(cur_remaining),
        "timer_anchor_ts": time.time() if running else None,
    }
    sb_update_live_game(game_id, payload)


def reset_timer(game_id: str) -> None:
    game = sb_get_live_game(game_id)
    if not game:
        return
    dur = int(game.get("duration_seconds") or 0)
    payload = {
        "updated_at": now_utc().isoformat(),
        "timer_running": False,
        "timer_remaining_seconds": int(dur),
        "timer_anchor_ts": None,
    }
    sb_update_live_game(game_id, payload)


def update_score(game_id: str, side: str, delta: int) -> None:
    game = sb_get_live_game(game_id)
    if not game:
        return
    a = int(game.get("score_a") or 0)
    b = int(game.get("score_b") or 0)
    if side == "A":
        a = max(0, a + delta)
    else:
        b = max(0, b + delta)
    sb_upsert_live_game(
        game_id,
        {
            "updated_at": now_utc().isoformat(),
            "score_a": a,
            "score_b": b,
        },
    )
    sb_insert_event(
        {
            "game_id": game_id,
            "created_at": now_utc().isoformat(),
            "event_type": "score",
            "side": side,
            "delta": int(delta),
        }
    )


def add_stat(game_id: str, player_id: str, team_name: str, stat_key: str, delta: int = 1) -> None:
    sb_insert_event(
        {
            "game_id": game_id,
            "created_at": now_utc().isoformat(),
            "event_type": "stat",
            "player_id": player_id,
            "team_name": team_name,
            "stat_key": stat_key,
            "delta": int(delta),
        }
    )


def aggregated_stats(events: List[dict]) -> pd.DataFrame:
    """Aggregate stat events to a pivot table for quick view."""
    stat_rows = [e for e in events if e.get("event_type") == "stat"]
    if not stat_rows:
        return pd.DataFrame()
    df = pd.DataFrame(stat_rows)
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce").fillna(0).astype(int)
    piv = (
        df.pivot_table(
            index=["team_name", "player_id"],
            columns="stat_key",
            values="delta",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    # flatten columns
    piv.columns = [str(c) for c in piv.columns]
    return piv


# =========================
# UI: scoreboard display (client-side ticking)
# =========================
def scoreboard_widget(game: dict) -> None:
    """A JS clock ticks every second without hammering the server."""
    team_a = game.get("team_a1") or "Team A"
    team_b = game.get("team_b1") or "Team B"
    score_a = int(game.get("score_a") or 0)
    score_b = int(game.get("score_b") or 0)

    dur = int(game.get("duration_seconds") or 0)
    remaining = int(game.get("timer_remaining_seconds") or dur)
    running = bool(game.get("timer_running"))
    anchor = game.get("timer_anchor_ts")  # epoch float in our updates

    # Use epoch anchor for client; if missing, treat as paused.
    anchor_epoch = None
    if isinstance(anchor, (int, float)):
        anchor_epoch = float(anchor)

    html = f"""
    <div class="scoreboard">
      <div class="scoreline">
        <div class="teamname">{team_a}</div>
        <div class="score" id="scoreA">{score_a}</div>
        <div class="pill">{game.get("sport","")}</div>
        <div class="score" id="scoreB">{score_b}</div>
        <div class="teamname" style="text-align:right">{team_b}</div>
      </div>

      <div class="clock" id="clock">--:--</div>

      <div class="scoreline tiny">
        <div class="pill">League: {LEAGUE_LABELS.get(game.get("league_key",""), game.get("league_key",""))}</div>
        <div class="pill">Level: {game.get("level","")}</div>
        <div class="pill">Mode: {game.get("mode","1v1")}</div>
        <div class="pill">Status: {"RUNNING" if running else "PAUSED"}</div>
      </div>
    </div>

    <script>
      const running = {str(running).lower()};
      const initialRemaining = {remaining};
      const anchor = {anchor_epoch if anchor_epoch is not None else 'null'};

      function fmt(sec) {{
        if (sec < 0) sec = 0;
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return String(m).padStart(2,'0') + ":" + String(s).padStart(2,'0');
      }}

      function tick() {{
        let rem = initialRemaining;
        if (running && anchor !== null) {{
          const now = Date.now() / 1000.0;
          const elapsed = Math.floor(now - anchor);
          rem = initialRemaining - elapsed;
        }}
        document.getElementById("clock").innerText = fmt(rem);
      }}

      tick();
      setInterval(tick, 1000);
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Pages
# =========================
def page_setup(current_league: str) -> None:
    st.header("Setup")
    st.caption("Upload rosters and confirm your sheets are connected.")

    if not sheets_ready():
        st.warning(
            "Google Sheets is not configured. Add `sheet_id` and `gcp_service_account` to Streamlit secrets."
        )
        return

    lk = current_league
    sheet_name = roster_sheet_name(lk)

    st.subheader(f"Roster: {LEAGUE_LABELS[lk]}")
    df = df_from_ws(sheet_name)

    with st.expander("Upload roster CSV (replaces entire roster)", expanded=False):
        st.write("CSV **must** contain columns: `player_id, first_name, last_name, team_name, bunk`")
        up = st.file_uploader("Upload roster CSV", type=["csv"], key=f"roster_upload_{lk}")
        if up is not None:
            new_df = pd.read_csv(up)
            required = {"player_id", "first_name", "last_name", "team_name", "bunk"}
            missing = required - set(new_df.columns)

def page_live_games_home():
    st.header("Live Games (Create / Open)")
    if not current_league:
        st.warning("Pick a league first (sidebar).")
        return

    roster = roster_df(current_league)
    if roster.empty:
        st.error("Roster sheet is empty / missing.")
        return

    # Normalize roster columns
    roster = roster.copy()
    roster["player_id"] = roster["player_id"].astype(str).str.strip()
    roster["team_name"] = roster["team_name"].astype(str).str.strip()

    sport = st.selectbox("Sport", options=["Basketball", "Softball"], index=0)
    level = st.selectbox("Level", options=["sophomores", "juniors", "seniors"], index=0)
    mode = st.selectbox("Mode", options=["1v1", "2v2"], index=0)

    teams = sorted([t for t in roster["team_name"].unique().tolist() if t])

    c1, c2 = st.columns(2)
    with c1:
        team_a1 = st.selectbox("Left team A1", options=teams, index=0 if teams else None)
        team_a2 = st.selectbox("Left team A2 (only for 2v2)", options=[""] + teams, index=0)
    with c2:
        team_b1 = st.selectbox("Right team B1", options=teams, index=0 if teams else None)
        team_b2 = st.selectbox("Right team B2 (only for 2v2)", options=[""] + teams, index=0)

    if mode == "1v1":
        team_a2 = ""
        team_b2 = ""

    # Lineup selection
    st.caption("Pick who is playing (optional but recommended). This makes the in-game stat buttons super fast.")
    player_opts = []
    pid_by_label = {}
    for r in roster.to_dict("records"):
        pid = str(r.get("player_id","")).strip()
        if not pid:
            continue
        name = f"{str(r.get('first_name','')).strip()} {str(r.get('last_name','')).strip()}".strip() or pid
        team = str(r.get("team_name","")).strip()
        lab = f"{name} ({team})" if team else name
        if lab in pid_by_label:
            lab = f"{lab} [{pid[:6]}]"
        pid_by_label[lab] = pid
        player_opts.append(lab)

    lineup_labels = st.multiselect("Players playing", options=player_opts, default=[])

    notes = st.text_input("Notes (optional)", value="")

    if st.button("Create Live Game", type="primary"):
        lineup_ids = [pid_by_label.get(l) for l in lineup_labels]
        lineup_ids = [p for p in lineup_ids if p]
        notes_json = build_notes(notes, lineup_ids) if lineup_ids else notes

        payload = {
            "league_key": current_league,
            "sport": sport,
            "level": level,
            "mode": mode,
            "team_a1": team_a1,
            "team_a2": team_a2 or None,
            "team_b1": team_b1,
            "team_b2": team_b2 or None,
            "score_a": 0,
            "score_b": 0,
            "duration_seconds": 1800 if sport.lower() == "basketball" else 3600,
            "timer_running": False,
            "timer_anchor_ts": None,
            "timer_remaining_seconds": 1800 if sport.lower() == "basketball" else 3600,
            "clock_style": "nonnunning",
            "status": "active",
            "notes": notes_json or None,
            "created_at": now_utc().isoformat(),
            "updated_at": now_utc().isoformat(),
        }
        gid = sb_create_live_game(payload)
        st.success("Game created.")
        st.session_state["active_game_id"] = gid
        st.rerun()

    st.markdown("---")
    st.subheader("Open an existing live game")
    games = sb_list_live_games(current_league)
    if not games:
        st.info("No active games yet.")
        return

    options = []
    id_by_label = {}
    for g in games:
        lab = f"{g['id'][:8]} — {g.get('sport','')} {g.get('level','')} {g.get('mode','')} | {g.get('team_a1','')} vs {g.get('team_b1','')} | {g.get('status','')}"
        options.append(lab)
        id_by_label[lab] = g["id"]

    chosen = st.selectbox("Select a game", options=options)
    if st.button("Open game"):
        st.session_state["active_game_id"] = id_by_label[chosen]
        st.rerun()


def page_run_live_game(current_league: str) -> None:
    st.header("Run Live Game")
    if not supabase_ready():
        st.error("Supabase not configured.")
        return

    gid = st.session_state.get("active_game_id")
    if not gid:
        st.info("Go to **Live Games (Create/Open)** and create or open a game.")
        return

    game = sb_get_live_game(gid)
    if not game:
        st.error("That game no longer exists.")
        return

    if game.get("status") != "active":
        st.warning("This game is not active.")
    inject_css()
    scoreboard_widget(game)

# ---- Controls row
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("▶ Start", use_container_width=True):
            set_timer_running(gid, True)
            st.rerun()
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            set_timer_running(gid, False)
            st.rerun()
    with c3:
        if st.button("↺ Reset", use_container_width=True):
            reset_timer(gid)
            st.rerun()
    with c4:
        st.caption("Tip: the big clock ticks every second on its own (client-side), no refresh spam.")

    st.markdown("---")
    # ---- Score buttons
    st.subheader("Score")
    sport = game.get("sport", "Basketball")
    buttons = SCORE_BUTTONS.get(sport, [1, -1])

    left, right = st.columns(2)
    with left:
        st.write(f"**{game.get('team_a1','Team A')}**")
        cols = st.columns(len(buttons))
        for i, delta in enumerate(buttons):
            label = f"{delta:+d}"
            if cols[i].button(label, key=f"scoreA_{delta}"):
                update_score(gid, "A", delta)
                st.toast(f"Score A {delta:+d}", icon="✅")
                st.rerun()

    with right:
        st.write(f"**{game.get('team_b1','Team B')}**")
        cols = st.columns(len(buttons))
        for i, delta in enumerate(buttons):
            label = f"{delta:+d}"
            if cols[i].button(label, key=f"scoreB_{delta}"):
                update_score(gid, "B", delta)
                st.toast(f"Score B {delta:+d}", icon="✅")
                st.rerun()

    st.markdown("---")
    # ---- Stats buttons
    st.subheader("Stats (tap to add)")
    roster = roster_df(game.get("league_key", current_league))
    if roster.empty:
        st.info("No roster loaded for this league yet.")
        return

    # Active lineup selection (so you only see the players in THIS game)
    teams = teams_in_roster(roster)
    mode = game.get("mode", "1v1")
    if mode == "1v1":
        teams_in_game = [game.get("team_a1"), game.get("team_b1")]
    else:
        teams_in_game = [game.get("team_a1"), game.get("team_a2"), game.get("team_b1"), game.get("team_b2")]
    teams_in_game = [t for t in teams_in_game if t]

    st.caption("Active lineup: pick who is actually playing (so you don't see the entire league).")
    # --- Lineup (who is playing) ---
    notes_obj = parse_notes(str(game.get("notes","")))
    saved_lineup = notes_obj.get("lineup_player_ids") or []
    if isinstance(saved_lineup, str):
        saved_lineup = [p.strip() for p in saved_lineup.split(",") if p.strip()]

    # Build quick lookup for roster
    roster = roster.copy()
    roster["player_id"] = roster["player_id"].astype(str).str.strip()
    roster = roster[roster["player_id"] != ""]
    roster_by_id = {r["player_id"]: r for r in roster.to_dict("records")}

    # Default lineup = saved lineup if present, else empty (counselor can set it)
    lineup_ids = [pid for pid in saved_lineup if pid in roster_by_id]

    with st.expander("Who’s playing? (set once, then you’re good)", expanded=(len(lineup_ids) == 0)):
        opts = []
        label_to_pid = {}
        for r in roster.to_dict("records"):
            pid = r["player_id"]
            name = f"{str(r.get('first_name','')).strip()} {str(r.get('last_name','')).strip()}".strip()
            team = str(r.get("team_name","")).strip()
            lab = f"{name} ({team})" if team else name
            if lab in label_to_pid:
                lab = f"{lab} [{pid[:6]}]"
            label_to_pid[lab] = pid
            opts.append(lab)

        default_labels = []
        for pid in lineup_ids:
            r = roster_by_id.get(pid, {})
            name = f"{str(r.get('first_name','')).strip()} {str(r.get('last_name','')).strip()}".strip()
            team = str(r.get("team_name","")).strip()
            lab = f"{name} ({team})" if team else name
            # handle possible de-dupe labels
            if lab not in label_to_pid:
                for k, v in label_to_pid.items():
                    if v == pid:
                        lab = k
                        break
            if lab in label_to_pid:
                default_labels.append(lab)

        picked_labels = st.multiselect("Players playing", options=opts, default=default_labels, key=f"lineup_labels_{gid}")
        if st.button("Save lineup", key=f"save_lineup_{gid}"):
            picked_ids = [label_to_pid.get(l) for l in picked_labels]
            picked_ids = [p for p in picked_ids if p]
            notes_json = build_notes(str(game.get("notes","")), picked_ids)
            sb_update_live_game(gid, {"notes": notes_json, "updated_at": now_utc().isoformat()})
            st.success("Lineup saved.")
            st.rerun()

    # Re-load after possible save
    notes_obj = parse_notes(str(game.get("notes","")))
    saved_lineup = notes_obj.get("lineup_player_ids") or []
    if isinstance(saved_lineup, str):
        saved_lineup = [p.strip() for p in saved_lineup.split(",") if p.strip()]
    lineup_ids = [pid for pid in saved_lineup if pid in roster_by_id]

    if not lineup_ids:
        st.info("Set the lineup above to unlock the fast tap stat buttons.")
    else:
        st.markdown("### Quick stats (tap fast)")

        auto_score = st.toggle("Auto-add to team score when you tap a player PTS button", value=True, key=f"auto_score_{gid}")

        # Helper for rendering player rows
        def _render_player_row(pid: str, side: str):
            r = roster_by_id.get(pid, {})
            name = f"{str(r.get('first_name','')).strip()} {str(r.get('last_name','')).strip()}".strip()
            if not name:
                name = pid
            cols = st.columns([3,1,1,1,1,1,1,1,1], gap="small") if sport.lower()=="basketball" else st.columns([3,1,1,1], gap="small")

            cols[0].markdown(f"**{name}**")
            if sport.lower() == "basketball":
                if cols[1].button("+1", key=f"{gid}_{pid}_p1"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "pts", 1)
                    if auto_score: update_score(gid, side, 1, current_league, sport, level)
                    st.rerun()
                if cols[2].button("+2", key=f"{gid}_{pid}_p2"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "pts", 2)
                    if auto_score: update_score(gid, side, 2, current_league, sport, level)
                    st.rerun()
                if cols[3].button("+3", key=f"{gid}_{pid}_p3"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "pts", 3)
                    if auto_score: update_score(gid, side, 3, current_league, sport, level)
                    st.rerun()
                if cols[4].button("+Ast", key=f"{gid}_{pid}_ast"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "ast", 1)
                    st.rerun()
                if cols[5].button("+Reb", key=f"{gid}_{pid}_reb"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "reb", 1)
                    st.rerun()
                if cols[6].button("+Stl", key=f"{gid}_{pid}_stl"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "stl", 1)
                    st.rerun()
                if cols[7].button("+Blk", key=f"{gid}_{pid}_blk"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "blk", 1)
                    st.rerun()
                if cols[8].button("+TO", key=f"{gid}_{pid}_to"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "to", 1)
                    st.rerun()
            else:
                # Softball (simple)
                if cols[1].button("+Hit", key=f"{gid}_{pid}_hit"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "hit", 1)
                    st.rerun()
                if cols[2].button("+Run", key=f"{gid}_{pid}_run"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "run", 1)
                    if auto_score: update_score(gid, side, 1, current_league, sport, level)
                    st.rerun()
                if cols[3].button("+RBI", key=f"{gid}_{pid}_rbi"):
                    add_stat(gid, current_league, sport, level, pid, str(r.get('team_name','')), "rbi", 1)
                    st.rerun()

        # Split lineup by which side they belong to (match team names in game config)
        team_a_names = [t for t in [team_a1, team_a2] if t]
        team_b_names = [t for t in [team_b1, team_b2] if t]

        a_ids = []
        b_ids = []
        for pid in lineup_ids:
            tname = str(roster_by_id.get(pid, {}).get("team_name","")).strip()
            if tname in team_a_names:
                a_ids.append(pid)
            elif tname in team_b_names:
                b_ids.append(pid)
            else:
                # If team doesn't match, default to A so player isn't lost
                a_ids.append(pid)

        cA, cB = st.columns(2, gap="large")
        with cA:
            st.markdown(f"#### Left (A): {' / '.join(team_a_names) if team_a_names else 'Team A'}")
            s1, s2, s3 = st.columns(3, gap="small")
            if s1.button("+1 Team", key=f"{gid}_A_plus1"): update_score(gid, "A", 1, current_league, sport, level); st.rerun()
            if s2.button("+2 Team", key=f"{gid}_A_plus2"): update_score(gid, "A", 2, current_league, sport, level); st.rerun()
            if s3.button("+3 Team", key=f"{gid}_A_plus3"): update_score(gid, "A", 3, current_league, sport, level); st.rerun()
            st.markdown("---")
            for pid in a_ids:
                _render_player_row(pid, "A")

        with cB:
            st.markdown(f"#### Right (B): {' / '.join(team_b_names) if team_b_names else 'Team B'}")
            s1, s2, s3 = st.columns(3, gap="small")
            if s1.button("+1 Team", key=f"{gid}_B_plus1"): update_score(gid, "B", 1, current_league, sport, level); st.rerun()
            if s2.button("+2 Team", key=f"{gid}_B_plus2"): update_score(gid, "B", 2, current_league, sport, level); st.rerun()
            if s3.button("+3 Team", key=f"{gid}_B_plus3"): update_score(gid, "B", 3, current_league, sport, level); st.rerun()
            st.markdown("---")
            for pid in b_ids:
                _render_player_row(pid, "B")

    st.markdown("---")
    st.subheader("This game: live stat totals")
    events = sb_list_events(gid)
    piv = aggregated_stats(events)
    if piv.empty:
        st.caption("No stats yet.")
    else:
        st.dataframe(piv, use_container_width=True)

    st.markdown("---")
    st.subheader("Event log (for this game)")
    if events:
        df_e = pd.DataFrame(events)
        show_cols = [c for c in ["created_at", "event_type", "side", "delta", "player_id", "team_name", "stat_key"] if c in df_e.columns]
        st.dataframe(df_e[show_cols].tail(200), use_container_width=True)
    else:
        st.caption("No events yet.")

    st.markdown("---")
    st.subheader("Finish / Save game")
    st.caption("This will mark the live game as finished. (Hook up post-game write to Sheets here if desired.)")

    if st.button("✅ Finish Game", type="primary"):
        # Pause and mark finished
        set_timer_running(gid, False)
        sb_upsert_live_game(
            gid,
            {
                "updated_at": now_utc().isoformat(),
                "status": "finished",
                "timer_running": False,
                "timer_anchor_ts": None,
                "timer_remaining_seconds": compute_remaining(sb_get_live_game(gid) or game),
            },
        )
        st.success("Game finished.")
        st.session_state.pop("active_game_id", None)
        st.rerun()


def page_standings(current_league: str) -> None:
    st.header("Standings")
    if not sheets_ready():
        st.warning("Sheets not configured.")
        return
    # This assumes you already have a sheet tab called "games" with at least:
    # league_key, sport, level, team_a, team_b, winner_team
    df = df_from_ws("games")
    if df.empty:
        st.info("No games yet.")
        return

    # Filter to selected league
    df = df[df.get("league_key", "") == current_league].copy()
    if df.empty:
        st.info("No games yet for this league.")
        return

    # Compute points
    # Expect columns: team_a, team_b, winner_team, sport, level
    for col in ["team_a", "team_b", "winner_team", "sport", "level"]:
        if col not in df.columns:
            st.error(f"Sheet 'games' is missing column: {col}")
            return

    teams = sorted(set(df["team_a"].tolist() + df["team_b"].tolist()))
    pts = {t: 0 for t in teams}
    for _, r in df.iterrows():
        sport = str(r["sport"])
        lvl = str(r["level"])
        winner = str(r["winner_team"])
        p = POINTS.get(current_league, {}).get(sport, {}).get(lvl, 0)
        if winner in pts:
            pts[winner] += int(p)

    out = pd.DataFrame({"team": list(pts.keys()), "points": list(pts.values())}).sort_values("points", ascending=False)
    st.dataframe(out, use_container_width=True)


def page_highlights() -> None:
    st.header("Highlights")
    st.caption("Phase: store video uploads to Supabase Storage (recommended).")
    st.info("We can wire this to Supabase Storage next. For now, this page is a placeholder.")
    st.file_uploader("Upload highlight video file", type=["mp4", "mov", "m4v"])
    st.write("When we wire storage: upload -> store public URL -> display playlist on Display Board.")


def clear_ws_keep_header(tab: str) -> None:
    w = ws(tab)
    header = w.row_values(1)
    w.clear()
    if header:
        w.update("A1", [header])


def supabase_delete_all(table: str) -> None:
    sb = get_supabase()
    if sb is None:
        return
    # PostgREST doesn't support TRUNCATE; delete with a wide filter.
    # Assumes created_at exists; if not, fall back to deleting everything via neq on a constant.
    try:
        sb.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    except Exception:
        sb.table(table).delete().neq("league_key", "__never__").execute()


def page_admin() -> None:
    st.header("Admin")
    st.caption("Password-protected tools to manage Crest League data (Google Sheets + Supabase live engine).")

    # --- Password gate ---
    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False

    if not st.session_state.admin_ok:
        pw = st.text_input("Admin password", type="password")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Unlock"):
                if pw == "Hyaffa26":
                    st.session_state.admin_ok = True
                    st.success("Admin unlocked.")
                else:
                    st.error("Incorrect password.")
        st.stop()

    st.success("Admin unlocked.")

    st.subheader("Clear data")
    st.warning("These actions permanently delete data. Use carefully.")

    clear_targets = st.multiselect(
        "Select what to clear",
        [
            "Supabase: live_games + live_events (Live Engine)",
            "Sheet: games",
            "Sheet: stats",
            "Sheet: NonGamePoints",
            "Sheet: Highlights",
            "Sheet: rosters (all leagues)",
        ],
        default=[],
    )

    confirm = st.text_input("Type CLEAR to enable delete buttons", value="")

    colA, colB = st.columns(2)

    with colA:
        if st.button("Clear selected", disabled=(confirm.strip() != "CLEAR" or not clear_targets)):
            try:
                if "Supabase: live_games + live_events (Live Engine)" in clear_targets:
                    # delete events first due to FK
                    supabase_delete_all("live_events")
                    supabase_delete_all("live_games")

                if "Sheet: games" in clear_targets:
                    clear_ws_keep_header("games")
                if "Sheet: stats" in clear_targets:
                    clear_ws_keep_header("stats")
                if "Sheet: NonGamePoints" in clear_targets:
                    clear_ws_keep_header("NonGamePoints")
                if "Sheet: Highlights" in clear_targets:
                    clear_ws_keep_header("Highlights")

                if "Sheet: rosters (all leagues)" in clear_targets:
                    clear_ws_keep_header("rosters_sophomore")
                    clear_ws_keep_header("rosters_junior")
                    clear_ws_keep_header("rosters_senior")

                st.success("Done.")
            except Exception as e:
                st.error(f"Admin action failed: {e}")

    with colB:
        if st.button("Lock admin"):
            st.session_state.admin_ok = False
            st.info("Admin locked.")

    st.divider()
    st.subheader("Local app session")
    if st.button("Clear local session (does NOT delete Sheets/Supabase)"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k, None)
        st.success("Local session cleared.")

def standings_table(df_games: pd.DataFrame, league_key: Optional[str] = None) -> pd.DataFrame:
    if df_games.empty:
        return pd.DataFrame(columns=["team", "points"])

    df = df_games.copy()

    if "league_key" in df.columns:
        df["league_key_norm"] = df["league_key"].apply(normalize_league_key)
        if league_key:
            df = df[df["league_key_norm"] == normalize_league_key(league_key)]
    else:
        df["league_key_norm"] = ""

    required = {"team_a1", "team_a2", "team_b1", "team_b2", "points_a", "points_b", "mode"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["team", "points"])

    df["points_a"] = pd.to_numeric(df["points_a"], errors="coerce").fillna(0).astype(int)
    df["points_b"] = pd.to_numeric(df["points_b"], errors="coerce").fillna(0).astype(int)

    totals: Dict[str, int] = {}

    def add(team: str, pts: int):
        team = str(team or "").strip()
        if not team:
            return
        totals[team] = totals.get(team, 0) + int(pts)

    for _, r in df.iterrows():
        mode = str(r.get("mode", "1v1")).strip()
        a1, a2 = str(r.get("team_a1", "")).strip(), str(r.get("team_a2", "")).strip()
        b1, b2 = str(r.get("team_b1", "")).strip(), str(r.get("team_b2", "")).strip()
        pa, pb = int(r["points_a"]), int(r["points_b"])

        if mode == "2v2":
            a_teams = [t for t in [a1, a2] if t]
            b_teams = [t for t in [b1, b2] if t]
            if a_teams:
                share = pa / max(1, len(a_teams))
                for t in a_teams:
                    add(t, share)
            if b_teams:
                share = pb / max(1, len(b_teams))
                for t in b_teams:
                    add(t, share)
        else:
            add(a1, pa)
            add(b1, pb)

    out = pd.DataFrame({"team": list(totals.keys()), "points": list(totals.values())})
    out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0)
    out = out.sort_values("points", ascending=False).reset_index(drop=True)
    return out


def stat_leaders(df_stats: pd.DataFrame, roster_all: pd.DataFrame, league_key: Optional[str], stat_key: str, top_n: int = 10) -> pd.DataFrame:
    if df_stats.empty or "stat_key" not in df_stats.columns:
        return pd.DataFrame(columns=["player", "team", "total"])

    df = df_stats.copy()
    if "league_key" in df.columns:
        df["league_key_norm"] = df["league_key"].apply(normalize_league_key)
        if league_key:
            df = df[df["league_key_norm"] == normalize_league_key(league_key)]

    df = df[df["stat_key"] == stat_key].copy()
    if df.empty:
        return pd.DataFrame(columns=["player", "team", "total"])

    df["value"] = pd.to_numeric(df.get("value"), errors="coerce").fillna(0)

    agg = df.groupby(["player_id", "team_name"], dropna=False)["value"].sum().reset_index()
    agg = agg.rename(columns={"value": "total"})

    # merge in names from roster sheets
    if not roster_all.empty and "player_id" in roster_all.columns:
        roster_all = roster_all.copy()
        roster_all["player_id"] = roster_all["player_id"].astype(str)
        roster_all["player"] = (roster_all.get("first_name", "").astype(str).str.strip() + " " + roster_all.get("last_name", "").astype(str).str.strip()).str.strip()
        lookup = roster_all[["player_id", "player"]].drop_duplicates()
        agg["player_id"] = agg["player_id"].astype(str)
        agg = agg.merge(lookup, on="player_id", how="left")
    else:
        agg["player"] = agg["player_id"].astype(str)

    agg["player"] = agg["player"].fillna(agg["player_id"].astype(str))
    agg["team_name"] = agg["team_name"].fillna("")
    agg = agg.sort_values("total", ascending=False).head(top_n).reset_index(drop=True)
    return agg[["player", "team_name", "total"]].rename(columns={"team_name": "team"})


def page_display_board() -> None:
    st.header("Display Board")
    st.caption("Big-screen mode for standings, stat leaders, and highlights.")

    if not sheets_ready():
        st.warning("Google Sheets not configured.")
        return

    mode = st.selectbox(
        "Board mode",
        [
            "Standings (one league)",
            "Standings (3 leagues + overall camp)",
            "Stat Leaders",
            "Highlights",
        ],
    )

    games_df = df_from_ws("games")
    stats_df = df_from_ws("stats")
    highlights_df = df_from_ws("Highlights")

    roster_all = pd.DataFrame()
    try:
        roster_all = pd.concat(
            [df_from_ws("rosters_sophomore"), df_from_ws("rosters_junior"), df_from_ws("rosters_senior")],
            ignore_index=True,
        )
    except Exception:
        roster_all = pd.DataFrame()

    if mode == "Standings (one league)":
        league = st.selectbox("League", ["Sophomore", "Junior", "Senior"])
        lk = league.lower()
        table = standings_table(games_df, lk)
        st.subheader(f"{league} Standings")
        st.dataframe(table, use_container_width=True, hide_index=True)

    elif mode == "Standings (3 leagues + overall camp)":
        st.subheader("League Standings")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Sophomore")
            st.dataframe(standings_table(games_df, "sophomore"), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### Junior")
            st.dataframe(standings_table(games_df, "junior"), use_container_width=True, hide_index=True)
        with c3:
            st.markdown("### Senior")
            st.dataframe(standings_table(games_df, "senior"), use_container_width=True, hide_index=True)

        st.markdown("### Entire Camp Standings (All Leagues Combined)")
        st.dataframe(standings_table(games_df, None), use_container_width=True, hide_index=True)

    elif mode == "Stat Leaders":
        if stats_df.empty:
            st.info("No stats yet.")
            return
        if "stat_key" not in stats_df.columns:
            st.error("Stats sheet missing 'stat_key' column.")
            return

        league = st.selectbox("League", ["Sophomore", "Junior", "Senior", "All"])
        lk = None if league == "All" else league.lower()

        stat_keys = sorted([s for s in stats_df["stat_key"].dropna().unique().tolist() if str(s).strip() != ""])
        stat_key = st.selectbox("Stat", stat_keys) if stat_keys else None
        top_n = st.slider("Top N", 5, 25, 10)

        if not stat_key:
            st.info("No stat keys found.")
            return

        out = stat_leaders(stats_df, roster_all, lk, stat_key, top_n=top_n)
        st.subheader(f"{stat_key} Leaders — {league}")
        st.dataframe(out, use_container_width=True, hide_index=True)

    else:  # Highlights
        st.subheader("Highlights")
        if highlights_df.empty:
            st.info("No highlights posted yet.")
            return
        # sort newest first if possible
        if "uploaded_at" in highlights_df.columns:
            highlights_df = highlights_df.sort_values("uploaded_at", ascending=False)
        show = highlights_df.head(25)
        st.dataframe(show, use_container_width=True, hide_index=True)
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()

    # Sidebar
    st.sidebar.markdown(f"<div class='bc-title'>League Manager</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='bc-subtitle'>Crest League • Live engine (Supabase)</div>", unsafe_allow_html=True)
    try:
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    except Exception:
        pass

    # League selector (affects Setup + standings + roster-dependent pages)
    current_league = st.sidebar.selectbox(
        "League (for Setup/Stats)",
        LEAGUE_KEYS,
        format_func=lambda k: LEAGUE_LABELS[k],
        index=2,
    )

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Go to",
        [
            "Setup",
            "Live Games (Create/Open)",
            "Run Live Game",
            "Standings",
            "Highlights",
            "Display Board",
            "Admin / Clear Data",
        ],
    )

    # Quick health checks
    with st.sidebar.expander("Connections", expanded=False):
        st.write("Sheets:", "✅" if sheets_ready() else "⚠️ not configured")
        st.write("Supabase:", "✅" if supabase_ready() else "⚠️ not configured")
        if supabase_ready():
            st.caption("Supabase is used for live games so multiple games can run at once.")
        if sheets_ready():
            st.caption("Google Sheets can still store rosters / post-game results.")

    # Routes
    if page == "Setup":
        page_setup(current_league)
    elif page == "Live Games (Create/Open)":
        page_live_games_home(current_league)
    elif page == "Run Live Game":
        page_run_live_game(current_league)
    elif page == "Standings":
        page_standings(current_league)
    elif page == "Highlights":
        page_highlights()
    elif page == "Display Board":
        page_display_board()
    else:
        page_admin()


if __name__ == "__main__":
    main()
