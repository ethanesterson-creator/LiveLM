
# ============================================================
# Bauercrest League Manager – Production Streamlit App
# Live games via Supabase, historical data via Google Sheets
# ============================================================

import time
import json
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

# -----------------------------
# Optional imports (safe)
# -----------------------------
try:
    from supabase import create_client
except Exception:
    create_client = None

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Bauercrest League Manager",
    layout="wide",
)

# ============================================================
# Secrets helpers
# ============================================================

def sheets_ready() -> bool:
    return (
        gspread is not None
        and "sheet_id" in st.secrets
        and "gcp_service_account" in st.secrets
    )


def supabase_ready() -> bool:
    return (
        create_client is not None
        and st.secrets.get("supabase_url")
        and st.secrets.get("supabase_anon_key")
    )


@st.cache_resource
def get_gspread_client():
    if not sheets_ready():
        return None
    creds = Credentials.from_service_account_info(
        json.loads(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)


@st.cache_resource
def get_supabase():
    if not supabase_ready():
        return None
    return create_client(
        st.secrets["supabase_url"],
        st.secrets["supabase_anon_key"],
    )


# ============================================================
# Google Sheets helpers (HISTORICAL DATA)
# ============================================================

def ws(name: str):
    gc = get_gspread_client()
    if gc is None:
        raise RuntimeError("Google Sheets not configured")
    sh = gc.open_by_key(st.secrets["sheet_id"])
    return sh.worksheet(name)


def df_from_ws(name: str) -> pd.DataFrame:
    w = ws(name)
    values = w.get_all_values()
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)


def append_ws(name: str, row: list):
    w = ws(name)
    w.append_row([str(x) if x is not None else "" for x in row])


# ============================================================
# Supabase helpers (LIVE ENGINE ONLY)
# ============================================================

def sb():
    return get_supabase()


def sb_select(table, **kwargs):
    return sb().table(table).select("*").match(kwargs).execute()


def sb_insert(table, data: dict):
    return sb().table(table).insert(data).execute()


def sb_update(table, match: dict, data: dict):
    return sb().table(table).update(data).match(match).execute()


# ============================================================
# UI – Setup / Status
# ============================================================

def page_setup():
    st.header("Setup")
    st.write("Connection status for this app")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Google Sheets")
        if sheets_ready():
            st.success("Connected")
        else:
            st.error("Not configured")

    with col2:
        st.subheader("Supabase (Live Games)")
        if supabase_ready():
            st.success("Connected")
        else:
            st.error("Not configured")


# ============================================================
# Live Games – Create / Open
# ============================================================

def page_live_games():
    st.header("Live Games (Create / Open)")

    if not supabase_ready():
        st.error("Supabase not configured")
        return

    sbc = sb()

    games = sbc.table("live_games").select("*").execute().data or []

    with st.expander("Create New Game"):
        with st.form("create_game"):
            league_key = st.text_input("League key", "seniors")
            sport = st.text_input("Sport", "Basketball")
            level = st.text_input("Level", "A")
            mode = st.selectbox("Mode", ["1v1", "2v2"])
            team_a = st.text_input("Team A")
            team_b = st.text_input("Team B")
            duration = st.number_input("Game length (seconds)", 300, 3600, 1800)

            submitted = st.form_submit_button("Create Game")
            if submitted:
                sb_insert("live_games", {
                    "league_key": league_key,
                    "sport": sport,
                    "level": level,
                    "mode": mode,
                    "team_a1": team_a,
                    "team_b1": team_b,
                    "score_a": 0,
                    "score_b": 0,
                    "duration_seconds": duration,
                    "timer_running": False,
                    "timer_anchor": None,
                    "time_remaining_s": duration,
                    "status": "active",
                })
                st.success("Game created")
                st.rerun()

    st.subheader("Active Games")
    if not games:
        st.info("No active games")
        return

    for g in games:
        st.write(
            f"**{g['team_a1']} vs {g['team_b1']}** — "
            f"{g['sport']} ({g['league_key']})"
        )


# ============================================================
# Run Live Game (NO 1s rerun – real scoreboard behavior)
# ============================================================

def render_scoreboard_js(seconds):
    minutes = seconds // 60
    secs = seconds % 60
    return f"""
    <script>
    let remaining = {seconds};
    const el = document.getElementById("clock");
    function tick() {{
        if (remaining <= 0) return;
        remaining--;
        let m = Math.floor(remaining / 60);
        let s = remaining % 60;
        el.innerText = m + ":" + String(s).padStart(2, "0");
        setTimeout(tick, 1000);
    }}
    setTimeout(tick, 1000);
    </script>
    """


def page_run_live_game():
    st.header("Run Live Game")

    if not supabase_ready():
        st.error("Supabase not configured")
        return

    sbc = sb()
    games = sbc.table("live_games").select("*").execute().data or []

    if not games:
        st.info("No active games")
        return

    game = st.selectbox(
        "Select game",
        games,
        format_func=lambda g: f"{g['team_a1']} vs {g['team_b1']}",
    )

    remaining = game["time_remaining_s"]

    st.markdown(
        f"<h1 id='clock'>{remaining//60}:{remaining%60:02d}</h1>",
        unsafe_allow_html=True,
    )
    st.components.v1.html(render_scoreboard_js(remaining), height=0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start / Resume"):
            sb_update(
                "live_games",
                {"id": game["id"]},
                {
                    "timer_running": True,
                    "timer_anchor": datetime.utcnow().isoformat(),
                },
            )
            st.success("Timer started")

    with col2:
        if st.button("Pause"):
            sb_update(
                "live_games",
                {"id": game["id"]},
                {
                    "timer_running": False,
                },
            )
            st.success("Timer paused")


# ============================================================
# Standings (Google Sheets)
# ============================================================

def page_standings():
    st.header("Standings")
    if not sheets_ready():
        st.error("Google Sheets not configured")
        return
    df = df_from_ws("games")
    st.dataframe(df, use_container_width=True)


# ============================================================
# Leaderboards (Google Sheets)
# ============================================================

def page_leaderboards():
    st.header("Leaderboards")
    if not sheets_ready():
        st.error("Google Sheets not configured")
        return
    df = df_from_ws("stats")
    st.dataframe(df, use_container_width=True)


# ============================================================
# Highlights (Google Sheets logging)
# ============================================================

def page_highlights():
    st.header("Highlights")
    if not sheets_ready():
        st.error("Google Sheets not configured")
        return

    file = st.file_uploader("Upload highlight video")
    if file:
        append_ws("highlights", [
            datetime.utcnow().isoformat(),
            file.name,
        ])
        st.success("Highlight logged")


# ============================================================
# Main router
# ============================================================

def main():
    st.sidebar.title("Bauercrest League Manager")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Setup",
            "Live Games",
            "Run Live Game",
            "Standings",
            "Leaderboards",
            "Highlights",
        ],
    )

    if page == "Setup":
        page_setup()
    elif page == "Live Games":
        page_live_games()
    elif page == "Run Live Game":
        page_run_live_game()
    elif page == "Standings":
        page_standings()
    elif page == "Leaderboards":
        page_leaderboards()
    elif page == "Highlights":
        page_highlights()


if __name__ == "__main__":
    main()
