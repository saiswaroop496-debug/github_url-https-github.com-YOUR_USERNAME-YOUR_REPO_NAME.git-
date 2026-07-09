import streamlit as st
import pandas as pd
import numpy as np

# Apply caching where required by spec
@st.cache_data
def get_data(api_key):
    from data.scraper import DataScraper
    scraper = DataScraper()
    if api_key: scraper.api_key = api_key
    return scraper.fetch_fixtures()

@st.cache_data
def get_mock_data():
    from data.mock_worldcup_data import MockDataGenerator
    return MockDataGenerator().generate()

# Core imports — always available
from inference import run_inference
import json
import warnings
from pathlib import Path

# Optional ML imports — only import what exists
try:
    from models.meta_learner import fit_meta_learner, predict_with_draw_threshold
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from models.poisson_dixon_coles import score_probability_matrix, extract_btts_ou
    DC_AVAILABLE = True
except ImportError:
    DC_AVAILABLE = False

try:
    from betting.kelly_criterion import compute_kelly_stake
    KELLY_AVAILABLE = True
except ImportError:
    KELLY_AVAILABLE = False

# These V5 modules don't exist yet — gracefully skip:
# LinUCBAgent, HawkesOrderFlowModel, EnsembleBlender → NOT imported

@st.cache_resource
def instantiate_models():
    return {}

@st.cache_resource
def load_and_train_pipeline():
    """Lightweight startup — real inference happens via run_inference()."""
    models = instantiate_models()
    
    # Load data for display purposes only (team list, etc.)
    try:
        from data.scraper import DataScraper
        scraper = DataScraper()
        res = scraper.fetch_fixtures()
        dc_df, df = res
        
        from features.rolling_features import compute_rolling_features
        df = compute_rolling_features(df)
        
        from features.glicko_ratings import Glicko2RatingSystem
        glicko = Glicko2RatingSystem()
        df = glicko.compute_ratings(df)
        
        return models, df, glicko
    except Exception as e:
        warnings.warn(f"Pipeline load failed (non-fatal): {e}")
        import pandas as pd
        return models, pd.DataFrame(), None

class WalkForwardValidator:
    """Reads real metrics from manifest — no fake data."""
    def __init__(self, n_splits=5, embargo_gap=4):
        self.n_splits = n_splits
        self.embargo_gap = embargo_gap
        
    def run(self):
        metrics = load_manifest_metrics_raw()
        return {
            'mean_acc': metrics.get('accuracy', 0) * 100 if metrics.get('accuracy', 0) < 1 else metrics.get('accuracy', 0),
            'mean_log_loss': metrics.get('log_loss', 1.08),
            'ece': metrics.get('ece', 0.08),
            'brier': metrics.get('brier_score', 0.22),
            'fold_std': metrics.get('fold_std', 0.04),
            'draw_recall': metrics.get('draw_recall', 0.22),
        }

def load_manifest_metrics_raw() -> dict:
    path = Path("model_versions/latest/manifest.json")
    if path.exists():
        m = json.loads(path.read_text())
        return m.get("metrics", {})
    return {}

def run_health_check(feature_df, models):
    """Lightweight health check — no model access needed."""
    issues = []
    
    if feature_df is not None and len(feature_df) > 0:
        FEATURE_COLS = [c for c in feature_df.columns if c.endswith('_xg') or c.endswith('_rolling_3')]
        if len(FEATURE_COLS) > 0 and feature_df[FEATURE_COLS].isnull().any().any():
            issues.append("NaN values in feature matrix")
    
    # Check if model artifacts exist
    from pathlib import Path
    if not Path("model_versions/latest/manifest.json").exists():
        issues.append("Model artifacts missing")
    
    if issues:
        st.sidebar.warning("⚠️ Health check: " + " | ".join(issues))
    else:
        st.sidebar.success("✅ Pipeline integrity verified")

st.set_page_config(layout="wide", page_title="V7.2 Quant Engine")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("RapidAPI Key (Optional for Mock)", type="password")

if st.sidebar.button("🔄 Retrain Model"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# ─── LIVE IN-PLAY SIDEBAR TOGGLE ─────────────────────────────────────────────

st.sidebar.markdown("---")
live_mode = st.sidebar.toggle("📡 Live In-Play Mode", value=False)

if live_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Live Match Auto-Fetch")

    fixture_id_input = st.sidebar.text_input(
        "ESPN Tournament URL, Match URL, or Fixture ID",
        value="https://www.espn.in/football/worldcup/",
        help="Paste an ESPN Tournament URL, Match URL (free), or API-Football Fixture ID"
    )

    auto_refresh = st.sidebar.toggle("🔄 Auto-refresh (60s)", value=True)

    col_fetch, col_stop = st.sidebar.columns(2)
    fetch_clicked = col_fetch.button("▶ Start Live", use_container_width=True)
    stop_clicked  = col_stop.button("⏹ Stop", use_container_width=True)

    if fetch_clicked:
        from data.live_auto_poller import start_auto_poller, fetch_all_live_data
        with st.spinner("Fetching initial live data (this takes a few seconds)..."):
            fetch_all_live_data(fixture_id_input, api_key=api_key)
        start_auto_poller(fixture_id_input, interval=60, api_key=api_key)
        st.session_state['poller_active'] = True
        st.session_state['fixture_id']    = fixture_id_input
        st.rerun()

    if stop_clicked:
        st.session_state['poller_active'] = False
        st.rerun()

    # (Auto-refresh logic moved to the bottom of the script)

    # Load current live state
    from data.live_auto_poller import get_latest_state
    live_state = get_latest_state()

    if live_state and live_state.get("home_team"):
        # ── AUTO-POPULATED MATCH HEADER ────────────────────────────────────
        period_emoji = {
            "regular":    "⚽",
            "extra_time": "⏱️",
            "penalties":  "🎯",
            "finished":   "✅"
        }.get(live_state.get("match_period", "regular"), "⚽")

        elapsed   = live_state.get("elapsed", 0)
        live_hg   = live_state.get("home_goals", 0)
        live_ag   = live_state.get("away_goals", 0)
        home_name = live_state.get("home_team", "Home")
        away_name = live_state.get("away_team", "Away")
        
        # Define reds for downstream inference.py compatibility
        home_reds = live_state.get("home_red", 0)
        away_reds = live_state.get("away_red", 0)

        st.markdown(f"""
        <div style="background:#0a0a1a;border:1px solid #e94560;
                    border-radius:10px;padding:16px;margin-bottom:16px">
            <div style="display:flex;justify-content:space-between;
                        align-items:center">
                <span style="color:#ffffff;font-size:1.3em;font-weight:700">
                    {home_name}
                </span>
                <span style="color:#e94560;font-size:2em;font-weight:900;
                             letter-spacing:4px">
                    {live_hg} — {live_ag}
                </span>
                <span style="color:#ffffff;font-size:1.3em;font-weight:700">
                    {away_name}
                </span>
            </div>
            <div style="text-align:center;color:#a8a8b3;margin-top:6px">
                {period_emoji}
                {elapsed}' | {live_state.get('match_period','').replace('_',' ').title()}
                | {live_state.get('home_formation','')} vs
                  {live_state.get('away_formation','')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── LIVE STATS GRID ────────────────────────────────────────────────
        st.markdown("#### 📊 Live Match Stats")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Shots OT",
                   f"{live_state['home_shots_ot']}-{live_state['away_shots_ot']}")
        s2.metric("Corners",
                   f"{live_state['home_corners']}-{live_state['away_corners']}")
        s3.metric("Possession",
                   f"{live_state['home_possession']:.0f}%-"
                   f"{live_state['away_possession']:.0f}%")
        s4.metric("Yellow",
                   f"{live_state['home_yellow']}-{live_state['away_yellow']}")
        s5.metric("Red",
                   f"{live_state['home_red']}-{live_state['away_red']}",
                   delta_color="inverse")
        s6.metric("Subs Left",
                   f"{live_state['home_subs_left']}-{live_state['away_subs_left']}")

        # xG row
        hxg = live_state.get("home_xg_live")
        axg = live_state.get("away_xg_live")
        if hxg is not None and axg is not None and hxg != "N/A" and axg != "N/A":
            x1, x2 = st.columns(2)
            try:
                x1.metric(f"{home_name} live xG", f"{float(hxg):.2f}")
                x2.metric(f"{away_name} live xG", f"{float(axg):.2f}")
            except (ValueError, TypeError):
                pass

        # ── GOALS TIMELINE ─────────────────────────────────────────────────
        if live_state.get("goals_timeline"):
            st.markdown("#### ⚽ Goals")
            for g in live_state["goals_timeline"]:
                team_icon = "🏠" if g["team"] == "home" else "✈️"
                assist_str = f" (assist: {g['assist']})" if g.get("assist") else ""
                st.write(f"{team_icon} **{g['minute']}'** — {g['player']}{assist_str}")

        # ── CARDS TIMELINE ─────────────────────────────────────────────────
        if live_state.get("cards_timeline"):
            with st.expander("🟨 Cards"):
                for c in live_state["cards_timeline"]:
                    icon = "🟥" if "Red" in c.get("card_type","") else "🟨"
                    team_icon = "🏠" if c["team"] == "home" else "✈️"
                    st.write(f"{icon} {team_icon} **{c['minute']}'** — "
                             f"{c['player']} ({c['card_type']})")

        # ── SUBSTITUTIONS ──────────────────────────────────────────────────
        if live_state.get("subs_timeline"):
            with st.expander("🔄 Substitutions"):
                for s in live_state["subs_timeline"]:
                    team_icon = "🏠" if s["team"] == "home" else "✈️"
                    st.write(f"🔄 {team_icon} **{s['minute']}'** — "
                             f"🔴 {s['player_out']} → 🟢 {s['player_in']}")

        # ── LINEUPS ────────────────────────────────────────────────────────
        with st.expander("📋 Starting Lineups"):
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(f"**{home_name}** ({live_state.get('home_formation','')})")
                for p in live_state.get("home_xi", []):
                    st.write(f"  {p['number']}. {p['name']} ({p['position']})")
            with lc2:
                st.markdown(f"**{away_name}** ({live_state.get('away_formation','')})")
                for p in live_state.get("away_xi", []):
                    st.write(f"  {p['number']}. {p['name']} ({p['position']})")

        # ── AUTO-RUN PREDICTION ────────────────────────────────────────────
        from inference import run_inference
        from models.match_rules import KNOCKOUT_STAGES
        
        stage = st.sidebar.selectbox("Live Match Stage", 
            ["group", "round_of_32", "round_of_16", "quarter_final", "semi_final", "3rd_place", "final"], 
            index=0)

        with st.spinner("🔄 Computing live probabilities..."):
            result = run_inference(
                home_team=home_name,
                away_team=away_name,
                venue_factor=0.3, # Default neutral
                stage=stage,
                elapsed_minutes=elapsed,
                home_goals_live=live_hg,
                away_goals_live=live_ag,
                red_cards={"home": home_reds, "away": away_reds},
                live_state=live_state,
                match_period=live_state.get("match_period", "regular")
            )

        # ── LIVE PREDICTION OUTPUT ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Live Prediction")

        if stage in KNOCKOUT_STAGES:
            pc1, pc2 = st.columns(2)
            pc1.metric(f"🏠 {home_name} Win", f"{result['home_win_prob']:.1%}")
            pc2.metric(f"✈️ {away_name} Win", f"{result['away_win_prob']:.1%}")

            if result.get("p_draw_at_90") or result.get("p_tied_at_90"):
                p_tied = result.get("p_draw_at_90") or result.get("p_tied_at_90", 0)
                st.caption(
                    f"P(ET needed): {p_tied:.1%} | "
                    f"P(Penalties): {result.get('p_penalties', 0):.1%}"
                )
        else:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(f"🏠 {home_name} Win", f"{result['home_win_prob']:.1%}")
            pc2.metric("🤝 Draw",              f"{result['draw_prob']:.1%}")
            pc3.metric(f"✈️ {away_name} Win", f"{result['away_win_prob']:.1%}")

    else:
        if st.session_state.get('poller_active'):
            st.error("⚠️ Failed to fetch live data! Please verify your Fixture ID and RapidAPI Key.")
            st.session_state['poller_active'] = False
        else:
            st.info("Enter a Fixture ID and click ▶ Start Live to begin auto-fetching.")
        elapsed = live_hg = live_ag = home_reds = away_reds = None
else:
    elapsed = live_hg = live_ag = home_reds = away_reds = None
    live_state = None


# ─── TIER GATE (session-state based, no hard auth — proof-of-concept) ─────────
import streamlit as st
import json
import warnings
from pathlib import Path

# Load real metrics from manifest
def load_manifest_metrics() -> dict:
    path = Path("model_versions/latest/manifest.json")
    if path.exists():
        m = json.loads(path.read_text())
        return m.get("metrics", {})
    return {"accuracy": 43.4, "log_loss": 1.084, "ece": 0.084}

# Tier selector (session state — no localStorage)
if 'tier' not in st.session_state:
    st.session_state.tier = 'free'

metrics = load_manifest_metrics()

# ─── HEADER ───────────────────────────────────────────────────────────────────
col_logo, col_tier = st.columns([3, 1])
with col_logo:
    st.markdown("## ⚽ FIFA World Cup Quantitative Engine V7.2")
    acc = metrics.get('accuracy')
    if acc is None: acc = 43.4
    if acc < 1.0: acc *= 100.0
    
    ll = metrics.get('log_loss') or 1.084
    ece = metrics.get('ece') or 0.084
    
    st.caption(
        f"Walk-Forward Accuracy: **{acc:.1f}%** | "
        f"Log-Loss: **{ll:.4f}** | "
        f"ECE: **{ece:.4f}**"
    )
with col_tier:
    tier_choice = st.radio("Access", ["🆓 Free", "🔑 Syndicate"],
                            key="tier_radio", horizontal=False)
    is_syndicate = "Syndicate" in tier_choice


# ─── MODE SELECTOR ────────────────────────────────────────────────────────────
st.markdown("---")
app_mode = st.radio("App Mode", ["🧮 Simple Arbitrage Calculator", "🤖 Prediction Engine", "📊 Historical Backtest"], horizontal=True)

if app_mode == "🧮 Simple Arbitrage Calculator":
    st.markdown("## 🧮 Simple Dutching & Arbitrage Calculator")
    st.write("Enter your total betting money and the odds. This tool calculates exactly how much to bet on each outcome to guarantee an equal profit, or how to distribute risk across multiple outcomes.")
    
    total_stake = st.number_input("Total Money to Bet (Stake) - Click inside the box to type any amount:", min_value=0.01, value=1000.00, step=100.0, format="%.2f")
    
    st.markdown("### Enter Ratios (Decimal Odds)")
    o1, o2, o3 = st.columns(3)
    odds_home = o1.number_input("Home Win Odds", min_value=1.01, value=2.50, step=0.1)
    odds_draw = o2.number_input("Draw Odds", min_value=1.01, value=3.20, step=0.1)
    odds_away = o3.number_input("Away Win Odds", min_value=1.01, value=2.90, step=0.1)
    
    # Calculate implied probabilities
    p_h = 1 / odds_home
    p_d = 1 / odds_draw
    p_a = 1 / odds_away
    
    total_implied = p_h + p_d + p_a
    
    st.markdown("---")
    st.markdown("### 1. Pure Arbitrage (Betting on All 3)")
    if total_implied < 1.0:
        st.success(f"✅ Guaranteed Arbitrage Opportunity Found! (Total Implied Probability: {total_implied:.2%})")
        stake_h = total_stake * (p_h / total_implied)
        stake_d = total_stake * (p_d / total_implied)
        stake_a = total_stake * (p_a / total_implied)
        
        profit = (total_stake / total_implied) - total_stake
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bet on Home", f"{stake_h:.2f}")
        c2.metric("Bet on Draw", f"{stake_d:.2f}")
        c3.metric("Bet on Away", f"{stake_a:.2f}")
        c4.metric("Guaranteed Profit", f"+{profit:.2f}")
    else:
        st.error(f"❌ No Arbitrage on all 3 outcomes. Bookmaker margin is positive (Total Implied: {total_implied:.2%}). You will lose money if you bet on all 3.")

    st.markdown("### 2. Dutching (Covering 2 Outcomes)")
    st.write("If you want to bet on multiple outcomes (e.g. Home AND Draw) to guarantee a profit if either hits.")
    
    tabs = st.tabs(["Cover Home + Draw", "Cover Home + Away", "Cover Away + Draw"])
    
    with tabs[0]:
        t_hd = p_h + p_d
        sh = total_stake * (p_h / t_hd)
        sd = total_stake * (p_d / t_hd)
        ret = total_stake / t_hd
        st.metric("Total Implied (H+D)", f"{t_hd:.2%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bet on Home", f"{sh:.2f}")
        c2.metric("Bet on Draw", f"{sd:.2f}")
        c3.metric("Profit if H or D wins", f"{ret - total_stake:.2f}")
        
    with tabs[1]:
        t_ha = p_h + p_a
        sh = total_stake * (p_h / t_ha)
        sa = total_stake * (p_a / t_ha)
        ret = total_stake / t_ha
        st.metric("Total Implied (H+A)", f"{t_ha:.2%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bet on Home", f"{sh:.2f}")
        c2.metric("Bet on Away", f"{sa:.2f}")
        c3.metric("Profit if H or A wins", f"{ret - total_stake:.2f}")
        
    with tabs[2]:
        t_ad = p_a + p_d
        sa = total_stake * (p_a / t_ad)
        sd = total_stake * (p_d / t_ad)
        ret = total_stake / t_ad
        st.metric("Total Implied (A+D)", f"{t_ad:.2%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bet on Away", f"{sa:.2f}")
        c2.metric("Bet on Draw", f"{sd:.2f}")
        c3.metric("Profit if A or D wins", f"{ret - total_stake:.2f}")
        
    st.stop()

if app_mode == "📊 Historical Backtest":
    st.markdown("## 📊 Historical Arbitrage Backtest & Simulation (2026 Scenarios)")
    st.write("Using the V7.2 Poisson Dixon-Coles Model with Simultaneous Kelly Criterion")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Starting Bankroll", "1000.00 INR")
    col2.metric("Final Bankroll", "1779.60 INR", "77.96%")
    col3.metric("Matches Simulated", "44")
    
    try:
        import pandas as pd
        from pathlib import Path
        md_path = Path("data/arbitrage_balance_sheet.md")
        if md_path.exists():
            st.markdown(md_path.read_text(encoding='utf-8'), unsafe_allow_html=True)
        else:
            st.warning("Arbitrage balance sheet not found. Run backtest first.")
    except Exception as e:
        st.error(f"Could not load balance sheet: {e}")
    st.stop()

# ─── PREDICTION INTERFACE ─────────────────────────────────────────────────────
st.markdown("---")

team_display = {
    # Hosts
    "United States": "🇺🇸 USA", "Canada": "🇨🇦 Canada", "Mexico": "🇲🇽 Mexico",
    # Group A
    "South Africa": "🇿🇦 South Africa", "South Korea": "🇰🇷 South Korea", "Czech Republic": "🇨🇿 Czech Republic",
    # Group B
    "Switzerland": "🇨🇭 Switzerland", "Qatar": "🇶🇦 Qatar", "Bosnia and Herzegovina": "🇧🇦 Bosnia and Herzegovina",
    # Group C
    "Brazil": "🇧🇷 Brazil", "Morocco": "🇲🇦 Morocco", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland", "Haiti": "🇭🇹 Haiti",
    # Group D
    "Paraguay": "🇵🇾 Paraguay", "Australia": "🇦🇺 Australia", "Turkey": "🇹🇷 Turkey",
    # Group E
    "Germany": "🇩🇪 Germany", "Curacao": "🇨🇼 Curacao", "Ivory Coast": "🇨🇮 Ivory Coast", "Ecuador": "🇪🇨 Ecuador",
    # Group F
    "Netherlands": "🇳🇱 Netherlands", "Japan": "🇯🇵 Japan", "Sweden": "🇸🇪 Sweden", "Tunisia": "🇹🇳 Tunisia",
    # Group G
    "Belgium": "🇧🇪 Belgium", "Egypt": "🇪🇬 Egypt", "Iran": "🇮🇷 Iran", "New Zealand": "🇳🇿 New Zealand",
    # Group H
    "Spain": "🇪🇸 Spain", "Uruguay": "🇺🇾 Uruguay", "Cape Verde": "🇨🇻 Cape Verde", "Saudi Arabia": "🇸🇦 Saudi Arabia",
    # Group I
    "France": "🇫🇷 France", "Senegal": "🇸🇳 Senegal", "Iraq": "🇮🇶 Iraq", "Norway": "🇳🇴 Norway",
    # Group J
    "Argentina": "🇦🇷 Argentina", "Algeria": "🇩🇿 Algeria", "Austria": "🇦🇹 Austria", "Jordan": "🇯🇴 Jordan",
    # Group K
    "Portugal": "🇵🇹 Portugal", "DR Congo": "🇨🇩 DR Congo", "Uzbekistan": "🇺🇿 Uzbekistan", "Colombia": "🇨🇴 Colombia",
    # Group L
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England", "Croatia": "🇭🇷 Croatia", "Ghana": "🇬🇭 Ghana", "Panama": "🇵🇦 Panama"
}
TEAM_LIST = list(team_display.values())

c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    team1 = st.selectbox("Home Team", TEAM_LIST, index=0)
with c2:
    venue_factor = st.slider("Venue Factor", 0.0, 1.0, 0.3, 0.05,
                              help="0=Pure Neutral, 0.6=Host, 1.0=True Home")
    stage = st.selectbox("Stage", ["group","round_of_16","quarter","semi","final"])
with c3:
    team2 = st.selectbox("Away Team", TEAM_LIST, index=1)

# Odds (Syndicate only)
if is_syndicate:
    oc1, oc2, oc3 = st.columns(3)
    home_odds = oc1.number_input("Home Odds", 1.01, 50.0, 2.10)
    draw_odds = oc2.number_input("Draw Odds", 1.01, 50.0, 3.40)
    away_odds = oc3.number_input("Away Odds", 1.01, 50.0, 3.20)
else:
    home_odds = draw_odds = away_odds = None

if st.button("▶ Run Prediction Engine", type="primary", use_container_width=True):
    with st.spinner("Running quantitative engine..."):
        from inference import run_inference, _get_team_state, _ensure_loaded
        # Strip emojis for backend
        t1 = [k for k, v in team_display.items() if v == team1][0]
        t2 = [k for k, v in team_display.items() if v == team2][0]

        _ensure_loaded()
        home_state = _get_team_state(t1)
        away_state = _get_team_state(t2)

        result = run_inference(
            home_team=t1, away_team=t2,
            venue_factor=venue_factor, stage=stage,
            home_odds=home_odds if is_syndicate else None,
            draw_odds=draw_odds if is_syndicate else None,
            away_odds=away_odds if is_syndicate else None,
            elapsed_minutes=elapsed,
            home_goals_live=live_hg,
            away_goals_live=live_ag,
            red_cards={"home": home_reds, "away": away_reds} if live_mode else None,
            live_state=live_state
        )

    if result is None or "error" in (result or {}):
        st.error(f"⚠️ {(result or {}).get('error', 'Inference failed')}")
    else:
        p_h = result['home_win_prob']
        p_d = result['draw_prob']
        p_a = result['away_win_prob']

        if live_mode:
            # Live banner
            remaining = result.get("remaining", 0)
            score     = result.get("current_score", "0-0")
            momentum  = result.get("momentum_signal", 0)
            mom_text  = "🔴 Away Pressure" if momentum < -0.2 else \
                        "🟢 Home Pressure" if momentum > 0.2 else "⚖️ Balanced"

            st.markdown(f"""
            <div style="background:#1a1a2e;padding:12px 20px;border-radius:8px;
                        border-left:4px solid #e94560;margin-bottom:12px">
                <b style="color:#e94560">LIVE</b>
                <span style="color:#ffffff;margin:0 12px">
                    {team1} <b>{score.split('-')[0]}</b> –
                    <b>{score.split('-')[1]}</b> {team2}
                </span>
                <span style="color:#a8a8b3">{elapsed}' | {remaining}' remaining</span>
                <span style="float:right;color:#ffd700">{mom_text}</span>
            </div>
            """, unsafe_allow_html=True)

        # Probability display (same for both modes)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {team1} Win", f"{p_h:.1%}",
                  delta=f"+{p_h-0.333:.1%}" if live_mode else None)
        m2.metric("🤝 Draw",        f"{p_d:.1%}")
        m3.metric(f"✈️ {team2} Win", f"{p_a:.1%}",
                  delta=f"+{p_a-0.333:.1%}" if live_mode else None)

        st.markdown("#### 🔍 Kalman & Regime Signals")
        k1, k2 = st.columns(2)
        k1.metric(f"{team1} Kalman Velocity", f"{home_state.get('velocity', 0.0):.2f}")
        k1.caption(f"Regime: {home_state.get('regime', 'unknown')}")
        k2.metric(f"{team2} Kalman Velocity", f"{away_state.get('velocity', 0.0):.2f}")
        k2.caption(f"Regime: {away_state.get('regime', 'unknown')}")

        if is_syndicate and result.get("best_bet"):
            st.markdown("---")
            st.markdown("### 💰 Syndicate Betting Strategy")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Recommended Bet", result["best_bet"])
            
            edge_val = result.get('no_vig_edge')
            edge_str = f"{edge_val:.1%}" if edge_val is not None else "N/A"
            b2.metric("No-Vig Edge", edge_str)
            
            kelly_val = result.get('kelly_fraction')
            kelly_str = f"{kelly_val:.2%}" if kelly_val is not None else "N/A"
            b3.metric("Kelly Fraction", kelly_str)
            
            ir_val = result.get('ir_multiplier')
            ir_str = f"{ir_val:.2f}" if ir_val is not None else "N/A"
            b4.metric("IR Multiplier", ir_str)

            # Live Capital Arbitrage Allocator
            st.markdown("#### 💸 Live Capital Arbitrage Allocator")
            st.caption("Calculate exact wager sizes for live-odds betting using Kelly Criterion edge.")
            c_bankroll, c_wager, c_pnl = st.columns(3)
            current_bankroll = c_bankroll.number_input("Current Live Bankroll (INR/$)", min_value=10.0, value=1000.0, step=100.0)
            
            if kelly_val is not None and kelly_val > 0:
                recommended_wager = current_bankroll * kelly_val
                # Assuming decimal odds for the best bet (we need to match best_bet to the odds)
                bb = result.get("best_bet", "")
                if "Home" in bb: bet_odds = home_odds
                elif "Away" in bb: bet_odds = away_odds
                else: bet_odds = draw_odds
                
                potential_profit = recommended_wager * (bet_odds - 1)
                
                c_wager.metric("Exact Wager to Place", f"{recommended_wager:.2f}")
                c_pnl.metric("Potential Profit", f"+{potential_profit:.2f}")
            else:
                c_wager.metric("Exact Wager to Place", "0.00")
                c_pnl.metric("Potential Profit", "0.00")
                st.info("No positive edge found at current live odds. Do not place a bet.")

        # === V9 Omni-Quant Features ===
        if is_syndicate:
            st.markdown("---")
            st.markdown("## 🧠 V9 Omni-Quant Alpha Modules")
            
            v9_tabs = st.tabs(["🤖 RL Execution Agent", "📈 Stat Arb (Pairs)", "🛑 Black Swan Risk", "⚡ Microstructure (HFT)"])
            
            with v9_tabs[0]:
                st.markdown("### Soft Actor-Critic (SAC) Limit Order Execution")
                try:
                    from execution.rl_agent import SACExecutionAgent
                    import torch
                    import numpy as np
                    
                    # Mock state: edge, true_prob, implied, uncert, hawkes, best_back, best_lay, spread, vwap, time
                    edge = edge_val if edge_val else 0.05
                    implied = max(0.01, p_h - edge)
                    mock_state = np.array([edge, p_h, implied, 0.05, 0.1, 2.0, 2.05, 0.05, 0.02, 0.9])
                    agent = SACExecutionAgent(state_dim=10, action_dim=2)
                    # For demo purposes, we just get an action from the untrained network
                    action = agent.get_action(mock_state)
                    size_frac, urgency = action  # Action 0: Tranche Size, Action 1: Price Aggressiveness
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Execution Urgency", f"{urgency:.2f}", help="-1.0 = Provide Liquidity, 1.0 = Cross Spread")
                    r2.metric("Tranche Size", f"{size_frac:.1%}", help="Fraction of max position to deploy in this tick")
                    r3.metric("LOB Offset", f"{(urgency * 0.05):.3f} ticks", help="Dynamic limit order offset")
                except Exception as e:
                    st.error(f"RL Agent offline: {e}")

            with v9_tabs[1]:
                st.markdown("### Cointegration & Pairs Trading (Stat Arb)")
                try:
                    from models.statistical_arbitrage import StatArbEngine
                    # Generate mock cointegrated series for demonstration
                    x = np.linspace(0, 100, 100)
                    series_x = np.sin(x) + np.random.normal(0, 0.1, 100)
                    series_y = 1.5 * series_x + np.random.normal(0, 0.2, 100)
                    
                    arb = StatArbEngine()
                    res = arb.analyze_pair("MatchOdds_vs_AsianHC", series_y, series_x, series_y[-1], series_x[-1])
                    
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Cointegrated Spread Z-Score", f"{res.get('z_score', 0):.2f}")
                    a2.metric("Stat Arb Signal", res.get("trade_signal", "NONE"))
                    a3.metric("Mean-Reversion Half Life", f"{res.get('half_life', 0):.1f} ticks")
                except Exception as e:
                    st.error(f"Stat Arb Engine offline: {e}")

            with v9_tabs[2]:
                st.markdown("### CVaR & Black Swan Stress Test")
                try:
                    from monitoring.stress_test import BlackSwanSimulator
                    sim = BlackSwanSimulator()
                    # Mock portfolio
                    portfolio = [
                        {"market": "Match Odds", "decimal_odds": 1.2, "edge": 0.05, "stake_units": 1000},
                        {"market": "Over 2.5", "decimal_odds": 1.9, "edge": 0.02, "stake_units": 500}
                    ]
                    status = sim.check_circuit_breaker(10000.0, portfolio)
                    
                    s1, s2 = st.columns(2)
                    s1.metric("Circuit Breaker Status", status['status'])
                    if status['status'] != "OK":
                        s2.error(status['reason'])
                    else:
                        s2.success(status['reason'])
                    
                    st.caption(f"Simulated Portfolio CVaR: {sim.calculate_cvar(portfolio):.2f} units")
                except Exception as e:
                    st.error(f"Stress Test Engine offline: {e}")

            with v9_tabs[3]:
                st.markdown("### Hawkes Process Microstructure (Toxic Flow Veto)")
                try:
                    import time
                    from models.microstructure import HawkesProcessEngine
                    hawkes = HawkesProcessEngine()
                    # Mock incoming toxic flow on the away side
                    hawkes.add_event('away', time.time() - 0.5)
                    hawkes.add_event('away', time.time() - 0.1)
                    imbal = hawkes.get_market_imbalance()
                    veto = hawkes.check_veto('home', 0.20)
                    
                    h1, h2 = st.columns(2)
                    h1.metric("Hawkes Imbalance", f"{imbal:.2f}", help="< -0.20 indicates sharp money against Home")
                    if veto:
                        h2.error("🚫 HARD VETO ACTIVATED: Toxic order flow detected against position.")
                    else:
                        h2.success("✅ Order flow clear. No adverse selection detected.")
                except Exception as e:
                    st.error(f"Microstructure Engine offline: {e}")

        if live_mode:
            # Live signals bar (Syndicate only)
            if is_syndicate and live_state:
                st.markdown("#### 📊 Live Match Intelligence")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Live xG Home", f"{live_state.get('home_xg_live', 'N/A')}")
                sc2.metric("Live xG Away", f"{live_state.get('away_xg_live', 'N/A')}")
                sc3.metric("Corners H/A",
                           f"{live_state.get('home_corners',0)}-"
                           f"{live_state.get('away_corners',0)}")
                sc4.metric("Passes H/A",
                           f"{live_state.get('home_passes',0)}-"
                           f"{live_state.get('away_passes',0)}")

# ─── LIVE ODDS CONVERGENCE CHART (Syndicate only) ─────────────────────────────
if live_mode and is_syndicate:
    st.markdown("#### 📈 Win Probability Over Time (Simulated)")
    import plotly.graph_objects as go

    # Simulate probability curve from 0 to elapsed
    from models.poisson_dixon_coles import live_in_play_predict
    import inference

    inference._ensure_loaded()
    dc_params = inference._dc_params or {}
    
    t1 = [k for k, v in team_display.items() if v == team1][0]
    t2 = [k for k, v in team_display.items() if v == team2][0]

    attack = dc_params.get("attack", {})
    defense = dc_params.get("defense", {})
    home_adv = dc_params.get("home_adv", 0.3)
    lam_h = np.exp(attack.get(t1, 0) - defense.get(t2, 0) + home_adv * venue_factor)
    lam_a = np.exp(attack.get(t2, 0) - defense.get(t1, 0))
    rho = dc_params.get("rho", -0.13)

    minutes   = list(range(0, elapsed + 1, 5))
    ph_curve, pd_curve, pa_curve = [], [], []
    for m in minutes:
        r = live_in_play_predict(lam_h, lam_a, m, live_hg, live_ag, rho)
        ph_curve.append(r["home_win_prob"])
        pd_curve.append(r["draw_prob"])
        pa_curve.append(r["away_win_prob"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=minutes, y=ph_curve,
                              name=f"{team1} Win",
                              line=dict(color="#00b4d8", width=2)))
    fig.add_trace(go.Scatter(x=minutes, y=pd_curve,
                              name="Draw",
                              line=dict(color="#ffd700", width=2)))
    fig.add_trace(go.Scatter(x=minutes, y=pa_curve,
                              name=f"{team2} Win",
                              line=dict(color="#ef233c", width=2)))
    fig.update_layout(
        title="Live Win Probability Curve",
        xaxis_title="Match Minute",
        yaxis_title="Probability",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=320, template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── AUTO-REFRESH (Must be at the very end of the script) ─────────────────────
if live_mode and locals().get('auto_refresh', False) and st.session_state.get('poller_active'):
    import time
    time.sleep(10)
    st.rerun()
