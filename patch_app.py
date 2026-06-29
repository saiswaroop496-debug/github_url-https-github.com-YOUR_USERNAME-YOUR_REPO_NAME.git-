import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

sidebar_code = """
# ─── LIVE IN-PLAY SIDEBAR TOGGLE ─────────────────────────────────────────────

st.sidebar.markdown("---")
live_mode = st.sidebar.toggle("📡 Live In-Play Mode", value=False)

if live_mode:
    st.sidebar.markdown("### Live Match State")
    elapsed   = st.sidebar.slider("Elapsed Minutes", 0, 90, 45)
    live_hg   = st.sidebar.number_input("Home Goals", 0, 10, 0, step=1)
    live_ag   = st.sidebar.number_input("Away Goals", 0, 10, 0, step=1)
    home_reds = st.sidebar.number_input("Home Red Cards", 0, 3, 0, step=1)
    away_reds = st.sidebar.number_input("Away Red Cards", 0, 3, 0, step=1)

    # Auto-poll toggle
    auto_poll = st.sidebar.checkbox("🔄 Auto-refresh every 60s", value=False)
    if auto_poll:
        st.sidebar.info("Polling API-Football live data...")
        if st.sidebar.button("Fetch Live State Now"):
            from data.live_match_poller import get_live_match_state
            live_state = get_live_match_state()
            if live_state:
                elapsed  = live_state.get("elapsed", elapsed)
                live_hg  = live_state.get("home_goals", live_hg)
                live_ag  = live_state.get("away_goals", live_ag)
                st.sidebar.success(f"Live: {live_hg}-{live_ag} @ {elapsed}'")
    else:
        live_state = None
else:
    elapsed = live_hg = live_ag = home_reds = away_reds = None
    live_state = None

"""

content = content.replace("    st.rerun()\n", "    st.rerun()\n" + sidebar_code)

button_logic = """if st.button("▶ Run Prediction Engine", type="primary", use_container_width=True):
    with st.spinner("Running quantitative engine..."):
        from inference import run_inference
        # Strip emojis for backend
        t1 = [k for k, v in team_display.items() if v == team1][0]
        t2 = [k for k, v in team_display.items() if v == team2][0]

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
            mom_text  = "🔴 Away Pressure" if momentum < -0.2 else \\
                        "🟢 Home Pressure" if momentum > 0.2 else "⚖️ Balanced"

            st.markdown(f\"\"\"
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
            \"\"\", unsafe_allow_html=True)

        # Probability display (same for both modes)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {team1} Win", f"{p_h:.1%}",
                  delta=f"+{p_h-0.333:.1%}" if live_mode else None)
        m2.metric("🤝 Draw",        f"{p_d:.1%}")
        m3.metric(f"✈️ {team2} Win", f"{p_a:.1%}",
                  delta=f"+{p_a-0.333:.1%}" if live_mode else None)

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
"""

# Find the line containing 'Run Prediction Engine'
lines = content.split('\n')
start_idx = -1
for i, line in enumerate(lines):
    if "Run Prediction Engine" in line and "st.button" in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find button")
    import sys; sys.exit(1)

# Replace the lines from start_idx onwards
content = '\n'.join(lines[:start_idx]) + '\n' + button_logic

chart_code = """
# ─── LIVE ODDS CONVERGENCE CHART (Syndicate only) ─────────────────────────────
if live_mode and is_syndicate:
    st.markdown("#### 📈 Win Probability Over Time (Simulated)")
    import plotly.graph_objects as go

    # Simulate probability curve from 0 to elapsed
    from models.poisson_dixon_coles import live_in_play_predict
    from inference import _load_dc_params, _compute_prematch_lambdas, _load_team_states

    dc_params   = _load_dc_params()
    team_states = _load_team_states()
    
    t1 = [k for k, v in team_display.items() if v == team1][0]
    t2 = [k for k, v in team_display.items() if v == team2][0]

    lam_h, lam_a = _compute_prematch_lambdas(t1, t2, venue_factor,
                                               dc_params, team_states)
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
"""

content += chart_code

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("patched app.py successfully")
