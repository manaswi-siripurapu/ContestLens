"""
app.py — LeetCode Contest Anomaly Explorer
Streamlit frontend that ties together all modules.

Run:  streamlit run app.py
Env:  ANTHROPIC_API_KEY  (optional — enables LLM explanations)
"""

import os
import math
import time
import uuid
import json
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import cache
import fetcher
import features as feat_module
import scorer as scorer_module
import explainer as explainer_module

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LeetCode Anomaly Explorer",
    page_icon="code.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------

cache.init_db()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

SESSION_ID = st.session_state["session_id"]

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .score-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .score-danger  { background: #fff1f0; border: 1.5px solid #ffccc7; }
    .score-warning { background: #fffbe6; border: 1.5px solid #ffe58f; }
    .score-caution { background: #e6f7ff; border: 1.5px solid #91d5ff; }
    .score-normal  { background: #f6ffed; border: 1.5px solid #b7eb8f; }
    .disclaimer {
        font-size: 0.78rem;
        color: #888;
        padding: 10px;
        border-left: 3px solid #ddd;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "GROQ API Key (optional)",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Enables AI-generated explanations. Leave blank to use rule-based fallback.",
    )
    st.markdown("---")
    deep_analysis = st.toggle(
        "Deep analysis (contest timing)",
        value=False,
        help="Fetches per-problem solve times from contest ranking pages. Adds ~15s per lookup but significantly improves accuracy.",
    )
    st.markdown("---")
    
    st.markdown("""
**About this tool**

This is a small personal project where I explore whether patterns in LeetCode contest performance can tell us something interesting. By looking at things like solve speed, rating changes, practice consistency, and submission habits, it builds an “anomaly index”, just a way to highlight unusual patterns.
It’s not meant to judge anyone or claim anything serious — just a fun, data-driven way to look at things for those who are curious like me 🙂
\n👉 [Report concerns to LeetCode](https://support.leetcode.com)
""")



with st.expander("How the Anomaly Index is Calculated", expanded=False):
    st.markdown("""
**Performance Patterns**
- **Solve time vs field** *(deep analysis, 20% weight)* — Are harder problems solved faster than easier ones? Non-monotonic solve order across Q1→Q4 is the strongest signal.
- **Contest solve speed** *(14% weight)* — How fast is a full 4-problem clear? Under 15 minutes is world-record territory. Mitigated if 800+ problems solved.
- **Early contest performance** *(6% weight)* — How strong was the debut across the first 3 contests?

**Consistency Checks**
- **Ranking consistency** *(12% weight)* — Is rank suspiciously stable across all contests? Real humans have off days.
- **Submission entropy** *(4% weight)* — Is the submission schedule unnaturally repetitive? Very low day-of-week entropy suggests scripted activity.
- **Language distribution** *(1% weight, near-zero)* — Copiers tend to use one language consistently, so a split is actually neutral.

**Practice vs Results**
- **Percentile vs problems solved** *(16% weight)* — Does practice depth match contest percentile? Top 1% globally with 24 problems solved is a hard contradiction.
- **Profile depth** *(9% weight)* — Are problems solved consistent with rating bracket?
- **Rating velocity** *(11% weight)* — How large are single-contest rating jumps? Legitimate grinders gain +50–150 per contest on average.

**Transparency Signals**
- **Hidden submission graph** *(7% weight)* — High-rated players rarely hide their practice history. Hidden graph + high rating is suspicious.

---
⚠️ **This is not a verdict.** Scores reflect statistical patterns in public data only.  
Legitimate elite programmers (ICPC medalists, heavy grinders) may trigger certain signals.  
Always use your own judgement — this tool is a starting point, not a conclusion.
    """)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("LeetCode Contest Anomaly Explorer")
st.markdown(
    "Enter a LeetCode username to analyse contest behaviour patterns "
    "and surface statistical anomalies."
)

st.markdown("""
<div class="disclaimer">
⚠️ This tool surfaces patterns from public data. A high anomaly score is not proof of cheating.
Legitimate users — especially elite competitive programmers — may trigger certain signals.
Do not use this output as sole basis for any action against a user.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

col_in, col_btn, col_pad = st.columns([3, 1, 3])

with col_in:
    username_input = st.text_input(
        "LeetCode username",
        placeholder="e.g. leetcoder",
        label_visibility="collapsed",
    )

with col_btn:
    analyse_clicked = st.button("Analyse", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_username(u: str) -> tuple[bool, str]:
    u = u.strip()
    if not u:
        return False, "Please enter a username."
    if len(u) < 2:
        return False, "Username is too short."
    if len(u) > 50:
        return False, "Username is too long (max 50 chars)."
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    if not all(c in allowed for c in u):
        return False, "Username contains invalid characters."
    return True, ""


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _chart_rating_trajectory(contests: list[dict]) -> go.Figure:
    attended = [c for c in contests if c.get("rating", 0) > 0]
    if not attended:
        return None

    df = pd.DataFrame(attended)
    df["date"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
    df["contest_n"] = range(1, len(df) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["rating"],
        mode="lines+markers",
        name="Rating",
        line=dict(color="#5B8FF9", width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>Rating: %{y:.0f}<extra></extra>",
        text=df["title"],
    ))

    # Highlight top-3 spikes
    if len(df) >= 3:
        df["delta"] = df["rating"].diff().fillna(0)
        top_spikes = df.nlargest(3, "delta")
        fig.add_trace(go.Scatter(
            x=top_spikes["date"],
            y=top_spikes["rating"],
            mode="markers",
            name="Big jump",
            marker=dict(size=12, color="#F4664A", symbol="star"),
            hovertemplate="<b>%{text}</b><br>Jump: +%{customdata:.0f}<extra></extra>",
            text=top_spikes["title"],
            customdata=top_spikes["delta"],
        ))

    fig.update_layout(
        title="Contest rating trajectory",
        xaxis_title="Date",
        yaxis_title="Rating",
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    return fig


def _chart_submission_heatmap(profile: dict) -> go.Figure:
    calendar: dict = profile.get("calendar", {})
    if not calendar:
        return None

    dow_counts = [0] * 7
    week_data: dict[int, dict[int, int]] = {}  # week_num -> {dow: count}

    for ts_str, count in calendar.items():
        try:
            ts = int(ts_str)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            dow = dt.weekday()
            dow_counts[dow] += int(count)
            iso = dt.isocalendar()
            week_key = iso.year * 100 + iso.week
            if week_key not in week_data:
                week_data[week_key] = {}
            week_data[week_key][dow] = week_data[week_key].get(dow, 0) + int(count)
        except (ValueError, OSError, OverflowError):
            continue

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Bar(
        x=days,
        y=dow_counts,
        marker_color="#5B8FF9",
        hovertemplate="<b>%{x}</b><br>Submissions: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Submission activity by day of week",
        xaxis_title="Day",
        yaxis_title="Total submissions",
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def _chart_contest_performance(contests: list[dict]) -> go.Figure:
    attended = [c for c in contests if c.get("attended", True) and c.get("ranking", 0) > 0]
    if len(attended) < 2:
        return None

    df = pd.DataFrame(attended)
    df["date"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
    df["color"] = df["problems_solved"] / df["total_problems"].clip(lower=1)

    fig = go.Figure(go.Scatter(
        x=df["date"],
        y=df["ranking"],
        mode="markers",
        marker=dict(
            size=9,
            color=df["color"],
            colorscale="RdYlGn",
            colorbar=dict(title="Solve ratio", thickness=10),
            showscale=True,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Rank: #%{y}<br>"
            "Solved: %{customdata[0]}/%{customdata[1]}<extra></extra>"
        ),
        text=df["title"],
        customdata=df[["problems_solved", "total_problems"]].values,
    ))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title="Contest rank over time (lower = better, colour = solve ratio)",
        xaxis_title="Date",
        yaxis_title="Contest rank",
        height=300,
        margin=dict(t=40, b=40, l=60, r=20),
    )
    return fig


def _chart_feature_scores(feature_scores: list[dict]) -> go.Figure:
    active = [f for f in feature_scores if f.get("eff_weight", 0) > 0]
    if not active:
        return None

    active_sorted = sorted(active, key=lambda x: x["score"])
    labels = [f["label"] for f in active_sorted]
    scores = [f["score"] for f in active_sorted]
    colors = [
        "#F4664A" if s >= 7 else "#FAAD14" if s >= 4 else "#52C41A"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x}/10<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 11])
    fig.update_layout(
        title="Feature scores (red = high anomaly)",
        xaxis_title="Anomaly score (0-10)",
        height=280,
        margin=dict(t=40, b=40, l=200, r=60),
    )
    return fig


def _gauge_chart(score: float, verdict: str, severity: str) -> go.Figure:
    color_map = {
        "danger":  "#F5222D",
        "warning": "#FA8C16",
        "caution": "#1890FF",
        "normal":  "#52C41A",
    }
    color = color_map.get(severity, "#000000")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "/10", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0, 2.5],  "color": "#d4faad"},
                {"range": [2.5, 4.5], "color": "#fee3ab"},
                {"range": [4.5, 6.5], "color": "#ffba90"},
                {"range": [6.5, 10],  "color": "#fd9191"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_button(username: str) -> None:
    """Show a Retry button that forces a fresh fetch bypassing cache."""
    if st.button("🔄 Retry", key=f"retry_btn_{username}"):
        st.session_state[f"retry_{username}"] = True
        st.rerun()


# ---------------------------------------------------------------------------
# Main analysis flow
# ---------------------------------------------------------------------------

def run_analysis(username: str, api_key: str, deep: bool = False) -> None:
    username = username.strip()

    # --- Rate limit ---
    allowed, remaining = cache.check_rate_limit(SESSION_ID)
    if not allowed:
        st.error(
            f"⏱️ Rate limit reached. You can look up {cache.RATE_LIMIT_MAX} profiles "
            f"per hour. Reset in ~{remaining // 60} min."
        )
        return

    st.markdown(f"### Results for `{username}`")

    # --- Cache check ---
    age_mins      = cache.cache_age_minutes(username)
    cached_data   = cache.get_cached(username)
    force_refresh = st.session_state.pop(f"retry_{username}", False)

    if cached_data and not force_refresh:
        st.caption(f"Using cached data from {age_mins:.0f} min ago.")
        raw = cached_data
    else:
        with st.spinner("Fetching data from LeetCode… (may take 5–10 s)"):
            try:
                raw = fetcher.fetch_all(username)
                cache.set_cached(username, raw)
            except fetcher.UserNotFoundError as e:
                st.error(f"❌ {e}")
                return
            except fetcher.RateLimitedError as e:
                st.warning(f"⏳ {e}")
                _retry_button(username)
                return
            except fetcher.NetworkError as e:
                st.error(f"🌐 {e}")
                _retry_button(username)
                return
            except fetcher.PrivateProfileError as e:
                st.warning(f"🔒 {e}")
                return
            except fetcher.LeetCodeError as e:
                st.error(f"⚠️ {e}")
                _retry_button(username)
                return
            except Exception as e:
                st.error(f"Unexpected error: {type(e).__name__}: {e}")
                _retry_button(username)
                return

    profile   = raw.get("profile", {})
    contests  = raw.get("contests", [])

    # --- Validate data completeness ---
    if not profile:
        st.error("Profile data is empty. The account may have been deleted or is private.")
        return

    # --- Profile header ---
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.metric("Contest rating", f"{profile.get('contest_rating', 0):.0f}")
    with p_col2:
        st.metric("Contests attended", profile.get("attended_count", 0))
    with p_col3:
        total_solved = profile.get("problems", {}).get("All", 0)
        st.metric("Problems solved", total_solved)
    with p_col4:
        top_pct = profile.get("top_percentage")
        if top_pct is not None:
            st.metric("Top percentile", f"{top_pct:.2f}%")
        else:
            st.metric("Top percentile", "N/A")

    st.markdown("---")

    # --- Feature computation ---
    with st.spinner("Computing anomaly signals…"):
        all_features = feat_module.compute_all_features(
            profile, contests,
            username=username,
            include_contest_timing=deep,
        )
        score_result  = scorer_module.compute_anomaly_score(all_features)

    # --- Score display ---
    score    = score_result["score"]
    verdict  = score_result["verdict"]
    severity = score_result["severity"]
    quality  = score_result["data_quality"]

    gauge_col, verdict_col = st.columns([1, 1])

    with gauge_col:
        gauge = _gauge_chart(score, verdict, severity)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    with verdict_col:
        sev_class = f"score-{severity}"
        emoji = {"danger": "🔴", "warning": "🟠", "caution": "🔵", "normal": "🟢"}.get(severity, "⚪")
        st.markdown(f"""
<div class="score-card {sev_class}">
    <h2 style="color:#000">{emoji} {verdict} anomaly</h2>
    <p style="color:#000; font-size:1.1rem; margin:4px 0">Anomaly index: <strong>{score}/10</strong></p>
    <p style="color:#888; font-size:0.85rem">
        Data quality: <strong>{quality}</strong> · 
        {score_result['active_features']} signals active
    </p>
</div>
""", unsafe_allow_html=True)

        if quality == "insufficient":
            st.warning(
                "⚠️ **Limited data**: This profile hasn't attended enough contests, "
                "or key data fields were unavailable. Score reliability is low."
            )
        elif quality == "partial":
            st.info(
                "ℹ️ Some signals had insufficient data and were excluded from scoring."
            )

    st.markdown("---")

    # --- LLM Explanation ---
    st.subheader("📝 Analysis summary")
    with st.spinner("Generating explanation…"):
        explanation = explainer_module.generate_explanation(
            username, score_result, all_features, api_key=api_key or None
        )
    st.markdown(explanation)

    st.markdown("---")

    # --- Charts ---
    st.subheader("📊 Signal breakdown")

    feat_chart = _chart_feature_scores(score_result["feature_scores"])
    if feat_chart:
        st.plotly_chart(feat_chart, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No feature data available to chart.")

    # --- Feature detail expanders ---
    with st.expander("🔬 Per-feature detail", expanded=False):
        bonus = score_result.get("interaction_bonus", 0)
        if bonus > 0:
            st.info(
                f"⚡ **Interaction bonus: +{bonus:.2f}** — "
                "fast solve speed combined with shallow problem history "
                "amplified the final score."
            )
        for feat in all_features:
            conf = feat.get("confidence", "low")
            score_val = feat.get("score", 0)
            icon = "🔴" if score_val >= 7 else "🟠" if score_val >= 4 else "🟢"
            conf_badge = {"high": "✅ High confidence", "medium": "⚠️ Medium", "low": "❓ Low (excluded)"}.get(conf, conf)
            st.markdown(
                f"**{icon} {feat['label']}** — Score: `{score_val}/10` · {conf_badge}"
            )
            st.caption(feat.get("description", "—"))

    st.markdown("---")

    # --- Rating trajectory ---
    st.subheader("📈 Contest history")

    if contests:
        traj = _chart_rating_trajectory(contests)
        if traj:
            st.plotly_chart(traj, use_container_width=True, config={"displayModeBar": False})

        perf = _chart_contest_performance(contests)
        if perf:
            st.plotly_chart(perf, use_container_width=True, config={"displayModeBar": False})

        # Raw contest table
        with st.expander("Contest table", expanded=False):
            df_contests = pd.DataFrame(contests)
            if not df_contests.empty:
                df_display = df_contests.copy()
                df_display["date"] = pd.to_datetime(
                    df_display["start_time"], unit="s", utc=True
                ).dt.strftime("%Y-%m-%d")
                cols = [c for c in ["date", "title", "rating", "ranking",
                                     "problems_solved", "total_problems",
                                     "finish_seconds"] if c in df_display.columns]
                st.dataframe(df_display[cols], use_container_width=True, height=300)
    else:
        st.info("This user has not attended any rated contests.")

    st.markdown("---")

    # --- Submission patterns ---
    st.subheader("📅 Submission patterns")

    heat = _chart_submission_heatmap(profile)
    if heat:
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})

        # Entropy value callout
        entropy_feat = next(
            (f for f in all_features if f.get("label") == "Submission pattern entropy"), None
        )
        if entropy_feat and entropy_feat.get("value") is not None:
            ent_val = entropy_feat["value"]
            max_ent = math.log2(7)
            st.caption(
                f"Day-of-week entropy: **{ent_val:.3f} bits** "
                f"(max possible: {max_ent:.3f} bits). "
                f"Lower = more concentrated on specific days."
            )
    else:
        st.info("No submission calendar data available.")

    st.markdown("---")

    # --- Language breakdown ---
    langs = profile.get("languages", {})
    if langs and sum(langs.values()) > 0:
        st.subheader("💻 Language breakdown")
        lang_df = pd.DataFrame(
            sorted(langs.items(), key=lambda x: -x[1])[:10],
            columns=["Language", "Problems solved"],
        )
        fig_lang = px.bar(
            lang_df, x="Problems solved", y="Language",
            orientation="h", color="Problems solved",
            color_continuous_scale="Blues", height=240,
        )
        fig_lang.update_layout(
            margin=dict(t=20, b=20, l=100, r=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_lang, use_container_width=True, config={"displayModeBar": False})

    # --- Footer disclaimer ---
    st.markdown("""
<div class="disclaimer">
This tool performs automated statistical analysis on publicly available data.
A high anomaly score is not evidence of rule violations — it indicates patterns
that are statistically unusual and may warrant further investigation.
If you believe a user has violated LeetCode's Terms of Service,
<a href="https://support.leetcode.com" target="_blank">report it to LeetCode directly</a>.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if analyse_clicked:
    valid, err_msg = validate_username(username_input)
    if not valid:
        st.error(err_msg)
    else:
        run_analysis(username_input.strip(), api_key_input, deep=deep_analysis)
elif not username_input:
    # Landing state — show example profiles
    st.markdown("### Try these example lookups")
    cols = st.columns(4)
    example_users = ["cpcs", "Ma_Lin", "votrubac", "PhoenixDD"]
    for i, user in enumerate(example_users):
        with cols[i]:
            if st.button(f"{user}", use_container_width=True):
                st.session_state["prefill"] = user
                st.rerun()

    if "prefill" in st.session_state:
        prefill = st.session_state.pop("prefill")
        run_analysis(prefill, api_key_input, deep=deep_analysis)
