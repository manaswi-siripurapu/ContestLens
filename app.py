"""
app.py — ContestLens
Streamlit frontend for LeetCode contest profile anomaly analysis.

Run:  streamlit run app.py
Env:  GROQ_API_KEY  (optional — enables AI-generated explanations)
"""

import os
import math
import uuid
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import cache
import fetcher
import features as feat_module
import scorer as scorer_module
import explainer as explainer_module

try:
    from percentile_store import init_percentile_table, record_score, get_percentile
    init_percentile_table()
    _PERCENTILE_OK = True
except Exception:
    _PERCENTILE_OK = False
    def record_score(*a, **k): pass
    def get_percentile(*a, **k): return {"enough_data": False, "total_profiles": 0}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ContestLens",
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
    .notice {
        font-size: 0.80rem;
        color: #888;
        padding: 10px 14px;
        border-left: 3px solid #ddd;
        margin-top: 10px;
        margin-bottom: 4px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "Groq API Key (optional)",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Enables AI-generated explanations. Leave blank to use the built-in fallback.",
    )
    st.markdown("---")
    deep_analysis = st.toggle(
        "🔬 Deep analysis (contest timing)",
        value=False,
        help=(
            "Fetches per-problem solve times from contest ranking pages "
            "and checks whether harder problems took longer than easier ones. "
            "Adds ~15s per lookup. Enables the highest-weight signal (20%)."
        ),
    )
    st.markdown("---")
    st.markdown("""
**About ContestLens**

A personal side project built out of curiosity — I wanted to see whether publicly
available contest data contains statistically interesting patterns.

ContestLens looks at things like solve speed relative to practice history,
rating changes over time, and submission habits to produce an **Anomaly Index**.

This is a data exploration tool, not a moderation tool. A high score means
*"this profile has unusual patterns worth a closer look"* — nothing more.

👉 [Report concerns to LeetCode](https://support.leetcode.com)
""")

# ---------------------------------------------------------------------------
# How it works — expander
# ---------------------------------------------------------------------------

with st.expander("How the Anomaly Index is calculated", expanded=False):
    st.markdown("""
The Anomaly Index is a weighted combination of 9 independent signals.
Each signal scores 0–10 based on how unusual that particular pattern is.
Signals with insufficient data are excluded automatically.

**Performance signals**
- **Solve time vs field** *(deep analysis, 20% weight)* — Checks whether
  harder problems took longer to solve than easier ones. Human problem-solvers
  naturally slow down as difficulty increases.
- **Contest solve speed** *(14% weight)* — Total time to complete all problems.
  Accounts for practice volume — more problems solved = lower suspicion for
  fast completions.
- **Early contest performance** *(6% weight)* — Average rank in the first
  few contests attended.

**Consistency signals**
- **Ranking consistency** *(12% weight)* — Variance in contest rank over time.
  Performance naturally fluctuates for most people.
- **Submission entropy** *(4% weight)* — Day-of-week distribution of
  submissions. Very low entropy may indicate scripted activity.

**Practice vs results signals**
- **Percentile vs problems solved** *(16% weight)* — Compares contest
  percentile ranking against total problems solved. A very high percentile
  with very few solved problems is statistically unusual.
- **Profile depth** *(9% weight)* — Problems solved relative to contest
  rating bracket.
- **Rating velocity** *(11% weight)* — Size of single-contest rating jumps.

**Transparency signal**
- **Hidden submission graph** *(7% weight)* — Presence or absence of a
  visible submission history. Treated as a weak corroborating indicator only,
  not a primary signal. Many users hide their graphs for privacy reasons.

---

**Important:** This index reflects statistical patterns in public data.
It is not a verdict, not an accusation, and not evidence of any rule violation.
Legitimate high-performing users — including competitive programming
specialists — may receive elevated scores on certain signals.
Always apply your own judgement. This tool is a starting point for curiosity,
not a basis for action.
""")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔍 ContestLens")
st.markdown(
    "Enter a LeetCode username to explore statistical patterns "
    "in their contest history."
)

st.markdown("""
<div class="notice">
This tool analyses publicly available LeetCode data.
A high Anomaly Index reflects unusual statistical patterns — it does not
imply rule violations, cheating, or any wrongdoing.
Scores should never be the sole basis for any action or judgement against a user.
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
# Validation
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

def _chart_rating_trajectory(contests: list[dict]) -> go.Figure | None:
    attended = [c for c in contests if c.get("rating", 0) > 0]
    if not attended:
        return None

    df = pd.DataFrame(attended)
    df["date"] = pd.to_datetime(df["start_time"], unit="s", utc=True)

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

    if len(df) >= 3:
        df["delta"] = df["rating"].diff().fillna(0)
        top_spikes = df.nlargest(3, "delta")
        fig.add_trace(go.Scatter(
            x=top_spikes["date"],
            y=top_spikes["rating"],
            mode="markers",
            name="Large jump",
            marker=dict(size=12, color="#F4664A", symbol="star"),
            hovertemplate="<b>%{text}</b><br>Jump: +%{customdata:.0f}<extra></extra>",
            text=top_spikes["title"],
            customdata=top_spikes["delta"],
        ))

    fig.update_layout(
        title="Contest rating over time",
        xaxis_title="Date",
        yaxis_title="Rating",
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    return fig


def _chart_submission_heatmap(profile: dict) -> go.Figure | None:
    calendar: dict = profile.get("calendar", {})
    if not calendar:
        return None

    dow_counts = [0] * 7
    for ts_str, count in calendar.items():
        try:
            ts = int(ts_str)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            dow_counts[dt.weekday()] += int(count)
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


def _chart_contest_performance(contests: list[dict]) -> go.Figure | None:
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


def _chart_feature_scores(feature_scores: list[dict]) -> go.Figure | None:
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
        title="Signal breakdown",
        xaxis_title="Score (0–10)",
        height=300,
        margin=dict(t=40, b=40, l=200, r=60),
    )
    return fig


def _gauge_chart(score: float, severity: str) -> go.Figure:
    color_map = {
        "danger":  "#F5222D",
        "warning": "#FA8C16",
        "caution": "#1890FF",
        "normal":  "#52C41A",
    }
    color = color_map.get(severity, "#888888")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "/10", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,   2.5], "color": "#d4faad"},
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
    fig.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20))
    return fig

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_button(username: str) -> None:
    if st.button("🔄 Retry", key=f"retry_btn_{username}"):
        st.session_state[f"retry_{username}"] = True
        st.rerun()

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(username: str, api_key: str, deep: bool = False) -> None:
    username = username.strip()

    allowed, remaining = cache.check_rate_limit(SESSION_ID)
    if not allowed:
        st.error(
            f"Rate limit reached ({cache.RATE_LIMIT_MAX} lookups/hr). "
            f"Resets in ~{remaining // 60} min."
        )
        return

    st.markdown(f"### Results for `{username}`")

    age_mins      = cache.cache_age_minutes(username)
    cached_data   = cache.get_cached(username)
    force_refresh = st.session_state.pop(f"retry_{username}", False)

    if cached_data and not force_refresh:
        st.caption(f"📦 Cached data from {age_mins:.0f} min ago.")
        raw = cached_data
    else:
        with st.spinner("Fetching profile data… (5–10 s)"):
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

    profile  = raw.get("profile", {})
    contests = raw.get("contests", [])

    if not profile:
        st.error("Could not load profile data. The account may be private or unavailable.")
        return

    # Profile metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Contest rating",    f"{profile.get('contest_rating', 0):.0f}")
    with c2:
        st.metric("Contests attended", profile.get("attended_count", 0))
    with c3:
        st.metric("Problems solved",   profile.get("problems", {}).get("All", 0))
    with c4:
        top_pct = profile.get("top_percentage")
        st.metric("Top percentile",
                  f"{top_pct:.2f}%" if top_pct is not None else "N/A")

    st.markdown("---")

    # Feature + score computation
    with st.spinner("Computing signals…"):
        all_features = feat_module.compute_all_features(
            profile, contests,
            username=username,
            include_contest_timing=deep,
        )
        score_result = scorer_module.compute_anomaly_score(all_features)

    score    = score_result["score"]
    verdict  = score_result["verdict"]
    severity = score_result["severity"]
    quality  = score_result["data_quality"]

    # Record to distribution
    record_score(username, score, quality)
    pct_data = get_percentile(score)

    # Score display
    gauge_col, info_col = st.columns([1, 1])

    with gauge_col:
        st.plotly_chart(
            _gauge_chart(score, severity),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with info_col:
        sev_class = f"score-{severity}"
        emoji = {"danger": "🔴", "warning": "🟠", "caution": "🔵", "normal": "🟢"}.get(severity, "⚪")
        st.markdown(f"""
<div class="score-card {sev_class}">
    <h2 style="color:#1a1a1a; margin:0 0 8px 0">{emoji} {verdict}</h2>
    <p style="color:#1a1a1a; font-size:1.1rem; margin:4px 0">
        Anomaly Index: <strong>{score}/10</strong>
    </p>
    <p style="color:#555; font-size:0.85rem; margin:4px 0">
        Data quality: <strong>{quality}</strong> ·
        {score_result['active_features']} signals active
    </p>
</div>
""", unsafe_allow_html=True)

        # Percentile
        if pct_data.get("enough_data"):
            pct = pct_data["percentile"]
            st.caption(
                f"This score is higher than **{pct:.0f}%** of the "
                f"{pct_data['total_profiles']} profiles analysed so far."
            )
        else:
            remaining_for_pct = max(0, 20 - pct_data.get("total_profiles", 0))
            if remaining_for_pct > 0:
                st.caption(
                    f"Percentile comparison available after "
                    f"{remaining_for_pct} more lookups."
                )

        # Methodology version
        version = score_result.get("methodology", "v1.1-calibrated")
        st.caption(f"Scoring: `{version}` — heuristic weights, validated on 36 labeled profiles.")

        # Data quality warnings
        if quality == "insufficient":
            st.warning(
                "⚠️ Limited data — this profile has too few contests or "
                "missing fields for a reliable score."
            )
        elif quality == "partial":
            st.info("ℹ️ Some signals lacked sufficient data and were excluded.")

    st.markdown("---")

    # Explanation
    st.subheader("📝 Summary")
    with st.spinner("Generating summary…"):
        explanation = explainer_module.generate_explanation(
            username, score_result, all_features, api_key=api_key or None
        )
    st.markdown(explanation)

    st.markdown("---")

    # Signal breakdown
    st.subheader("📊 Signal breakdown")

    feat_chart = _chart_feature_scores(score_result["feature_scores"])
    if feat_chart:
        st.plotly_chart(feat_chart, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Per-signal detail", expanded=False):
        bonus = score_result.get("interaction_bonus", 0)
        if bonus > 0:
            st.info(
                f"⚡ Interaction bonus applied: +{bonus:.2f} — "
                "fast solve speed and shallow practice history "
                "are statistically unusual in combination."
            )
        for feat in all_features:
            conf      = feat.get("confidence", "low")
            score_val = feat.get("score", 0)
            icon      = "🔴" if score_val >= 7 else "🟠" if score_val >= 4 else "🟢"
            conf_label = {
                "high":   "✅ High confidence",
                "medium": "⚠️ Medium confidence",
                "low":    "❓ Insufficient data (excluded)",
            }.get(conf, conf)
            st.markdown(f"**{icon} {feat['label']}** — `{score_val}/10` · {conf_label}")
            st.caption(feat.get("description", "—"))

    st.markdown("---")

    # Contest history
    st.subheader("📈 Contest history")

    if contests:
        traj = _chart_rating_trajectory(contests)
        if traj:
            st.plotly_chart(traj, use_container_width=True, config={"displayModeBar": False})

        perf = _chart_contest_performance(contests)
        if perf:
            st.plotly_chart(perf, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Contest table", expanded=False):
            df_c = pd.DataFrame(contests)
            if not df_c.empty:
                df_c["date"] = pd.to_datetime(
                    df_c["start_time"], unit="s", utc=True
                ).dt.strftime("%Y-%m-%d")
                cols = [c for c in ["date", "title", "rating", "ranking",
                                     "problems_solved", "total_problems",
                                     "finish_seconds"] if c in df_c.columns]
                st.dataframe(df_c[cols], use_container_width=True, height=300)
    else:
        st.info("No rated contest history found for this profile.")

    st.markdown("---")

    # Submission patterns
    st.subheader("📅 Submission patterns")

    heat = _chart_submission_heatmap(profile)
    if heat:
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})

        entropy_feat = next(
            (f for f in all_features if f.get("label") == "Submission pattern entropy"),
            None,
        )
        if entropy_feat and entropy_feat.get("value") is not None:
            ent_val = entropy_feat["value"]
            max_ent = math.log2(7)
            st.caption(
                f"Day-of-week entropy: **{ent_val:.3f} bits** "
                f"(max {max_ent:.3f} bits). "
                "Lower values indicate activity concentrated on fewer days."
            )
    else:
        st.info("No submission calendar data available for this profile.")

    st.markdown("---")

    # Language breakdown
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

    # Footer
    st.markdown("""
<div class="notice">
ContestLens analyses publicly available data only.
A high Anomaly Index indicates statistically unusual patterns — it is not
evidence of any rule violation. If you have genuine concerns about a user's
contest behaviour, please report directly to
<a href="https://support.leetcode.com" target="_blank">LeetCode support</a>.
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
    st.markdown("### Try these example profiles")
    cols = st.columns(4)
    example_users = ["cpcs", "Ma_Lin", "votrubac", "PhoenixDD"]
    for i, user in enumerate(example_users):
        with cols[i]:
            if st.button(user, use_container_width=True):
                st.session_state["prefill"] = user
                st.rerun()

    if "prefill" in st.session_state:
        prefill = st.session_state.pop("prefill")
        run_analysis(prefill, api_key_input, deep=deep_analysis)