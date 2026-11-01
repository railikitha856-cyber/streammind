import html
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from twitch_chat import TwitchChatManager
from chat_insights import activity_spike, find_repeated_keywords, find_repeated_questions

from ai_engine import analyze_chat

AUTO_ANALYZE_MIN_MESSAGES = 10
AUTO_ANALYZE_COOLDOWN = timedelta(seconds=60)
AUTO_ANALYZE_BATCH_SIZE = 20
MANUAL_ANALYZE_BATCH_SIZE = 20
logger = logging.getLogger(__name__)


@st.cache_resource
def get_analysis_executor():
    """Share one worker so Groq requests never overlap across Streamlit reruns."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="streammind-groq")

# Live mode only displays messages and analysis received from Twitch/Groq.
st.set_page_config(
    page_title="StreamMind | Creator dashboard",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A small CSS layer gives Streamlit a dark, Twitch-inspired look.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg: #0d0b12; --panel: #17141f; --panel2: #1d1927; --line: #2d2838; --muted: #a49caf; --purple: #a970ff; --pink: #ed6ba7; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 75% -10%, #26163a 0%, var(--bg) 42%); color: #f5f2fa; }
    [data-testid="stSidebar"] { background: #121019; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.04em; }
    h1 { font-size: 2rem !important; margin-bottom: .2rem !important; }
    h2 { font-size: 1.18rem !important; }
    .brand { font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700; letter-spacing:-.05em; }
    .brand span { color:var(--purple); }
    .live-pill { display:inline-flex; align-items:center; gap:7px; background:#321a2a; color:#ff7ca7; border:1px solid #67304a; border-radius:99px; padding:5px 11px; font-size:11px; font-weight:700; letter-spacing:.1em; }
    .live-dot { width:7px; height:7px; border-radius:50%; background:#ff547c; box-shadow:0 0 10px #ff547c; }
    .eyebrow { color:var(--muted); text-transform:uppercase; font-size:10px; font-weight:700; letter-spacing:.16em; }
    .subtle { color:var(--muted); font-size:13px; }
    .panel { background:linear-gradient(145deg,#1a1722,#15121b); border:1px solid var(--line); border-radius:16px; padding:19px 20px; height:100%; }
    .metric-label { color:var(--muted); font-size:12px; }
    .metric-value { font-family:'Space Grotesk',sans-serif; font-size:27px; font-weight:700; margin-top:5px; }
    .metric-foot { color:#74d8b0; font-size:11px; margin-top:3px; }
    .topic { display:flex; justify-content:space-between; align-items:center; padding:11px 0; border-bottom:1px solid #292532; }
    .topic:last-child { border-bottom:0; padding-bottom:0; }
    .topic-name { font-weight:600; font-size:13px; }
    .topic-count { color:var(--muted); font-size:11px; }
    .rank { color:var(--purple); font-size:11px; font-weight:700; width:23px; }
    .chat-row { display:flex; gap:10px; padding:11px 0; border-bottom:1px solid #292532; align-items:flex-start; }
    .chat-row:last-child { border-bottom:0; }
    .chat-scroll { max-height:420px; overflow-y:auto; padding-right:7px; }
    .chat-scroll::-webkit-scrollbar { width:6px; }
    .chat-scroll::-webkit-scrollbar-thumb { background:#41384f; border-radius:99px; }
    .avatar { width:30px; height:30px; flex:none; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:13px; background:#34274b; }
    .chat-meta { font-size:11px; color:var(--muted); margin-bottom:3px; }
    .chat-user { color:#c99cff; font-weight:700; margin-right:6px; }
    .chat-text { font-size:12px; color:#e8e4ee; line-height:1.45; }
    .summary { color:#d8d1e2; font-size:13px; line-height:1.7; }
    .suggestion { background:#211a2c; border:1px solid #342847; border-radius:11px; padding:12px; margin-top:9px; font-size:12px; color:#e7dff1; line-height:1.5; }
    .suggestion b { color:#c99cff; margin-right:7px; }
    .profile { text-align:center; padding:12px 0 18px; }
    .profile-avatar { width:64px; height:64px; border-radius:22px; margin:0 auto 10px; display:flex; align-items:center; justify-content:center; font-size:29px; background:linear-gradient(135deg,#7a3ee6,#de62ae); box-shadow:0 8px 28px #8e42b744; }
    .profile-name { font-weight:700; font-size:15px; }
    .profile-handle { color:var(--muted); font-size:11px; margin-top:2px; }
    .sidebar-live { background:#1d1927; border:1px solid var(--line); border-radius:12px; padding:12px; margin:9px 0 23px; }
    .sidebar-label { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.12em; }
    .sidebar-value { font-size:12px; font-weight:600; margin-top:5px; }
    div[data-testid="stMetric"] { background:#17141f; border:1px solid var(--line); border-radius:14px; padding:15px 16px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; }
    .block-container { padding-top:1.7rem; padding-bottom:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Example chat is available only when Demo Mode is explicitly enabled.
demo_chat_data = [
    {"time": "10:42:18", "user": "pixelpanda", "avatar": "🐼", "message": "That last play was actually unreal 🔥"},
    {"time": "10:42:12", "user": "luna.exe", "avatar": "🌙", "message": "The new map looks so good, explore the left side!"},
    {"time": "10:42:05", "user": "ToastWizard", "avatar": "🍞", "message": "W stream as always 💜"},
    {"time": "10:41:57", "user": "mintyfresh", "avatar": "🍃", "message": "Wait is that the secret room behind the waterfall?"},
    {"time": "10:41:49", "user": "criticalhit", "avatar": "⚡", "message": "boss fight incoming… chat predictions?"},
    {"time": "10:41:40", "user": "cloudberry", "avatar": "☁️", "message": "I love this soundtrack so much"},
    {"time": "10:41:31", "user": "not_a_bot", "avatar": "🤖", "message": "left side! there’s a chest near the bridge"},
    {"time": "10:41:22", "user": "StarryNights", "avatar": "⭐", "message": "You got this! We believe in you 🙌"},
]

# Keep the live feed, pending AI batch, and analysis history between reruns.
state_defaults = {
    "chat_analysis": None,
    "twitch_manager": None,
    "recent_chat": [],
    "pending_chat": [],
    "activity_events": [],
    "last_activity_alert_at": None,
    "activity_alert": "",
    "sentiment_history": [],
    "last_analysis_at": None,
    "last_analysis_message_count": 0,
    "total_analyzed_messages": 0,
    "last_auto_analysis_at": None,
    "last_auto_attempt_at": None,
    "auto_analysis_error": "",
    "active_channel": "",
    "received_message_total": 0,
    "chat_analysis_channel": "",
    "chat_analysis_demo": False,
    "analysis_job": None,
    "analysis_notice": "",
    "analysis_error": "",
    "last_analysis_duration": None,
}
for key, default_value in state_defaults.items():
    if key not in st.session_state:
        if key == "twitch_manager":
            st.session_state[key] = TwitchChatManager()
        elif isinstance(default_value, list):
            st.session_state[key] = list(default_value)
        else:
            st.session_state[key] = default_value

twitch_manager = st.session_state.twitch_manager
now_utc = datetime.now(timezone.utc)

# Drain the thread-safe queue only on Streamlit's main thread.
def add_incoming_messages(messages, received_at):
    for item in messages:
        item.setdefault("received_at", received_at.isoformat())
        try:
            event_time = datetime.fromisoformat(item["received_at"])
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            event_time = received_at
        st.session_state.activity_events.append(event_time)
    st.session_state.recent_chat.extend(messages)
    st.session_state.recent_chat = st.session_state.recent_chat[-100:]
    st.session_state.pending_chat.extend(messages)
    st.session_state.pending_chat = st.session_state.pending_chat[-500:]
    st.session_state.received_message_total += len(messages)


incoming_messages = twitch_manager.drain_messages()
add_incoming_messages(incoming_messages, now_utc)
st.session_state.activity_events = [
    event for event in st.session_state.activity_events
    if now_utc - event <= timedelta(minutes=6)
]
live_channel = getattr(twitch_manager, "_target_channel", "")

# These checks are local and free: no Groq request is used to spot repeats/spikes.
repeated_questions = find_repeated_questions(st.session_state.recent_chat)
repeated_keywords = find_repeated_keywords(st.session_state.recent_chat)
spike_detected, minute_message_count, baseline_per_minute = activity_spike(
    st.session_state.activity_events, now_utc
)
new_activity_alert = False
if spike_detected and (
    st.session_state.last_activity_alert_at is None
    or now_utc - st.session_state.last_activity_alert_at >= timedelta(seconds=90)
):
    st.session_state.last_activity_alert_at = now_utc
    st.session_state.activity_alert = (
        f"Chat is moving fast: {minute_message_count} messages in the last minute "
        f"(usual pace: {baseline_per_minute}/minute)."
    )
    new_activity_alert = True
if (
    st.session_state.last_activity_alert_at
    and now_utc - st.session_state.last_activity_alert_at > timedelta(minutes=3)
):
    st.session_state.activity_alert = ""

# The sidebar holds a lightweight streamer profile and navigation.
with st.sidebar:
    st.markdown('<div class="brand">stream<span>mind</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile"><div class="profile-avatar">🎮</div><div class="profile-name">{html.escape("@" + live_channel if live_channel else "Twitch streamer")}</div><div class="profile-handle">Connected channel</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-live"><div class="sidebar-label">Stream status</div><div class="sidebar-value">{"🟢 &nbsp;" + html.escape(twitch_manager.status) if twitch_manager.running else "⚪ &nbsp;Disconnected"}</div></div>', unsafe_allow_html=True)
    demo_mode = st.toggle("Demo Mode", key="demo_mode", help="Show sample chat messages instead of live Twitch data.")
    st.markdown('<div class="eyebrow" style="margin:0 0 8px">Workspace</div>', unsafe_allow_html=True)
    section = st.radio("Dashboard navigation", ["Overview", "Chat insights", "Stream history"], label_visibility="collapsed")
    auto_analyze = st.toggle("Auto Analyze", key="auto_analyze", help="Analyze a small batch of new chat at most once per minute.")
    st.caption(f"Auto mode waits for {AUTO_ANALYZE_MIN_MESSAGES} new messages and a 60-second cooldown.")
    st.markdown('<div class="eyebrow" style="margin:18px 0 8px">Twitch chat</div>', unsafe_allow_html=True)
    twitch_channel = st.text_input("Twitch channel name", placeholder="e.g. twitchdev", key="twitch_channel_input", label_visibility="collapsed")
    connect_col, disconnect_col = st.columns(2)
    with connect_col:
        if st.button(
            "Connect", use_container_width=True,
            disabled=twitch_manager.running or st.session_state.analysis_job is not None,
        ):
            requested_channel = (twitch_channel or "").strip().lstrip("#").lower()
            old_channel = st.session_state.active_channel or live_channel.lower()
            if requested_channel != old_channel:
                st.session_state.recent_chat.clear()
                st.session_state.pending_chat.clear()
                st.session_state.activity_events.clear()
                st.session_state.sentiment_history.clear()
                st.session_state.chat_analysis = None
                st.session_state.last_analysis_at = None
                st.session_state.last_analysis_message_count = 0
                st.session_state.total_analyzed_messages = 0
                st.session_state.received_message_total = 0
                st.session_state.activity_alert = ""
                st.session_state.last_activity_alert_at = None
                st.session_state.last_auto_analysis_at = None
                st.session_state.last_auto_attempt_at = None
                st.session_state.analysis_job = None
                st.session_state.analysis_notice = ""
                st.session_state.analysis_error = ""
                st.session_state.last_analysis_duration = None
                st.session_state.active_channel = requested_channel
                st.session_state.chat_analysis_channel = ""
                st.session_state.twitch_manager.drain_messages()
            connect_error = twitch_manager.start(twitch_channel)
            if connect_error:
                st.error(connect_error)
    with disconnect_col:
        if st.button("Disconnect", use_container_width=True, disabled=not twitch_manager.running):
            twitch_manager.stop()
    if not (
        twitch_manager.status == "Disconnected"
        or twitch_manager.status.startswith(("Starting Twitch", "Connected to @", "Channel @"))
        or "Authorize Twitch chat access" in twitch_manager.status
    ):
        st.error(twitch_manager.status)
    else:
        st.caption(twitch_manager.status)
    if "Authorize Twitch chat access" in twitch_manager.status:
        st.link_button(
            "Authorize Twitch chat reader",
            "http://localhost:4343/oauth?scopes=user:read:chat&force_verify=true",
            use_container_width=True,
        )
    st.markdown('<div style="position:fixed;bottom:20px;color:#777180;font-size:11px">✦ &nbsp;StreamMind AI Copilot <span style="float:right">v0.1</span></div>', unsafe_allow_html=True)

# Keep all local detections scoped to the selected source.
chat_data = demo_chat_data if demo_mode else st.session_state.recent_chat
source_messages = chat_data
repeated_questions = find_repeated_questions(source_messages)
repeated_keywords = find_repeated_keywords(source_messages)
live_connected = twitch_manager.running and twitch_manager.status.startswith("Connected to @")

# Header and live state.
left, right = st.columns([5, 1])
with left:
    st.markdown('<div class="eyebrow">Creator dashboard &nbsp; / &nbsp; Live overview</div>', unsafe_allow_html=True)
    st.title(f"Live overview · @{live_channel}" if live_channel else "Live overview")
    st.markdown('<div class="subtle">Here’s what’s happening in your stream right now.</div>', unsafe_allow_html=True)
with right:
    if demo_mode:
        st.markdown('<div style="text-align:right;padding-top:15px"><span class="live-pill">DEMO MODE</span></div>', unsafe_allow_html=True)
    elif live_connected:
        st.markdown('<div style="text-align:right;padding-top:15px"><span class="live-pill"><span class="live-dot"></span> LIVE DATA</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right;padding-top:15px"><span class="live-pill">WAITING FOR LIVE DATA</span></div>', unsafe_allow_html=True)
st.write("")

# AI calls run on one worker. Streamlit polls the Future while the Twitch listener
# continues independently, then commits results to session state on the main thread.
def request_analysis(messages, questions):
    started = time.perf_counter()
    result = analyze_chat(messages, repeated_questions=questions)
    # Prefer measured HTTP request duration; this also includes a fallback call
    # when JSON mode needed recovery. The wrapper time is a safe test/custom-engine fallback.
    duration = float(result.get("api_response_seconds", time.perf_counter() - started))
    return result, duration


def commit_analysis(result, messages, channel, is_demo, duration):
    analyzed_at = datetime.now().astimezone()
    analyzed_count = int(result.get("messages_analyzed", len(messages)))
    st.session_state.chat_analysis = result
    st.session_state.chat_analysis_channel = channel.lower()
    st.session_state.chat_analysis_demo = is_demo
    st.session_state.last_analysis_at = analyzed_at
    st.session_state.last_analysis_message_count = analyzed_count
    st.session_state.last_analysis_duration = duration
    st.session_state.total_analyzed_messages += analyzed_count
    counts = result["sentiment_counts"]
    st.session_state.sentiment_history.append({
        "time": analyzed_at,
        "positive": counts["positive"],
        "neutral": counts["neutral"],
        "negative": counts["negative"],
        "messages": analyzed_count,
        "channel": channel.lower(),
        "demo": is_demo,
    })
    st.session_state.sentiment_history = st.session_state.sentiment_history[-30:]


def submit_analysis(messages, kind, questions):
    if st.session_state.analysis_job is not None:
        return False
    batch = list(messages)
    future = get_analysis_executor().submit(request_analysis, batch, list(questions))
    st.session_state.analysis_job = {
        "future": future,
        "batch": batch,
        "kind": kind,
        "channel": live_channel,
        "demo": bool(demo_mode),
        "started": time.perf_counter(),
    }
    st.session_state.analysis_error = ""
    st.session_state.analysis_notice = ""
    return True


analysis_job = st.session_state.analysis_job
if analysis_job is not None and analysis_job["future"].done():
    try:
        result, duration = analysis_job["future"].result()
        commit_analysis(
            result, analysis_job["batch"], analysis_job["channel"],
            analysis_job["demo"], duration,
        )
        completed_ids = {id(item) for item in analysis_job["batch"]}
        st.session_state.pending_chat = [
            item for item in st.session_state.pending_chat if id(item) not in completed_ids
        ]
        st.session_state.analysis_notice = (
            f"Analysis complete: {st.session_state.last_analysis_message_count} messages "
            f"in {duration:.2f} seconds."
        )
        st.session_state.analysis_error = ""
        st.session_state.auto_analysis_error = ""
    except Exception as error:
        logger.exception("StreamMind background analysis failed")
        elapsed = time.perf_counter() - analysis_job["started"]
        st.session_state.analysis_error = (
            f"Analysis failed after {elapsed:.1f} seconds ({type(error).__name__}): {error}. "
            "The messages remain queued; correct the problem and retry."
        )
        st.session_state.analysis_notice = ""
    finally:
        st.session_state.analysis_job = None

analysis_job_running = st.session_state.analysis_job is not None

# Manual runs use only the latest 20 actual chat lines and do not depend on Auto Analyze.
button_col, _ = st.columns([1, 5])
manual_analysis_clicked = False
with button_col:
    if st.button(
        "✦  Analyze Chat", type="primary", use_container_width=True,
        disabled=analysis_job_running,
    ):
        manual_analysis_clicked = True
        incoming_messages = twitch_manager.drain_messages()
        add_incoming_messages(incoming_messages, datetime.now(timezone.utc))
        source_messages = demo_chat_data if demo_mode else st.session_state.recent_chat
        analysis_messages = list(source_messages[-MANUAL_ANALYZE_BATCH_SIZE:])
        if not analysis_messages:
            st.session_state.analysis_error = "No chat messages are available yet. Wait for live messages and try again."
        elif not submit_analysis(
            analysis_messages, "manual", find_repeated_questions(analysis_messages)
        ):
            st.session_state.analysis_error = "An analysis is already running. Please wait for it to finish."

# Auto mode keeps the existing 10-message threshold and 60-second cooldown.
if (
    auto_analyze and not demo_mode and live_connected and not manual_analysis_clicked
    and not analysis_job_running
    and len(st.session_state.pending_chat) >= AUTO_ANALYZE_MIN_MESSAGES
):
    last_attempt = st.session_state.last_auto_attempt_at
    cooldown_complete = last_attempt is None or now_utc - last_attempt >= AUTO_ANALYZE_COOLDOWN
    if cooldown_complete:
        auto_batch = list(st.session_state.pending_chat[:AUTO_ANALYZE_BATCH_SIZE])
        if submit_analysis(auto_batch, "auto", find_repeated_questions(auto_batch)):
            st.session_state.last_auto_attempt_at = now_utc
elif auto_analyze and demo_mode:
    st.info("Auto Analyze is paused in Demo Mode. Turn Demo Mode off to analyze live Twitch messages.")
elif auto_analyze and not live_connected:
    st.info("Auto Analyze is waiting for an authorized Twitch chat connection.")
elif auto_analyze and len(st.session_state.pending_chat) < AUTO_ANALYZE_MIN_MESSAGES:
    st.caption(f"Auto Analyze is waiting for {AUTO_ANALYZE_MIN_MESSAGES - len(st.session_state.pending_chat)} more new messages.")

# Poll while either Twitch is active or a background AI request is pending.
if twitch_manager.running or st.session_state.analysis_job is not None:
    st_autorefresh(interval=1000, key="twitch_chat_refresh")

if st.session_state.analysis_job is not None:
    job_kind = st.session_state.analysis_job["kind"]
    with st.spinner(f"{job_kind.title()} analysis is running on the latest 20 messages…"):
        st.caption("Twitch chat keeps receiving messages while Groq processes this batch.")
if st.session_state.analysis_notice:
    st.success(st.session_state.analysis_notice)
if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)

if new_activity_alert:
    st.toast(st.session_state.activity_alert, icon="⚡")
if st.session_state.auto_analysis_error:
    st.error(st.session_state.auto_analysis_error)
analysis_matches_source = (
    st.session_state.chat_analysis is not None
    and st.session_state.chat_analysis_demo == bool(demo_mode)
    and (demo_mode or st.session_state.chat_analysis_channel == live_channel.lower())
)
analysis = st.session_state.chat_analysis if analysis_matches_source else None
if st.session_state.last_analysis_at and analysis:
    response_time = st.session_state.last_analysis_duration
    response_time_label = (
        f"Groq response {response_time:.2f}s"
        if isinstance(response_time, (int, float)) else "Groq response time unavailable"
    )
    st.caption(
        f"Last analyzed: {st.session_state.last_analysis_at.strftime('%b %d, %Y %I:%M:%S %p')} · "
        f"{st.session_state.last_analysis_message_count} messages in that batch · "
        f"{st.session_state.total_analyzed_messages} total analyzed · "
        f"{response_time_label}"
    )
else:
    st.caption("Last analyzed: not yet analyzed for this channel and data mode")
if auto_analyze and twitch_manager.running:
    st.caption(f"Auto Analyze is on · {len(st.session_state.pending_chat)} new messages queued")

# Key stream numbers at a glance.
metric_cols = st.columns(4)
metric_message_total = len(chat_data) if demo_mode else st.session_state.received_message_total
unique_chatters = len({str(item.get("user", "")).casefold() for item in chat_data if item.get("user")})
messages_per_minute = sum(now_utc - timedelta(minutes=1) <= event <= now_utc for event in st.session_state.activity_events)
analyzed_count = st.session_state.last_analysis_message_count if analysis else 0
metrics = [("MESSAGES RECEIVED", f"{metric_message_total:,}", "Demo sample" if demo_mode else "Since this channel was connected"), ("UNIQUE CHATTERS", f"{unique_chatters:,}", "In the recent message buffer"), ("MESSAGES / MIN", f"{messages_per_minute:,}", "Observed in the last minute"), ("MESSAGES ANALYZED", f"{analyzed_count:,}", "Latest analysis batch" if analysis else "Waiting for analysis")]
for col, (label, value, foot) in zip(metric_cols, metrics):
    with col:
        st.markdown(f'<div class="panel" style="padding:15px 17px"><div class="eyebrow">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">↗ &nbsp;{foot}</div></div>', unsafe_allow_html=True)
st.write("")

# Put the AI insight cards before the live feed so they stay easy to find.
st.write("")
trends_col, summary_col, suggestions_col = st.columns([1, 1, 1])
with trends_col:
    st.markdown('<div class="panel"><h2 style="margin:0">Trending topics</h2><div class="subtle" style="margin:5px 0 10px">What chat can’t stop talking about</div>', unsafe_allow_html=True)
    if analysis:
        topics = [(html.escape(str(item.get("topic", "Chat topic"))), f"{item.get('count', 0)} messages") for item in analysis["topics"]]
    else:
        topics = []
    if not topics:
        st.markdown('<div class="subtle">Waiting for chat analysis...</div>', unsafe_allow_html=True)
    for index, (topic, count) in enumerate(topics, 1):
        st.markdown(f'<div class="topic"><div style="display:flex;align-items:center"><span class="rank">0{index}</span><span class="topic-name">{topic}</span></div><span class="topic-count">{count}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with summary_col:
    summary_text = html.escape(analysis["summary"]) if analysis else "Waiting for chat analysis..."
    summary_status = "AI analysis" if analysis else ("Demo Mode · analyze sample chat" if demo_mode else "Live Twitch data · analyze chat for insights")
    st.markdown(f'<div class="panel"><div class="eyebrow">✦ &nbsp;AI copilot</div><h2 style="margin:8px 0 5px">Chat summary</h2><div class="subtle" style="margin-bottom:12px">A quick read on the conversation</div><div class="summary">{summary_text}</div><div style="margin-top:15px;color:#c99cff;font-size:11px">✦ &nbsp;{summary_status}</div></div>', unsafe_allow_html=True)
with suggestions_col:
    if analysis:
        suggestion_cards = "".join(
            f'<div class="suggestion"><b>{index:02d}</b> {html.escape(str(point))}</div>'
            for index, point in enumerate(analysis["talking_points"], 1)
        )
    else:
        suggestion_cards = '<div class="subtle">Waiting for chat analysis...</div>'
    st.markdown(f'<div class="panel"><div class="eyebrow">✦ &nbsp;Try saying</div><h2 style="margin:8px 0 5px">Talking points</h2><div class="subtle" style="margin-bottom:7px">Keep the conversation flowing</div>{suggestion_cards}</div>', unsafe_allow_html=True)

# Surface repeated questions and locally detected alerts beside the AI insight cards.
questions_col, alerts_col = st.columns([1.1, 1])
with questions_col:
    st.markdown('<div class="panel"><h2 style="margin:0">Repeated questions</h2><div class="subtle" style="margin:5px 0 10px">Questions viewers are asking more than once</div>', unsafe_allow_html=True)
    answer_items = analysis.get("question_answers", []) if analysis else []
    if repeated_questions:
        for index, item in enumerate(repeated_questions[:3], 1):
            answer = next(
                (entry.get("answer", "") for entry in answer_items
                 if entry.get("question", "").strip().casefold() == item["question"].strip().casefold()),
                "Run Analyze Chat or enable Auto Analyze for a suggested answer.",
            )
            st.markdown(
                f'<div class="suggestion"><b>{index:02d}</b> {html.escape(item["question"])} '
                f'<span class="topic-count">· {item["count"]} times</span><br><span class="summary">Suggested answer: {html.escape(answer)}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="subtle">No repeated questions detected yet.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with alerts_col:
    st.markdown('<div class="panel"><h2 style="margin:0">Smart chat alerts</h2><div class="subtle" style="margin:5px 0 10px">Local activity signals from recent chat</div>', unsafe_allow_html=True)
    if st.session_state.activity_alert and not demo_mode:
        st.warning(st.session_state.activity_alert)
    else:
        st.markdown('<div class="subtle">No sudden activity spikes detected.</div>', unsafe_allow_html=True)
    if repeated_keywords:
        keyword = repeated_keywords[0]
        st.markdown(
            f'<div class="suggestion"><b>↻</b> Repeated keyword: '
            f'<strong>{html.escape(keyword["keyword"])}</strong> in {keyword["count"]} messages.</div>',
            unsafe_allow_html=True,
        )
    if repeated_questions:
        st.markdown(
            f'<div class="suggestion"><b>?</b> Most repeated question appears {repeated_questions[0]["count"]} times.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
# The chat history scrolls inside its card instead of growing the full page.
chat_col, sentiment_col = st.columns([1.05, 1])
with chat_col:
    chat_rows = "".join(
        f'<div class="chat-row"><div class="avatar">{item.get("avatar", "💬")}</div><div><div class="chat-meta"><span class="chat-user">{html.escape(str(item["user"]))}</span>{html.escape(str(item["time"]))}</div><div class="chat-text">{html.escape(str(item["message"]))}</div></div></div>'
        for item in reversed(chat_data)
    )
    st.markdown(
        f'<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center"><h2 style="margin:0">Live chat</h2><span class="subtle">● &nbsp;{len(chat_data):,} recent messages</span></div><div class="subtle" style="margin:5px 0 8px">The latest from your community · scroll to see more</div><div class="chat-scroll">{chat_rows}</div></div>',
        unsafe_allow_html=True,
    )

with sentiment_col:
    st.markdown('<div class="panel"><h2 style="margin:0">Chat sentiment</h2><div class="subtle" style="margin:5px 0">Sentiment from the latest AI analysis</div>', unsafe_allow_html=True)
    if analysis:
        sentiment_counts = analysis["sentiment_counts"]
        sentiment_values = [sentiment_counts["positive"], sentiment_counts["neutral"], sentiment_counts["negative"]]
    else:
        sentiment_values = [0, 0, 0]
    sentiment_df = pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"], "Messages": sentiment_values})
    sentiment_total = max(sum(sentiment_values), 1)
    positive_percent = round(sentiment_values[0] / sentiment_total * 100) if sum(sentiment_values) else 0
    fig = px.pie(sentiment_df, values="Messages", names="Sentiment", hole=0.72, color="Sentiment", color_discrete_map={"Positive": "#a970ff", "Neutral": "#535064", "Negative": "#ed6ba7"})
    fig.update_traces(textinfo="none", hovertemplate="%{label}: %{value} messages<extra></extra>", marker=dict(line=dict(color="#17141f", width=4)))
    fig.update_layout(showlegend=False, height=220, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", annotations=[dict(text=f"{positive_percent}%<br><span style='font-size:11px;color:#a49caf'>positive</span>", x=0.5, y=0.5, font=dict(size=25, color="#f5f2fa", family="Space Grotesk"), showarrow=False)])
    chart_col, legend_col = st.columns([1.1, 1])
    with chart_col:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with legend_col:
        st.markdown(f'<div style="padding-top:32px"><div class="topic"><span class="topic-name">🟣 &nbsp;Positive</span><b>{sentiment_values[0]}</b></div><div class="topic"><span class="topic-name">⚪ &nbsp;Neutral</span><b>{sentiment_values[1]}</b></div><div class="topic"><span class="topic-name">🔴 &nbsp;Negative</span><b>{sentiment_values[2]}</b></div><div class="subtle" style="margin-top:12px;font-size:10px">Message counts from recent chat</div></div>', unsafe_allow_html=True)
    visible_sentiment_history = [
        point for point in st.session_state.sentiment_history
        if point.get("demo", False) == bool(demo_mode)
        and (demo_mode or point.get("channel", "") == live_channel.lower())
    ]
    if visible_sentiment_history:
        trend_rows = [
            {"Time": point["time"], "Sentiment": label.title(), "Messages": point[label]}
            for point in visible_sentiment_history
            for label in ("positive", "neutral", "negative")
        ]
        trend_fig = px.line(
            pd.DataFrame(trend_rows), x="Time", y="Messages", color="Sentiment",
            markers=True,
            color_discrete_map={"Positive": "#a970ff", "Neutral": "#777184", "Negative": "#ed6ba7"},
        )
        trend_fig.update_layout(
            title="Sentiment over time", height=230,
            margin=dict(t=38, b=10, l=5, r=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d8d1e2", legend_title_text="",
        )
        trend_fig.update_xaxes(showgrid=False, title="")
        trend_fig.update_yaxes(gridcolor="#2d2838", title="Messages per batch")
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Sentiment trends appear after the first AI analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div style="text-align:center;color:#625c6c;font-size:10px;margin-top:24px">STREAMMIND &nbsp;·&nbsp; {"DEMO MODE" if demo_mode else "LIVE DATA" if live_connected else "WAITING FOR LIVE DATA"}</div>', unsafe_allow_html=True)

