import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import requests
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import subprocess
import sys
import re
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Call QA Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-high { border-left-color: #ff4444 !important; background-color: #fff5f5; }
    .alert-medium { border-left-color: #ffaa00 !important; background-color: #fffaf0; }
    .alert-low { border-left-color: #44ff44 !important; background-color: #f0fff4; }

    .stTab [role="tabpanel"] {
        padding-top: 1rem;
    }

    .call-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .score-excellent { color: #28a745; background-color: #d4edda; }
    .score-good { color: #17a2b8; background-color: #d1ecf1; }
    .score-average { color: #ffc107; background-color: #fff3cd; }
    .score-poor { color: #dc3545; background-color: #f8d7da; }

    .coaching-note {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }

    .issue-flag {
        background-color: #fff5f5;
        border-left: 4px solid #ff4444;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }

    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Database setup
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('call_qa_database.db')
    cursor = conn.cursor()

    # Calls table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT,
            agent_id TEXT,
            agent_name TEXT,
            customer_phone TEXT,
            lead_id TEXT,
            call_date TIMESTAMP,
            file_path TEXT,
            file_size INTEGER,
            duration_seconds INTEGER,
            transcript TEXT,
            sentiment_score REAL,
            clarity_score REAL,
            professionalism_score REAL,
            script_adherence_score REAL,
            overall_score REAL,
            silence_percentage REAL,
            interruptions_count INTEGER,
            keywords_found TEXT,
            issues_flagged TEXT,
            coaching_notes TEXT,
            processed_date TIMESTAMP,
            lead_source TEXT,
            call_direction TEXT,
            state TEXT,
            vendor TEXT,
            call_outcome TEXT,
            convoso_api_id TEXT
        )
    """)

    # Scoring weights table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scoring_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT UNIQUE,
            weight REAL,
            description TEXT,
            updated_date TIMESTAMP
        )
    """)

    # Agent information table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            agent_name TEXT,
            hire_date DATE,
            team TEXT,
            status TEXT,
            updated_date TIMESTAMP
        )
    """)

    # Insert default weights if none exist
    default_weights = [
        ('sentiment', 0.25, 'Overall call sentiment and tone'),
        ('clarity', 0.20, 'Audio clarity and communication quality'),
        ('professionalism', 0.20, 'Agent professionalism and courtesy'),
        ('script_adherence', 0.15, 'Following prescribed scripts and procedures'),
        ('silence_handling', 0.10, 'Management of dead air and pauses'),
        ('interruptions', 0.10, 'Appropriate conversation flow management')
    ]

    for category, weight, description in default_weights:
        cursor.execute("""
            INSERT OR IGNORE INTO scoring_weights (category, weight, description, updated_date)
            VALUES (?, ?, ?, ?)
        """, (category, weight, description, datetime.now()))

    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

def load_calls_data():
    """Load calls data from database"""
    try:
        conn = sqlite3.connect('call_qa_database.db')
        df = pd.read_sql_query("""
            SELECT * FROM calls 
            ORDER BY call_date DESC
        """, conn)
        conn.close()

        if not df.empty:
            df['call_date'] = pd.to_datetime(df['call_date'])
            df['processed_date'] = pd.to_datetime(df['processed_date'])

        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def get_score_color_class(score):
    """Get CSS class for score color"""
    if score >= 90:
        return "score-excellent"
    elif score >= 75:
        return "score-good"
    elif score >= 60:
        return "score-average"
    else:
        return "score-poor"

def format_duration(seconds):
    """Format duration in seconds to readable format"""
    if pd.isna(seconds) or seconds == 0:
        return "0:00"

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def main():
    """Main application"""

    # Header
    st.title("🎯 Call QA Intelligence Platform")
    st.markdown("*Motor Vehicle Accident Referral Service Quality Assurance*")

    # Load data
    calls_df = load_calls_data()

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")

    # Quick stats in sidebar
    if not calls_df.empty:
        st.sidebar.markdown("### 📈 Quick Stats")
        total_calls = len(calls_df)
        avg_score = calls_df['overall_score'].mean()
        flagged_calls = len(calls_df[calls_df['overall_score'] < 70])

        st.sidebar.metric("Total Calls", total_calls)
        st.sidebar.metric("Avg Score", f"{avg_score:.1f}")
        st.sidebar.metric("Flagged Calls", flagged_calls)

        # Today's stats
        today = datetime.now().date()
        today_calls = calls_df[calls_df['call_date'].dt.date == today]
        st.sidebar.metric("Today's Calls", len(today_calls))

        if len(today_calls) > 0:
            st.sidebar.metric("Today's Avg Score", f"{today_calls['overall_score'].mean():.1f}")

    # Page selection
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard",
        "🔍 Call Analysis", 
        "👥 Agent Performance",
        "📋 Reports & Export",
        "⚙️ Settings",
        "📤 Upload Test Call",
        "🔧 System Status"
    ])

    # Route to appropriate page
    if page == "📊 Dashboard":
        show_dashboard(calls_df)
    elif page == "🔍 Call Analysis":
        show_call_analysis(calls_df)
    elif page == "👥 Agent Performance":
        show_agent_performance(calls_df)
    elif page == "📋 Reports & Export":
        show_reports_export(calls_df)
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "📤 Upload Test Call":
        show_upload_test()
    elif page == "🔧 System Status":
        show_system_status(calls_df)

def show_dashboard(calls_df):
    """Show main dashboard"""

    st.header("📊 Dashboard Overview")

    if calls_df.empty:
        st.info("👋 Welcome! No calls processed yet. Upload a test call or wait for FTP transfers to begin.")
        st.markdown("""
        **Getting Started:**
        1. Upload a test call using the "📤 Upload Test Call" page
        2. Configure your FTP settings in "⚙️ Settings"  
        3. View processed calls and analytics here
        """)
        return

    # Time period selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        period = st.selectbox("Time Period", ["Today", "Last 7 days", "Last 30 days", "All time"])

    # Filter data based on period
    filtered_df = calls_df.copy()
    if period == "Today":
        filtered_df = calls_df[calls_df['call_date'].dt.date == datetime.now().date()]
    elif period == "Last 7 days":
        week_ago = datetime.now() - timedelta(days=7)
        filtered_df = calls_df[calls_df['call_date'] >= week_ago]
    elif period == "Last 30 days":
        month_ago = datetime.now() - timedelta(days=30)
        filtered_df = calls_df[calls_df['call_date'] >= month_ago]

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_calls = len(filtered_df)
        st.metric("Total Calls", total_calls)

    with col2:
        if not filtered_df.empty:
            avg_score = filtered_df['overall_score'].mean()
            prev_avg = calls_df['overall_score'].mean() if len(calls_df) > len(filtered_df) else avg_score
            delta = avg_score - prev_avg
            st.metric("Average Score", f"{avg_score:.1f}", delta=f"{delta:+.1f}")
        else:
            st.metric("Average Score", "0.0")

    with col3:
        if not filtered_df.empty:
            flagged_calls = len(filtered_df[filtered_df['overall_score'] < 70])
            flagged_pct = (flagged_calls / total_calls * 100) if total_calls > 0 else 0
            st.metric("Flagged Calls", flagged_calls, delta=f"{flagged_pct:.1f}%")
        else:
            st.metric("Flagged Calls", "0")

    with col4:
        if not filtered_df.empty:
            total_duration = filtered_df['duration_seconds'].sum()
            avg_duration = filtered_df['duration_seconds'].mean()
            st.metric("Total Talk Time", format_duration(total_duration), delta=format_duration(avg_duration))
        else:
            st.metric("Total Talk Time", "0:00")

    with col5:
        unique_agents = filtered_df['agent_id'].nunique() if not filtered_df.empty else 0
        st.metric("Active Agents", unique_agents)

    if filtered_df.empty:
        st.info(f"No calls found for {period.lower()}")
        return

    # Charts section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Score Distribution")
        fig_hist = px.histogram(
            filtered_df, 
            x='overall_score', 
            bins=20,
            title="Distribution of Call Scores",
            labels={'overall_score': 'Overall Score', 'count': 'Number of Calls'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("📞 Daily Call Volume")
        daily_stats = filtered_df.groupby(filtered_df['call_date'].dt.date).agg({
            'id': 'count',
            'overall_score': 'mean'
        }).reset_index()
        daily_stats.columns = ['Date', 'Calls', 'Avg Score']

        fig_line = px.line(
            daily_stats, 
            x='Date', 
            y='Calls',
            title="Daily Call Volume",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Score breakdown by category
    st.subheader("📊 Score Breakdown by Category")
    col1, col2 = st.columns(2)

    with col1:
        # Average scores by category
        category_scores = {
            'Sentiment': filtered_df['sentiment_score'].mean(),
            'Clarity': filtered_df['clarity_score'].mean(), 
            'Professionalism': filtered_df['professionalism_score'].mean(),
            'Script Adherence': filtered_df['script_adherence_score'].mean()
        }

        fig_bar = px.bar(
            x=list(category_scores.keys()),
            y=list(category_scores.values()),
            title="Average Scores by Category",
            labels={'x': 'Category', 'y': 'Average Score'},
            color=list(category_scores.values()),
            color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Top issues
        st.subheader("⚠️ Common Issues")
        all_issues = []
        for issues_str in filtered_df['issues_flagged'].dropna():
            try:
                issues = json.loads(issues_str) if issues_str else []
                all_issues.extend(issues)
            except:
                continue

        if all_issues:
            issue_counts = pd.Series(all_issues).value_counts().head(5)
            for issue, count in issue_counts.items():
                st.markdown(f"<div class='issue-flag'>⚠️ {issue} ({count} calls)</div>", 
                           unsafe_allow_html=True)
        else:
            st.success("✅ No common issues identified")

    # Recent calls table
    st.subheader("🕐 Recent Calls")
    if not filtered_df.empty:
        recent_calls = filtered_df.head(10).copy()
        recent_calls['Duration'] = recent_calls['duration_seconds'].apply(format_duration)
        recent_calls['Score Class'] = recent_calls['overall_score'].apply(get_score_color_class)

        # Display table with formatting
        display_cols = ['call_date', 'agent_id', 'customer_phone', 'overall_score', 'Duration']
        recent_display = recent_calls[display_cols].copy()
        recent_display.columns = ['Call Date', 'Agent', 'Customer', 'Score', 'Duration']
        recent_display['Call Date'] = recent_display['Call Date'].dt.strftime('%Y-%m-%d %H:%M')

        st.dataframe(
            recent_display,
            use_container_width=True,
            column_config={
                "Score": st.column_config.NumberColumn(
                    "Score",
                    help="Overall call quality score",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                )
            }
        )

def show_call_analysis(calls_df):
    """Show detailed call analysis"""

    st.header("🔍 Call Analysis")

    if calls_df.empty:
        st.info("No calls to analyze yet. Upload test calls or wait for FTP processing.")
        return

    # Filters section
    st.subheader("🎯 Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        agents = ["All"] + sorted(calls_df['agent_id'].unique().tolist())
        agent_filter = st.selectbox("Agent", agents)

    with col2:
        score_ranges = ["All", "Excellent (90+)", "Good (75-89)", "Average (60-74)", "Poor (<60)"]
        score_filter = st.selectbox("Score Range", score_ranges)

    with col3:
        campaigns = ["All"] + sorted(calls_df['campaign_id'].unique().tolist())
        campaign_filter = st.selectbox("Campaign", campaigns)

    with col4:
        # Date range filter
        min_date = calls_df['call_date'].min().date()
        max_date = calls_df['call_date'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), 
                                  min_value=min_date, max_value=max_date)

    # Apply filters
    filtered_df = calls_df.copy()

    if agent_filter != "All":
        filtered_df = filtered_df[filtered_df['agent_id'] == agent_filter]

    if campaign_filter != "All":
        filtered_df = filtered_df[filtered_df['campaign_id'] == campaign_filter]

    if score_filter != "All":
        if score_filter == "Excellent (90+)":
            filtered_df = filtered_df[filtered_df['overall_score'] >= 90]
        elif score_filter == "Good (75-89)":
            filtered_df = filtered_df[(filtered_df['overall_score'] >= 75) & (filtered_df['overall_score'] < 90)]
        elif score_filter == "Average (60-74)":
            filtered_df = filtered_df[(filtered_df['overall_score'] >= 60) & (filtered_df['overall_score'] < 75)]
        elif score_filter == "Poor (<60)":
            filtered_df = filtered_df[filtered_df['overall_score'] < 60]

    # Date range filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['call_date'].dt.date >= start_date) & 
            (filtered_df['call_date'].dt.date <= end_date)
        ]

    st.info(f"📊 Found {len(filtered_df)} calls matching filters")

    if filtered_df.empty:
        st.warning("No calls match the selected filters.")
        return

    # Call selection and analysis
    st.subheader("📞 Select Call for Detailed Analysis")

    # Create a more readable call selector
    call_options = []
    for _, call in filtered_df.iterrows():
        call_label = f"Call {call['id']} - Agent {call['agent_id']} - {call['call_date'].strftime('%Y-%m-%d %H:%M')} - Score: {call['overall_score']:.1f}"
        call_options.append((call['id'], call_label))

    if call_options:
        selected_call_id = st.selectbox(
            "Choose call",
            options=[opt[0] for opt in call_options],
            format_func=lambda x: next(opt[1] for opt in call_options if opt[0] == x)
        )

        if selected_call_id:
            call_data = filtered_df[filtered_df['id'] == selected_call_id].iloc[0]
            show_call_details(call_data)

def show_call_details(call_data):
    """Show detailed analysis of a specific call"""

    st.subheader(f"📞 Call Analysis - ID {call_data['id']}")

    # Basic info cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Agent</strong><br>
            {call_data['agent_id']}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Customer</strong><br>
            {call_data['customer_phone']}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Campaign</strong><br>
            {call_data['campaign_id']} | Lead: {call_data['lead_id']}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        score_class = get_score_color_class(call_data['overall_score'])
        st.markdown(f"""
        <div class="call-score {score_class}">
            {call_data['overall_score']:.1f}
        </div>
        """, unsafe_allow_html=True)

    # Call metadata
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📅 **Date:** {call_data['call_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.info(f"⏱️ **Duration:** {format_duration(call_data['duration_seconds'])}")
    with col2:
        st.info(f"📁 **File:** {call_data['file_path']}")
        st.info(f"📊 **Processed:** {call_data['processed_date'].strftime('%Y-%m-%d %H:%M')}")

    # Score breakdown with visual indicators
    st.subheader("📊 Detailed Score Breakdown")

    scores_data = {
        'Category': ['Sentiment', 'Clarity', 'Professionalism', 'Script Adherence'],
        'Score': [
            call_data['sentiment_score'],
            call_data['clarity_score'], 
            call_data['professionalism_score'],
            call_data['script_adherence_score']
        ]
    }

    # Create score visualization
    fig = px.bar(
        x=scores_data['Category'],
        y=scores_data['Score'],
        title="Score Breakdown by Category",
        color=scores_data['Score'],
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silence %", f"{call_data['silence_percentage']:.1f}%")
    with col2:
        st.metric("Interruptions", int(call_data['interruptions_count']))
    with col3:
        # Calculate talk ratio (rough estimate)
        talk_time = call_data['duration_seconds'] * (1 - call_data['silence_percentage']/100)
        st.metric("Talk Time", format_duration(talk_time))

    # Transcript section
    st.subheader("📝 Call Transcript")
    transcript_container = st.container()
    with transcript_container:
        if call_data['transcript']:
            # Make transcript expandable and searchable
            with st.expander("View Full Transcript", expanded=False):
                st.text_area("Transcript", call_data['transcript'], height=300, disabled=True)

            # Show transcript preview
            preview = call_data['transcript'][:500]
            if len(call_data['transcript']) > 500:
                preview += "..."
            st.text(preview)
        else:
            st.info("No transcript available")

    # Issues and coaching section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠️ Issues Flagged")
        try:
            issues = json.loads(call_data['issues_flagged']) if call_data['issues_flagged'] else []
            if issues:
                for issue in issues:
                    st.markdown(f"<div class='issue-flag'>⚠️ {issue}</div>", unsafe_allow_html=True)
            else:
                st.success("✅ No issues detected")
        except:
            if call_data['issues_flagged']:
                st.text(call_data['issues_flagged'])
            else:
                st.success("✅ No issues detected")

    with col2:
        st.subheader("💡 Coaching Recommendations")
        if call_data['coaching_notes']:
            notes = call_data['coaching_notes'].split(' | ')
            for note in notes:
                st.markdown(f"<div class='coaching-note'>💡 {note}</div>", unsafe_allow_html=True)
        else:
            st.info("No specific coaching recommendations")

    # Keywords and topics
    st.subheader("🔍 Keywords & Topics")
    try:
        keywords = json.loads(call_data['keywords_found']) if call_data['keywords_found'] else []
        if keywords:
            keyword_cols = st.columns(len(keywords))
            for i, keyword in enumerate(keywords):
                with keyword_cols[i]:
                    st.info(f"🏷️ {keyword}")
        else:
            st.info("No keywords identified")
    except:
        st.info("Keywords analysis not available")

def show_agent_performance(calls_df):
    """Show agent performance analytics"""

    st.header("👥 Agent Performance Analytics")

    if calls_df.empty:
        st.info("No performance data available yet.")
        return

    # Time period selector
    period = st.selectbox("Analysis Period", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])

    # Filter data based on period
    if period == "Last 7 days":
        cutoff = datetime.now() - timedelta(days=7)
    elif period == "Last 30 days":
        cutoff = datetime.now() - timedelta(days=30)
    elif period == "Last 90 days":
        cutoff = datetime.now() - timedelta(days=90)
    else:
        cutoff = datetime.min

    filtered_df = calls_df[calls_df['call_date'] >= cutoff] if cutoff != datetime.min else calls_df

    # Agent summary statistics
    st.subheader("📊 Agent Summary")

    agent_stats = filtered_df.groupby('agent_id').agg({
        'overall_score': ['mean', 'std', 'count'],
        'sentiment_score': 'mean',
        'clarity_score': 'mean',
        'professionalism_score': 'mean',
        'script_adherence_score': 'mean',
        'duration_seconds': 'mean',
        'silence_percentage': 'mean',
        'interruptions_count': 'mean'
    }).round(2)

    # Flatten column names
    agent_stats.columns = ['Avg Score', 'Score StdDev', 'Total Calls', 'Avg Sentiment', 
                          'Avg Clarity', 'Avg Professionalism', 'Avg Script', 
                          'Avg Duration', 'Avg Silence %', 'Avg Interruptions']
    agent_stats = agent_stats.reset_index()

    # Add performance indicators
    agent_stats['Performance'] = agent_stats['Avg Score'].apply(
        lambda x: '🟢 Excellent' if x >= 90 else '🟡 Good' if x >= 75 else '🔴 Needs Improvement'
    )

    # Display agent table with conditional formatting
    st.dataframe(
        agent_stats,
        use_container_width=True,
        column_config={
            "Avg Score": st.column_config.NumberColumn(
                "Avg Score",
                help="Average overall score",
                min_value=0,
                max_value=100,
                format="%.1f"
            ),
            "Avg Duration": st.column_config.NumberColumn(
                "Avg Duration (sec)",
                help="Average call duration in seconds",
                format="%.0f"
            )
        }
    )

    # Agent comparison charts
    if len(agent_stats) > 1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Agent Score Comparison")
            fig_comparison = px.bar(
                agent_stats.sort_values('Avg Score'),
                x='agent_id',
                y='Avg Score',
                title="Average Score by Agent",
                color='Avg Score',
                color_continuous_scale='RdYlGn',
                text='Avg Score'
            )
            fig_comparison.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_comparison.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_comparison, use_container_width=True)

        with col2:
            st.subheader("📞 Call Volume by Agent")
            fig_volume = px.bar(
                agent_stats.sort_values('Total Calls'),
                x='agent_id',
                y='Total Calls',
                title="Total Calls by Agent",
                text='Total Calls'
            )
            fig_volume.update_traces(textposition='outside')
            st.plotly_chart(fig_volume, use_container_width=True)

    # Individual agent deep dive
    st.subheader("🔍 Individual Agent Analysis")
    selected_agent = st.selectbox("Select Agent for Details", filtered_df['agent_id'].unique())

    if selected_agent:
        agent_calls = filtered_df[filtered_df['agent_id'] == selected_agent].copy()

        # Agent metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Calls", len(agent_calls))

        with col2:
            avg_score = agent_calls['overall_score'].mean()
            overall_avg = filtered_df['overall_score'].mean()
            delta = avg_score - overall_avg
            st.metric("Average Score", f"{avg_score:.1f}", delta=f"{delta:+.1f} vs avg")

        with col3:
            flagged = len(agent_calls[agent_calls['overall_score'] < 70])
            flagged_pct = (flagged / len(agent_calls) * 100) if len(agent_calls) > 0 else 0
            st.metric("Flagged Calls", flagged, delta=f"{flagged_pct:.1f}%")

        with col4:
            avg_duration = agent_calls['duration_seconds'].mean()
            st.metric("Avg Duration", format_duration(avg_duration))

        # Performance trend
        if len(agent_calls) > 1:
            st.subheader(f"📈 Performance Trend - Agent {selected_agent}")

            # Daily trend
            agent_daily = agent_calls.groupby(agent_calls['call_date'].dt.date).agg({
                'overall_score': 'mean',
                'id': 'count'
            }).reset_index()
            agent_daily.columns = ['Date', 'Avg Score', 'Call Count']

            fig_trend = px.line(
                agent_daily,
                x='Date',
                y='Avg Score',
                title=f"Daily Score Trend for Agent {selected_agent}",
                markers=True
            )
            fig_trend.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_trend, use_container_width=True)

            # Recent calls for this agent
            st.subheader("🕐 Recent Calls")
            recent_agent_calls = agent_calls.head(10)[['call_date', 'customer_phone', 'overall_score', 'duration_seconds', 'issues_flagged']]
            recent_agent_calls['Duration'] = recent_agent_calls['duration_seconds'].apply(format_duration)
            recent_agent_calls['Call Date'] = recent_agent_calls['call_date'].dt.strftime('%Y-%m-%d %H:%M')

            display_calls = recent_agent_calls[['Call Date', 'customer_phone', 'overall_score', 'Duration']].copy()
            display_calls.columns = ['Call Date', 'Customer', 'Score', 'Duration']

            st.dataframe(display_calls, use_container_width=True)

def show_reports_export(calls_df):
    """Show reports and export functionality"""

    st.header("📋 Reports & Export")

    if calls_df.empty:
        st.info("No data available for reports yet.")
        return

    # Report type selection
    report_type = st.selectbox("Report Type", [
        "Agent Performance Summary",
        "Call Quality Analysis", 
        "Issue Trend Report",
        "Campaign Performance",
        "Custom Data Export"
    ])

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())

    # Filter data by date range
    filtered_df = calls_df[
        (calls_df['call_date'].dt.date >= start_date) & 
        (calls_df['call_date'].dt.date <= end_date)
    ]

    if filtered_df.empty:
        st.warning("No data found for selected date range.")
        return

    st.info(f"Report period: {start_date} to {end_date} ({len(filtered_df)} calls)")

    if report_type == "Agent Performance Summary":
        show_agent_report(filtered_df)
    elif report_type == "Call Quality Analysis":
        show_quality_report(filtered_df)
    elif report_type == "Issue Trend Report":
        show_issue_report(filtered_df)
    elif report_type == "Campaign Performance":
        show_campaign_report(filtered_df)
    elif report_type == "Custom Data Export":
        show_custom_export(filtered_df)

def show_agent_report(df):
    """Generate agent performance report"""
    st.subheader("👥 Agent Performance Summary Report")

    # Generate report data
    agent_summary = df.groupby('agent_id').agg({
        'overall_score': ['mean', 'min', 'max', 'count'],
        'sentiment_score': 'mean',
        'clarity_score': 'mean',
        'professionalism_score': 'mean',
        'duration_seconds': 'mean'
    }).round(2)

    agent_summary.columns = ['Avg Score', 'Min Score', 'Max Score', 'Total Calls',
                            'Avg Sentiment', 'Avg Clarity', 'Avg Professionalism', 'Avg Duration']
    agent_summary = agent_summary.reset_index()

    st.dataframe(agent_summary, use_container_width=True)

    # Export button
    csv_data = agent_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Agent Report (CSV)",
        data=csv_data,
        file_name=f"agent_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

def show_quality_report(df):
    """Generate call quality analysis report"""
    st.subheader("📊 Call Quality Analysis Report")

    # Quality metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Average Score", f"{df['overall_score'].mean():.1f}")
        st.metric("Score Standard Deviation", f"{df['overall_score'].std():.1f}")

    with col2:
        excellent_calls = len(df[df['overall_score'] >= 90])
        good_calls = len(df[(df['overall_score'] >= 75) & (df['overall_score'] < 90)])
        st.metric("Excellent Calls (90+)", f"{excellent_calls} ({excellent_calls/len(df)*100:.1f}%)")
        st.metric("Good Calls (75-89)", f"{good_calls} ({good_calls/len(df)*100:.1f}%)")

    with col3:
        poor_calls = len(df[df['overall_score'] < 60])
        flagged_calls = len(df[df['overall_score'] < 70])
        st.metric("Poor Calls (<60)", f"{poor_calls} ({poor_calls/len(df)*100:.1f}%)")
        st.metric("Flagged Calls (<70)", f"{flagged_calls} ({flagged_calls/len(df)*100:.1f}%)")

    # Detailed quality breakdown
    quality_breakdown = pd.DataFrame({
        'Metric': ['Sentiment', 'Clarity', 'Professionalism', 'Script Adherence'],
        'Average': [
            df['sentiment_score'].mean(),
            df['clarity_score'].mean(),
            df['professionalism_score'].mean(),
            df['script_adherence_score'].mean()
        ],
        'Std Dev': [
            df['sentiment_score'].std(),
            df['clarity_score'].std(),
            df['professionalism_score'].std(),
            df['script_adherence_score'].std()
        ]
    }).round(2)

    st.dataframe(quality_breakdown, use_container_width=True)

    # Export button
    csv_data = quality_breakdown.to_csv(index=False)
    st.download_button(
        label="📥 Download Quality Report (CSV)",
        data=csv_data,
        file_name=f"quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

def show_issue_report(df):
    """Generate issue trend report"""
    st.subheader("⚠️ Issue Trend Analysis")

    # Extract all issues
    all_issues = []
    issue_dates = []

    for _, row in df.iterrows():
        try:
            issues = json.loads(row['issues_flagged']) if row['issues_flagged'] else []
            for issue in issues:
                all_issues.append(issue)
                issue_dates.append(row['call_date'].date())
        except:
            continue

    if all_issues:
        issue_df = pd.DataFrame({
            'Issue': all_issues,
            'Date': issue_dates
        })

        # Issue frequency
        issue_counts = issue_df['Issue'].value_counts()
        st.bar_chart(issue_counts)

        # Issue trend over time
        issue_trend = issue_df.groupby(['Date', 'Issue']).size().reset_index(name='Count')

        if len(issue_trend) > 0:
            fig_trend = px.line(issue_trend, x='Date', y='Count', color='Issue', 
                              title="Issue Trends Over Time")
            st.plotly_chart(fig_trend, use_container_width=True)

        # Export
        csv_data = issue_counts.to_csv()
        st.download_button(
            label="📥 Download Issue Report (CSV)",
            data=csv_data,
            file_name=f"issue_trends_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No issues identified in the selected period!")

def show_campaign_report(df):
    """Generate campaign performance report"""
    st.subheader("📋 Campaign Performance Report")

    if 'campaign_id' not in df.columns or df['campaign_id'].isna().all():
        st.info("No campaign data available.")
        return

    campaign_summary = df.groupby('campaign_id').agg({
        'overall_score': ['mean', 'count'],
        'duration_seconds': 'mean',
        'agent_id': 'nunique'
    }).round(2)

    campaign_summary.columns = ['Avg Score', 'Total Calls', 'Avg Duration', 'Unique Agents']
    campaign_summary = campaign_summary.reset_index()

    st.dataframe(campaign_summary, use_container_width=True)

    # Export
    csv_data = campaign_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Campaign Report (CSV)",
        data=csv_data,
        file_name=f"campaign_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

def show_custom_export(df):
    """Custom data export interface"""
    st.subheader("🔧 Custom Data Export")

    # Column selection
    available_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to export",
        available_columns,
        default=['call_date', 'agent_id', 'customer_phone', 'overall_score', 'transcript']
    )

    if selected_columns:
        export_df = df[selected_columns].copy()

        # Format dates for export
        if 'call_date' in export_df.columns:
            export_df['call_date'] = export_df['call_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'processed_date' in export_df.columns:
            export_df['processed_date'] = export_df['processed_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        st.write(f"Export preview ({len(export_df)} rows):")
        st.dataframe(export_df.head(10), use_container_width=True)

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

def show_settings():
    """Show settings and configuration"""

    st.header("⚙️ Settings & Configuration")

    # Scoring weights configuration
    st.subheader("🎯 Scoring Weights")
    st.info("Adjust the importance of different quality factors in the overall score calculation.")

    try:
        conn = sqlite3.connect('call_qa_database.db')
        weights_df = pd.read_sql_query("SELECT * FROM scoring_weights ORDER BY category", conn)

        if not weights_df.empty:
            weights_changed = False
            new_weights = {}

            for idx, row in weights_df.iterrows():
                col1, col2, col3 = st.columns([2, 1, 3])

                with col1:
                    st.write(f"**{row['category'].title()}**")

                with col2:
                    new_weight = st.slider(
                        "Weight",
                        0.0, 1.0, float(row['weight']),
                        step=0.05,
                        key=f"weight_{row['category']}"
                    )
                    new_weights[row['category']] = new_weight
                    if abs(new_weight - row['weight']) > 0.01:
                        weights_changed = True

                with col3:
                    st.write(row['description'])

            # Validation
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Total weights should sum to 1.0 (currently {total_weight:.2f})")

            # Update button
            if weights_changed and st.button("💾 Update Weights"):
                cursor = conn.cursor()
                for category, weight in new_weights.items():
                    cursor.execute("""
                        UPDATE scoring_weights 
                        SET weight = ?, updated_date = ? 
                        WHERE category = ?
                    """, (weight, datetime.now(), category))
                conn.commit()
                st.success("✅ Weights updated successfully!")
                time.sleep(1)
                st.rerun()

        conn.close()

    except Exception as e:
        st.error(f"Error loading scoring weights: {e}")

    # Processing settings
    st.subheader("🔧 Processing Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Whisper Model Settings**
        - Current: Base model (balanced speed/accuracy)
        - Options: tiny, base, small, medium, large
        - Change via environment variables
        """)

    with col2:
        st.info("""
        **File Processing**
        - Auto-delete after processing: ✅ Enabled
        - Watch folder: incoming_calls/
        - Processed folder: processed_calls/
        """)

    # System information
    st.subheader("📊 System Information")

    try:
        # Database stats
        conn = sqlite3.connect('call_qa_database.db')
        call_count = pd.read_sql_query("SELECT COUNT(*) as count FROM calls", conn)['count'].iloc[0]

        # Disk usage (rough estimate)
        db_size = os.path.getsize('call_qa_database.db') / (1024*1024)  # MB

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Calls in DB", call_count)

        with col2:
            st.metric("Database Size", f"{db_size:.1f} MB")

        with col3:
            # Available disk space would require additional imports
            st.metric("Status", "✅ Running")

        conn.close()

    except Exception as e:
        st.error(f"Error getting system info: {e}")

    # Data management
    st.subheader("🗃️ Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Cleanup Old Data (30+ days)", type="secondary"):
            try:
                conn = sqlite3.connect('call_qa_database.db')
                cutoff_date = datetime.now() - timedelta(days=30)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM calls WHERE call_date < ?", (cutoff_date,))
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                st.success(f"✅ Deleted {deleted_count} old records")
            except Exception as e:
                st.error(f"Error cleaning up data: {e}")

    with col2:
        if st.button("💾 Backup Database", type="secondary"):
            try:
                import shutil
                backup_name = f"call_qa_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2('call_qa_database.db', backup_name)
                st.success(f"✅ Database backed up as {backup_name}")
            except Exception as e:
                st.error(f"Error creating backup: {e}")

def show_upload_test():
    """Show test upload functionality"""

    st.header("📤 Upload Test Call")
    st.info("Upload an MP3 file to test the call processing pipeline.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MP3 file", 
        type=['mp3'],
        help="Upload a call recording to test the AI analysis pipeline"
    )

    if uploaded_file is not None:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

        # Parse filename to show expected metadata
        metadata = parse_test_filename(uploaded_file.name)
        if metadata:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"**Campaign:** {metadata['campaign_id']}")
            with col2:
                st.info(f"**Agent:** {metadata['agent_id']}")
            with col3:
                st.info(f"**Customer:** {metadata['customer_phone']}")
            with col4:
                st.info(f"**Lead ID:** {metadata['lead_id']}")

        # Processing options
        st.subheader("🔧 Processing Options")

        col1, col2 = st.columns(2)
        with col1:
            simulate_processing = st.checkbox("Simulate Processing (Demo Mode)", value=True,
                                            help="Use simulated data instead of real Whisper transcription")
        with col2:
            auto_cleanup = st.checkbox("Auto-cleanup after processing", value=True,
                                      help="Delete the uploaded file after processing")

        # Process button
        if st.button("🚀 Process Call", type="primary"):
            process_uploaded_file(uploaded_file, simulate_processing, auto_cleanup)

def parse_test_filename(filename):
    """Parse test filename for metadata"""
    try:
        basename = filename.replace('.mp3', '')
        parts = basename.split('_')

        if len(parts) >= 4:
            return {
                'campaign_id': parts[0],
                'agent_id': parts[1],
                'customer_phone': parts[2],
                'lead_id': parts[3]
            }
    except:
        pass
    return None

def process_uploaded_file(uploaded_file, simulate, auto_cleanup):
    """Process uploaded test file"""

    # Create progress indicator
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Save file
            status_text.text("💾 Saving uploaded file...")
            progress_bar.progress(20)

            # Ensure directories exist
            os.makedirs("incoming_calls", exist_ok=True)
            os.makedirs("processed_calls", exist_ok=True)

            file_path = os.path.join("incoming_calls", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            time.sleep(0.5)  # Visual delay

            # Step 2: Parse metadata
            status_text.text("📋 Parsing file metadata...")
            progress_bar.progress(40)

            metadata = parse_test_filename(uploaded_file.name)
            if not metadata:
                st.error("❌ Could not parse filename. Expected format: campaign_agent_phone_leadid_xxx.mp3")
                return

            time.sleep(0.5)

            # Step 3: Transcription (simulated or real)
            status_text.text("🎵 Transcribing audio...")
            progress_bar.progress(60)

            if simulate:
                # Simulated transcription
                transcript_data = {
                    'text': "Hello, thank you for calling about your motor vehicle accident case. I understand you were involved in an accident recently and may need legal representation. Can you tell me more about what happened and any injuries you sustained? I want to make sure we can connect you with the right attorney for your specific situation.",
                    'duration': 45.0
                }
            else:
                # Would use real Whisper here
                transcript_data = {
                    'text': "Real Whisper transcription would appear here",
                    'duration': 60.0
                }

            time.sleep(1.0)

            # Step 4: AI Analysis
            status_text.text("🧠 Analyzing call quality...")
            progress_bar.progress(80)

            # Simulate analysis
            analysis_results = {
                'sentiment_score': 78.5,
                'clarity_score': 82.0,
                'professionalism_score': 85.5,
                'script_adherence_score': 76.0,
                'silence_percentage': 12.5,
                'interruptions_count': 3,
                'overall_score': 80.5
            }

            issues = ["Minor filler words detected"]
            coaching_notes = "Good professional tone. Consider reducing filler words for improved delivery."

            time.sleep(1.0)

            # Step 5: Save to database
            status_text.text("💾 Saving results to database...")
            progress_bar.progress(90)

            save_test_call_data(metadata, transcript_data, analysis_results, issues, coaching_notes, uploaded_file.name)

            # Step 6: Cleanup
            if auto_cleanup and os.path.exists(file_path):
                os.remove(file_path)
                status_text.text("🧹 Cleaning up temporary files...")
            else:
                status_text.text("✅ Processing complete (file preserved)")

            progress_bar.progress(100)
            time.sleep(0.5)

            # Show results
            st.success("🎉 Call processed successfully!")

            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{analysis_results['overall_score']:.1f}")
            with col2:
                st.metric("Sentiment", f"{analysis_results['sentiment_score']:.1f}")
            with col3:
                st.metric("Clarity", f"{analysis_results['clarity_score']:.1f}")
            with col4:
                st.metric("Professionalism", f"{analysis_results['professionalism_score']:.1f}")

            # Show transcript preview
            st.subheader("📝 Transcript Preview")
            st.text_area("Generated Transcript", transcript_data['text'], height=100, disabled=True)

            # Show coaching notes
            if coaching_notes:
                st.subheader("💡 Coaching Notes")
                st.info(coaching_notes)

            # Show issues
            if issues:
                st.subheader("⚠️ Issues Identified")
                for issue in issues:
                    st.warning(f"⚠️ {issue}")

            st.info("💡 Check the Dashboard or Call Analysis pages to see this call in the system!")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")

def save_test_call_data(metadata, transcript_data, analysis, issues, coaching_notes, filename):
    """Save test call data to database"""
    try:
        conn = sqlite3.connect('call_qa_database.db')
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO calls (
                campaign_id, agent_id, customer_phone, lead_id, call_date,
                file_path, file_size, duration_seconds, transcript,
                sentiment_score, clarity_score, professionalism_score,
                script_adherence_score, overall_score, silence_percentage,
                interruptions_count, keywords_found, issues_flagged,
                coaching_notes, processed_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata['campaign_id'],
            metadata['agent_id'],
            metadata['customer_phone'],
            metadata['lead_id'],
            datetime.now(),
            filename,
            0,  # file_size (not relevant for test)
            transcript_data['duration'],
            transcript_data['text'],
            analysis['sentiment_score'],
            analysis['clarity_score'],
            analysis['professionalism_score'],
            analysis['script_adherence_score'],
            analysis['overall_score'],
            analysis['silence_percentage'],
            analysis['interruptions_count'],
            json.dumps(['motor vehicle', 'accident', 'legal']),
            json.dumps(issues),
            coaching_notes,
            datetime.now()
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        raise Exception(f"Database save failed: {e}")

def show_system_status(calls_df):
    """Show system status and monitoring"""

    st.header("🔧 System Status")

    # System health indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System Status", "🟢 Running")

    with col2:
        # Count recent activity
        recent_calls = len(calls_df[calls_df['processed_date'] >= datetime.now() - timedelta(hours=24)])
        st.metric("Calls Last 24h", recent_calls)

    with col3:
        # Processing health
        if not calls_df.empty:
            avg_processing_time = "< 2 min"  # Estimated
        else:
            avg_processing_time = "N/A"
        st.metric("Avg Processing Time", avg_processing_time)

    # Directory status
    st.subheader("📁 Directory Status")

    directories = {
        "Incoming Calls": "incoming_calls",
        "Processed Calls": "processed_calls"
    }

    for name, path in directories.items():
        exists = os.path.exists(path)
        if exists:
            file_count = len([f for f in os.listdir(path) if f.endswith('.mp3')])
            st.success(f"✅ {name}: {file_count} files")
        else:
            st.error(f"❌ {name}: Directory not found")

    # Database status
    st.subheader("🗄️ Database Status")

    try:
        conn = sqlite3.connect('call_qa_database.db')

        # Table information
        tables_info = pd.read_sql_query("""
            SELECT name, COUNT(*) as record_count
            FROM (
                SELECT 'calls' as name FROM calls
                UNION ALL
                SELECT 'scoring_weights' as name FROM scoring_weights
            ) 
            GROUP BY name
        """, conn)

        for _, row in tables_info.iterrows():
            st.info(f"📊 {row['name']}: {row['record_count']} records")

        # Recent activity
        if not calls_df.empty:
            latest_call = calls_df['processed_date'].max()
            st.info(f"🕐 Latest processing: {latest_call.strftime('%Y-%m-%d %H:%M:%S')}")

        conn.close()

    except Exception as e:
        st.error(f"❌ Database error: {e}")

    # Performance metrics
    st.subheader("📊 Performance Metrics")

    if not calls_df.empty:
        # Processing statistics
        total_processed = len(calls_df)
        avg_score = calls_df['overall_score'].mean()
        score_std = calls_df['overall_score'].std()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Processed", total_processed)

        with col2:
            st.metric("Average Quality Score", f"{avg_score:.1f}")

        with col3:
            st.metric("Score Consistency", f"σ={score_std:.1f}")

        # Show processing timeline
        if len(calls_df) > 1:
            st.subheader("📈 Processing Timeline")

            # Group by hour for recent activity
            hourly_counts = calls_df.groupby(calls_df['processed_date'].dt.floor('H')).size()

            if len(hourly_counts) > 0:
                fig = px.line(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Call Processing Over Time",
                    labels={'x': 'Time', 'y': 'Calls Processed'}
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📊 No processing metrics available yet")

    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()

if __name__ == "__main__":
    main()
