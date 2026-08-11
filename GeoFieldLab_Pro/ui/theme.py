"""Professional Streamlit theme helpers for GEONARBA."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Apply compact scientific-app styling."""

    st.markdown(
        """
        <style>
        :root {
            --geon-bg: #f5f7fb;
            --geon-panel: #ffffff;
            --geon-border: #d9e2ef;
            --geon-ink: #152238;
            --geon-muted: #64748b;
            --geon-accent: #0f766e;
            --geon-accent-soft: #e6f4f1;
            --geon-warn: #8a5a00;
            --geon-warn-bg: #fff7db;
        }

        .stApp {
            background: var(--geon-bg);
            color: var(--geon-ink);
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }

        section[data-testid="stSidebar"] * {
            color: #e5edf7;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #f8fafc;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        h1, h2, h3 {
            color: var(--geon-ink);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: var(--geon-panel);
            border: 1px solid var(--geon-border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--geon-muted);
        }

        .geon-hero {
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.12);
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        }

        .geon-hero h1 {
            color: #ffffff;
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }

        .geon-hero p {
            color: #cbd5e1;
            margin: 0.4rem 0 0 0;
            font-size: 1rem;
        }

        .geon-panel {
            background: var(--geon-panel);
            border: 1px solid var(--geon-border);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0 1rem 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .geon-panel h2,
        .geon-panel h3 {
            margin-top: 0;
        }

        .geon-intro {
            background: var(--geon-panel);
            border-left: 4px solid var(--geon-accent);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
            color: var(--geon-muted);
        }

        .geon-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.25rem 0 0.8rem 0;
        }

        .geon-badge {
            background: var(--geon-accent-soft);
            border: 1px solid #b7ded8;
            border-radius: 999px;
            color: #115e59;
            display: inline-flex;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.22rem 0.55rem;
            white-space: nowrap;
        }

        .geon-warning {
            background: var(--geon-warn-bg);
            border: 1px solid #f4d47c;
            border-radius: 8px;
            color: var(--geon-warn);
            padding: 0.75rem 0.9rem;
            margin: 0.5rem 0;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 6px;
            border: 1px solid #0f766e;
            color: #0f766e;
            font-weight: 600;
        }

        div.stButton > button[kind="primary"] {
            background: #0f766e;
            color: #ffffff;
            border-color: #0f766e;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--geon-border);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_header(app_name: str, active_section: str) -> None:
    st.markdown(
        f"""
        <div class="geon-hero">
            <h1>{app_name} V1</h1>
            <p>Potential-field processing for gravity and magnetic anomaly grids. Active workspace: {active_section}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="geon-intro">
            <strong>{title}</strong><br>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_row(items: list[str]) -> None:
    badges = "".join(f'<span class="geon-badge">{item}</span>' for item in items)
    st.markdown(f'<div class="geon-badge-row">{badges}</div>', unsafe_allow_html=True)


def panel_start() -> None:
    st.markdown('<div class="geon-panel">', unsafe_allow_html=True)


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
