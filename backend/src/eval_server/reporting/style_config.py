"""Theme and styling configuration for reports."""

from __future__ import annotations

from typing import Dict, NamedTuple


class Theme(NamedTuple):
    """Theme definition for report styling."""
    name: str
    primary_color: str
    secondary_color: str
    accent_color: str
    background_color: str
    text_color: str
    pass_color: str
    fail_color: str
    header_bg: str
    card_bg: str


# Predefined themes
THEMES = {
    "default": Theme(
        name="default",
        primary_color="#667eea",
        secondary_color="#764ba2",
        accent_color="#f093fb",
        background_color="#f5f7fa",
        text_color="#333333",
        pass_color="#10b981",
        fail_color="#ef4444",
        header_bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        card_bg="#ffffff",
    ),
    "dark": Theme(
        name="dark",
        primary_color="#3b82f6",
        secondary_color="#1e40af",
        accent_color="#60a5fa",
        background_color="#1f2937",
        text_color="#f3f4f6",
        pass_color="#34d399",
        fail_color="#f87171",
        header_bg="linear-gradient(135deg, #1e3a8a 0%, #111827 100%)",
        card_bg="#374151",
    ),
    "compact": Theme(
        name="compact",
        primary_color="#0ea5e9",
        secondary_color="#06b6d4",
        accent_color="#14b8a6",
        background_color="#f8fafc",
        text_color="#1e293b",
        pass_color="#16a34a",
        fail_color="#dc2626",
        header_bg="linear-gradient(135deg, #0284c7 0%, #0369a1 100%)",
        card_bg="#ffffff",
    ),
}


def get_theme(theme_name: str = "default") -> Theme:
    """Get a theme by name.

    Args:
        theme_name: Theme name (default, dark, compact).

    Returns:
        Theme configuration.
    """
    return THEMES.get(theme_name, THEMES["default"])


def generate_css_for_theme(theme: Theme) -> str:
    """Generate CSS for a given theme.

    Args:
        theme: Theme configuration.

    Returns:
        CSS string with theme variables.
    """
    return f"""
        :root {{
            --primary-color: {theme.primary_color};
            --secondary-color: {theme.secondary_color};
            --accent-color: {theme.accent_color};
            --background-color: {theme.background_color};
            --text-color: {theme.text_color};
            --pass-color: {theme.pass_color};
            --fail-color: {theme.fail_color};
        }}
        
        body {{
            background: var(--background-color);
            color: var(--text-color);
        }}
        
        header {{
            background: {theme.header_bg};
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        }}
        
        .section h2 {{
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
        }}
        
        .badge-pass {{
            background-color: var(--pass-color);
        }}
        
        .badge-fail {{
            background-color: var(--fail-color);
        }}
        
        .conversation-group {{
            border-left-color: var(--primary-color);
        }}
        
        .conversation-header h3 {{
            color: var(--primary-color);
        }}
        
        .metric-bar[style*="green"] {{
            background-color: var(--pass-color) !important;
        }}
        
        .metric-bar[style*="red"] {{
            background-color: var(--fail-color) !important;
        }}
        
        code {{
            background: var(--secondary-color);
            color: var(--accent-color);
        }}
"""


__all__ = ["Theme", "THEMES", "get_theme", "generate_css_for_theme"]
