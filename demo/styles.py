"""Custom CSS for the AortaCFD Agent demo app."""

CUSTOM_CSS = """
<style>
    /* Clean header spacing */
    .main .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
    }

    /* Pipeline stage cards */
    .stage-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Citation blockquotes */
    blockquote {
        border-left: 3px solid #3b82f6;
        padding-left: 1rem;
        margin-left: 0;
        color: #4b5563;
        font-style: italic;
    }

    /* Confidence badges */
    .badge-high { color: #059669; font-weight: 600; }
    .badge-medium { color: #d97706; font-weight: 600; }
    .badge-low { color: #dc2626; font-weight: 600; }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Divider spacing */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
"""
