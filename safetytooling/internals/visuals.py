"""Visualization utilities for displaying token activations."""

import html
import re
import uuid
from typing import Dict, List, Tuple

import numpy as np


def _interpolate_color(
    start_color: Tuple[int, int, int], end_color: Tuple[int, int, int], factor: float
) -> Tuple[int, int, int]:
    """Interpolate between two RGB colors."""
    return tuple(int(start + factor * (end - start)) for start, end in zip(start_color, end_color))


def _generate_highlighted_html(tokens: List[str], activations: List[float], use_orange_highlight: bool = False) -> str:
    """Generate HTML for text with activation-based highlighting.

    Args:
        tokens: List of text tokens
        activations: List of activation values for each token
        use_orange_highlight: Whether to use an orange highlight instead of red or blue

    Returns:
        HTML string with highlighted tokens and tooltips
    """
    if len(tokens) != len(activations):
        raise ValueError("Number of tokens and activations must match")

    css = """
    <style>
    .text-content span {
        position: relative;
        cursor: help;
    }
    .text-content span:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 5px;
        border-radius: 3px;
        font-size: 14px;
        white-space: nowrap;
    }
    .text-content {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .example {
        margin-bottom: 10px;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }
    .example-label {
        font-weight: bold;
        font-size: 12px;
        color: #666;
        margin-bottom: 2px;
    }
    </style>
    """

    html_content = [css, "<div class='text-content'>"]

    for token, act in zip(tokens, activations):
        act_display = f"{act:.2f}" if act is not None else ""

        if act not in (None, 0):
            # Scale activation using sqrt for better visual distribution
            factor = np.sqrt(abs(act))

            if use_orange_highlight:
                color = _interpolate_color((255, 237, 160), (240, 134, 0), factor)  # Light to dark orange
            else:
                if act > 0:
                    color = _interpolate_color((255, 255, 255), (255, 0, 0), factor)  # White to red
                else:
                    color = _interpolate_color((255, 255, 255), (0, 0, 255), factor)  # White to blue

            html_content.append(
                f'<span class="highlight" style="background-color: rgb{color};" '
                f'title="{token}: {act_display}">{html.escape(token)}</span>'
            )
        else:
            html_content.append(f'<span title="{token}: {act_display}">{html.escape(token)}</span>')

    html_content.append("</div>")
    return "\n".join(html_content)


def _generate_prompt_centric_html(
    examples: List[List[Tuple[str, float]]], title: str | None = None, use_orange_highlight: bool = False
) -> str:
    """Generate HTML content for the prompt-centric view."""
    html_parts = []
    if title:
        html_parts.append(f"<h2>{title}</h2>")

    html_parts.append("<div class='examples-container'>")
    for i, example in enumerate(examples, 1):
        tokens, acts = zip(*example)
        html_parts.extend(
            [
                f'<div class="example">',
                f'<div class="example-label">Example {i}:</div>',
                _generate_highlighted_html(list(tokens), list(acts), use_orange_highlight),
                "</div>",
            ]
        )
    html_parts.append("</div>")
    return "\n".join(html_parts)


def _combine_html_contents(*views: Tuple[str, str], title: str = "Combined View", nested: bool = False) -> str:
    """Combine multiple HTML contents with a dropdown selector."""
    instance_id = str(uuid.uuid4())[:8]

    css = """
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .content-selector {
        margin-bottom: 20px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
    }
    .content {
        display: none;
        background-color: white;
        padding: 20px;
        border-radius: 5px;
    }
    .content.active {
        display: block !important;
    }
    .combined-content {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """

    # Start with basic structure
    html_parts = []
    if not nested:
        html_parts.extend(["<!DOCTYPE html>", "<html>", "<head>", css, "</head>", "<body>"])
    else:
        html_parts.append(css)

    # Add container and title
    html_parts.extend(
        [
            f'<div class="combined-content" id="combined-content-{instance_id}">',
            f"<h2>{title}</h2>",
            '<select class="content-selector" ' f'onchange="showContent_{instance_id}(this.value)">',
            '<option value="">Select a section</option>',
        ]
    )

    # Add dropdown options
    for i, (name, _) in enumerate(views):
        html_parts.append(f'<option value="content-{instance_id}-{i}">{name}</option>')
    html_parts.append("</select>")

    # Add content divs
    for i, (_, content) in enumerate(views):
        html_parts.append(f'<div id="content-{instance_id}-{i}" class="content">{content}</div>')

    # Add JavaScript
    script = f"""
    <script>
    function showContent_{instance_id}(id) {{
        // Hide all content divs
        var contents = document.querySelectorAll('#combined-content-{instance_id} .content');
        contents.forEach(function(content) {{
            content.style.display = 'none';
        }});
        
        // Show selected content
        if (id) {{
            var selectedContent = document.getElementById(id);
            if (selectedContent) {{
                selectedContent.style.display = 'block';
            }}
        }}
    }}
    
    // Initialize first option
    document.addEventListener('DOMContentLoaded', function() {{
        var selector = document.querySelector('#combined-content-{instance_id} .content-selector');
        if (selector.value) {{
            showContent_{instance_id}(selector.value);
        }}
    }});
    </script>
    """

    html_parts.extend([script, "</div>"])  # Close combined-content div

    if not nested:
        html_parts.extend(["</body>", "</html>"])

    return "\n".join(html_parts)


def prompt_centric_view_generic(
    token_act_pairs: List[List[Tuple[str, float]]], title: str = "Examples", use_orange_highlight: bool = False
) -> str:
    """Generate HTML view for a list of token-activation pairs.

    Args:
        token_act_pairs: List of examples, where each example is a list of (token, activation) pairs
        title: Title for the view
        use_orange_highlight: Whether to use orange highlighting instead of red/blue

    Returns:
        HTML string displaying the examples with activation highlighting
    """
    if not isinstance(token_act_pairs, list) or not all(
        isinstance(example, list) and all(isinstance(item, tuple) for item in example) for example in token_act_pairs
    ):
        raise ValueError("Input must be a list of lists of (token, activation) tuples")

    return _generate_prompt_centric_html(token_act_pairs, title, use_orange_highlight)


def prompt_centric_view_generic_dict(
    token_act_pairs_dict: Dict[str, List[List[Tuple[str, float]]]],
    title: str = "Examples",
    use_orange_highlight: bool = False,
) -> str:
    """Generate HTML view for categorized token-activation pairs.

    Args:
        token_act_pairs_dict: Dictionary mapping category names to lists of examples
        title: Title for the combined view
        use_orange_highlight: Whether to use orange highlighting instead of red/blue

    Returns:
        HTML string with dropdown selector for different categories
    """
    views = [
        (name, _generate_prompt_centric_html(examples, name, use_orange_highlight))
        for name, examples in token_act_pairs_dict.items()
    ]
    return _combine_html_contents(*views, title=title, nested=True)
