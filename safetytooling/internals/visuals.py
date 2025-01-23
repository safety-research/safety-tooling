####################################################################################################
#
# This file is a modified version of the original file found at:
# https://github.com/abhay-sheshadri/sae_experiments/blob/abhayexps/src/visualization.py
#
####################################################################################################

import html
import re
import uuid

import matplotlib.pyplot as plt


def _interpolate_color(start_color, end_color, factor):
    """
    Interpolate between two colors.
    """
    return tuple(int(start + factor * (end - start)) for start, end in zip(start_color, end_color))


def _light_mode(html):

    # Define HTML wrapper for light mode display
    light_mode_wrapper = """
    <div style="background-color: white; color: black; padding: 20px;">
        {}
    </div>
    """

    # Generate the categorized examples HTML and wrap it in the light mode div
    return light_mode_wrapper.format(html)


def _generate_highlighted_html(tokens, acts, use_orange_highlight=True):
    """
    Generates HTML for a single example with highlighted text and activation tooltips.
    """

    # Make sure there are an equal number of tokens and activations
    if len(tokens) != len(acts):
        raise ValueError("The number of tokens and activations must match.")

    # Generate html content
    html_content = "<style>\n"
    html_content += """
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
    """
    html_content += "</style>\n"
    html_content += "<div class='text-content'>"

    # Iterate through each token and activation
    for token, act in zip(tokens, acts):

        # Format the activation value for display
        if act is None:
            act_display = ""
        else:
            act_display = f"{act:.2f}"  # Display activation with 2 decimal places

        # Highlight token if act != 0
        if act is not None and act != 0:

            # Normalize activation, max at 1
            factor = min(abs(act) / 10, 1)

            # Interpolate color
            if use_orange_highlight:
                light_orange = (255, 237, 160)  # RGB for very light orange
                dark_orange = (240, 134, 0)  # RGB for dark, saturated orange
                color = _interpolate_color(light_orange, dark_orange, factor)
            else:
                if act > 0:
                    color = _interpolate_color((255, 255, 255), (255, 0, 0), factor)  # White to Red
                else:
                    color = _interpolate_color((255, 255, 255), (0, 0, 255), factor)  # White to Blue
            html_content += f'<span class="highlight" style="background-color: rgb{color};" title="{token}: {act_display}">{html.escape(token)}</span>'
        else:
            html_content += f'<span title="{token}: {act_display}">{html.escape(token)}</span>'
    html_content += "</div>"
    return html_content


def _generate_categorized_examples(categorized_examples):
    """
    Generates complete HTML content for multiple categories of examples.
    """
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            line-height: 1.3;
            margin: 0;
            padding: 10px;
        }
        .highlight {
            border-radius: 2px;
            padding: 0 1px;
        }
        .category {
            margin-bottom: 20px;
        }
        .category-label {
            font-weight: bold;
            font-size: 16px;
            color: #333;
            margin-bottom: 10px;
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
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
        .text-content {
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
    </style>
    """

    # Generate HTML content for the categorized examples
    html_content = "<div class='examples-container'>"

    #  Iterate through each category and its examples
    for category, examples in categorized_examples.items():

        # Add a container for the category
        html_content += f'<div class="category"><div class="category-label"><h3>{html.escape(category)}</h3></div>'
        for i, (tokens, acts) in enumerate(examples, 1):
            html_content += f'<div class="example"><div class="example-label">Example {i}:</div>'
            html_content += _generate_highlighted_html(tokens, acts)
            html_content += "</div>"
        html_content += "</div>"

    # Close the examples container
    html_content += "</div>"
    full_html = f"<!DOCTYPE html><html><head>{css}</head><body>{html_content}</body></html>"
    return full_html


def _generate_logits_table(negative_logits, positive_logits):
    """
    Generate HTML content for a table of negative and positive logits
    """
    css = """
    <style>
        .logits-container {
            display: flex;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 20px auto;
        }
        .logits-column {
            flex: 1;
            margin: 0 5px;
        }
        .logits-column h2 {
            background-color: #f0f0f0;
            margin: 0;
            padding: 10px;
            border-radius: 5px 5px 0 0;
            font-size: 16px;
            font-weight: normal;
        }
        .logits-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        .logits-list li {
            display: flex;
            justify-content: space-between;
            padding: 5px 10px;
        }
        .negative .logits-list li:nth-child(odd) {
            background-color: #ffe6e6;
        }
        .negative .logits-list li:nth-child(even) {
            background-color: #ffcccc;
        }
        .positive .logits-list li:nth-child(odd) {
            background-color: #e6e6ff;
        }
        .positive .logits-list li:nth-child(even) {
            background-color: #ccccff;
        }
        .token {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: rgba(0, 0, 0, 0.1);
            padding: 2px 4px;
            border-radius: 3px;
            color: #333;
            font-weight: bold;  /* Make token text bold */
        }
        .value {
            font-weight: normal;
        }
    </style>
    """
    html_content = f"{css}<div class='logits-container'>"

    # Negative logits column
    html_content += """
    <div class='logits-column negative'>
        <h2>NEGATIVE LOGITS</h2>
        <ul class='logits-list'>
    """
    for token, value in negative_logits:
        html_content += (
            f"<li><span class='token'>{html.escape(token)}</span><span class='value'>{value:.5f}</span></li>"
        )
    html_content += "</ul></div>"

    # Positive logits column
    html_content += """
    <div class='logits-column positive'>
        <h2>POSITIVE LOGITS</h2>
        <ul class='logits-list'>
    """
    for token, value in positive_logits:
        html_content += (
            f"<li><span class='token'>{html.escape(token)}</span><span class='value'>{value:.5f}</span></li>"
        )
    html_content += "</ul></div>"

    html_content += "</div>"
    return html_content


def safe_remove_doctype(content):

    # Remove DOCTYPE declaration
    return re.sub(r"<!DOCTYPE[^>]*>", "", content, flags=re.IGNORECASE)


def safe_remove_html_tags(content):

    # Remove opening HTML tag
    content = re.sub(r"<html[^>]*>", "", content, flags=re.IGNORECASE)

    # Remove closing HTML tag
    content = re.sub(r"</html\s*>", "", content, flags=re.IGNORECASE)
    return content


def safe_remove_head(content):

    # Safely remove HEAD tags
    def find_head_end(s, start):
        depth = 0
        i = start
        while i < len(s):
            if s[i : i + 5].lower() == "<head":
                depth += 1
            elif s[i : i + 7].lower() == "</head>":
                depth -= 1
                if depth == 0:
                    return i + 7
            i += 1
        return -1

    # Find the start of the HEAD tag
    head_start = re.search(r"<head\b", content, re.IGNORECASE)

    # If the HEAD tag is found, find the end of the HEAD tag
    if head_start:
        head_end = find_head_end(content, head_start.start())
        if head_end != -1:
            return content[: head_start.start()] + content[head_end:]
    return content


def safe_remove_body_tags(content):

    # Remove opening BODY tag
    content = re.sub(r"<body[^>]*>", "", content, flags=re.IGNORECASE)

    # Remove closing BODY tag
    content = re.sub(r"</body\s*>", "", content, flags=re.IGNORECASE)
    return content


def safe_html_cleanup(content):

    # Safely remove DOCTYPE, HTML, HEAD, and BODY tags
    content = safe_remove_doctype(content)
    content = safe_remove_html_tags(content)
    content = safe_remove_head(content)
    content = safe_remove_body_tags(content)
    return content


def _combine_html_contents(*html_contents, title="Combined View", nested=False):
    """
    Combine multiple HTML contents into a single HTML document with a dropdown to select which content to display.
    Handles nested content and avoids ID conflicts.
    """
    instance_id = str(uuid.uuid4())[:8]
    combined_html = ""
    if not nested:
        combined_html += f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #2c3e50;
                    padding-bottom: 10px;
                    font-size: 24px;
                    margin-top: 10px;
                    margin-bottom: 15px;
                }}
                .content-selector {{
                    margin-bottom: 20px;
                    padding: 10px;
                    font-size: 16px;
                }}
                .content {{
                    display: none;
                }}
                .content.active {{
                    display: block;
                }}
            </style>
        </head>
        <body>
        """

    # Add a dropdown to select which content to display
    combined_html += f"""
    <div class="combined-content-{instance_id}">
        <h2>{title}</h2>
        <select class="content-selector" onchange="showContent_{instance_id}(this.value)">
            <option value="">Select a section</option>
    """

    # Add an option for each section
    for i, (section_title, _) in enumerate(html_contents):
        combined_html += f'<option value="content_{instance_id}_{i}">{section_title}</option>'
    combined_html += """
        </select>
    """

    # Add the content for each section
    for i, (section_title, content) in enumerate(html_contents):

        # Make sure that there are no nested HTML, HEAD, or BODY tags

        # content = re.sub(r'<!DOCTYPE.*?>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # content = re.sub(r'<html.*?>|</html>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # content = re.sub(r'<head.*?>.*?</head>', '', content, flags=re.DOTALL | re.IGNORECASE) # This line randomly hangs

        # content = re.sub(r'<body.*?>|</body>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = safe_html_cleanup(content)

        # Add the content with a unique ID
        combined_html += f"""
        <div id="content_{instance_id}_{i}" class="content">
            {content}
        </div>
        """

    # Add JavaScript to show the selected content
    combined_html += f"""
    <script>
        function showContent_{instance_id}(id) {{
            var contents = document.querySelectorAll('.combined-content-{instance_id} .content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}
            if (id) {{
                document.getElementById(id).classList.add('active');
            }}
        }}
    </script>
    </div>
    """

    # Close the HTML tags if not nested
    if not nested:
        combined_html += """
        </body>
        </html>
        """
    return combined_html


def feature_centric_view(feature, short=False, extra_prompts=None):
    if short:

        # Just add the top 5 max activating examples
        max_activation_examples = feature.get_max_activating(8)

        # Create a dictionary with these top activations
        display_dict = {"Top Activations": [ex.get_tokens_feature_lists(feature) for ex in max_activation_examples]}
    else:

        # Get the top 15 examples that maximally activate this feature
        max_activation_examples = feature.get_max_activating(15)

        # Create a dictionary with these top activations
        display_dict = {"Top Activations": [ex.get_tokens_feature_lists(feature) for ex in max_activation_examples]}

        # Get 8 quantiles with 5 examples each
        quantiles = feature.get_quantiles(8, 5)

        # Iterate through the quantiles in reverse order (highest to lowest)
        for i, (lower, upper) in enumerate(list(quantiles)[::-1]):

            # Get examples for this quantile
            examples = quantiles[(lower, upper)]

            # Add examples for this quantile to the display dictionary

            # The key is formatted as "Interval {i} - (lower_bound, upper_bound)"
            display_dict[f"Interval {i} - ({lower:.2f}, {upper:.2f})"] = [
                ex.get_tokens_feature_lists(feature) for ex in examples
            ]

    # Generate HTML content for the categorized examples
    categorized_examples_html = _generate_categorized_examples(display_dict)

    # Generate HTML content for the logits table
    top_logits, bottom_logits = feature.get_logits(8)
    logits_table_html = _generate_logits_table(bottom_logits, top_logits)

    # If examples are provided, add their prompt centric views to this viz
    if extra_prompts:
        assert isinstance(extra_prompts, dict)
        prompt_htmls = []
        for example_id, example in extra_prompts.items():
            prompt_html = prompt_centric_view_feature(example, feature)
            prompt_htmls.append((example_id, prompt_html))
        prompt_html = _combine_html_contents(*prompt_htmls, title="Prompt-Centric Views", nested=True)
        combined_html = _combine_html_contents(
            ("Categorized Examples", categorized_examples_html),
            ("Logits Analysis", logits_table_html),
            ("Prompt-Centric Views", prompt_html),
            title=f"Feature-Centric View - Feature {feature.feature_id}, Hook Point {feature.hook_name}",
        )
    else:
        combined_html = _combine_html_contents(
            ("Categorized Examples", categorized_examples_html),
            ("Logits Analysis", logits_table_html),
            title=f"Feature-Centric View - Feature {feature.feature_id}, Hook Point {feature.hook_name}",
        )
    return _light_mode(combined_html)


def _generate_prompt_centric_html(examples, title, get_tokens_and_acts_func, use_orange_highlight):

    # Generate HTML content for the prompt-centric view

    # Add a title if provided
    if title is not None:
        html_content = f"<h2>{title}</h2>"
    else:
        html_content = ""

    # Add a container for the examples
    html_content += "<div class='examples-container'>"

    # Iterate through each example
    for i, example in enumerate(examples):

        # Get the tokens and activations for the example
        str_tokens, acts = get_tokens_and_acts_func(example)
        html_content += f'<div class="example"><div class="example-label">Example {i}:</div>'

        # Generate highlighted HTML for the example
        html_content += _generate_highlighted_html(str_tokens, acts, use_orange_highlight)
        html_content += "</div>"
    html_content += "</div>"
    return html_content


def _generate_prompt_centric_view(examples, title, get_tokens_and_acts_func, use_orange_highlight):

    # Check if examples is a single example or a list of examples
    if not isinstance(examples, list) and not isinstance(examples, dict):
        examples = [examples]

    # Generate the prompt-centric view HTML content
    if isinstance(examples, dict):
        html_contents = []
        for category, category_examples in examples.items():
            category_html = _generate_prompt_centric_html(
                category_examples, None, get_tokens_and_acts_func, use_orange_highlight
            )
            html_contents.append((category, category_html))
        return _light_mode(_combine_html_contents(*html_contents, title=title))
    else:
        return _light_mode(
            _generate_prompt_centric_html(examples, title, get_tokens_and_acts_func, use_orange_highlight)
        )


def prompt_centric_view_feature(examples, feature):

    # Generate the title for the prompt-centric view
    title = f"Prompt-Centric View - Feature {feature.feature_id}, Hook Point {feature.hook_name}"

    # Define a function to get the tokens and activations for a given example
    get_tokens_and_acts = lambda ex: (
        ex.str_tokens,
        ex.get_feature_activation(feature).tolist(),
    )

    # Generate the prompt-centric view HTML content
    return _generate_prompt_centric_view(examples, title, get_tokens_and_acts, True)


def prompt_centric_view_direction(examples, encoder, hook_name, direction):

    # Generate the title for the prompt-centric view
    title = f"Prompt-Centric View - Hook Point {hook_name}"

    # Define a function to get the tokens and activations for a given example
    get_tokens_and_acts = lambda ex: ex.get_tokens_direction_scores(encoder, direction, hook_name)

    # Generate the prompt-centric view HTML content
    return _generate_prompt_centric_view(examples, title, get_tokens_and_acts, False)


def prompt_centric_view_generic(token_act_pairs, title="Generic Prompt-Centric View"):
    # Display prompt-centric view for a list of annotated examples
    # Check if the input is a list of lists of (token, activation) tuples
    if not isinstance(token_act_pairs, list) or not all(
        isinstance(example, list) and all(isinstance(item, tuple) for item in example) for example in token_act_pairs
    ):
        raise ValueError(
            f"Input should be a list of lists of (token, activation) tuples., is instead {type(token_act_pairs)}"
        )

    # Separate tokens and activations for each example
    examples = []
    for example in token_act_pairs:
        tokens, acts = zip(*example)
        examples.append((list(tokens), list(acts)))

    # Define a function to get the tokens and activations for a given example
    def get_tokens_and_acts(example):
        return example

    # Generate the prompt-centric view HTML content
    return _generate_prompt_centric_view(examples, title, get_tokens_and_acts, False)


def prompt_centric_view_generic_dict(token_act_pairs_dict, title="Generic Prompt-Centric View"):
    # Display prompt-centric view for a list of annotated examples
    html_contents = []
    for split_name, token_act_pairs in token_act_pairs_dict.items():
        html_contents.append((split_name, prompt_centric_view_generic(token_act_pairs, split_name)))
    return _light_mode(_combine_html_contents(*html_contents, title=title, nested=True))
