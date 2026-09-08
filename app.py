"""PLM Factory — multi-course student drill + a constrained live debug tab for the
multi-agent PLM pipeline (Retrieval/PLM/Research agents; Geography, Python, Chess).

Timed, interleaved classification trials with ARTS adaptive sequencing. The student
drill calls each course's generator directly (dataset oracle) with NO
agent/LLM in the per-trial loop — perceptual learning requires seconds-fast trials, and
the PLM Agent's LLM round trips (2-30s observed) would break that. The agent's LLM
reasoning belongs in the separate debug tab / offline item pre-generation, not live
per-trial sequencing.

The agent debug tab deliberately does NOT use smolagents' built-in `GradioUI` (a raw
free-text chat into a `CodeAgent`): on a public Space that would let any visitor prompt
the agent into executing arbitrary generated Python in the container. Fixed
course/concept dropdowns keep the same live agentic pipeline (and its code trace)
visible without that open-ended attack surface.
"""

import json
import re
import time

import gradio as gr
from smolagents import ActionStep

from src.generators import geography, python, chess
from src.plm_core.arts import ArtsTracker, CategoryState
from src.plm_core.plm_agent import build_plm_agent

MAX_CHOICES = 5  # widest choice set across courses

# Shared by both the student drill and the agent debug tab below.
COURSES = {
    "GEOGRAPHY": {
        "make_item": geography.make_item,
        "categories": lambda: geography.CATEGORIES,
        "blurb": (
            "**GeoSense** — answer geography questions, identify spatial patterns, "
            "and recognize geographic features. Answer keys are dataset-verified."
        ),
    },
    "PYTHON": {
        "make_item": python.make_item,
        "categories": lambda: python.CATEGORIES,
        "blurb": (
            "**CodeSense** — predict code output, choose data structures, and analyze "
            "algorithm complexity. Answer keys are dataset-verified."
        ),
    },
    "CHESS": {
        "make_item": chess.make_item,
        "categories": lambda: chess.CATEGORIES,
        "blurb": (
            "**ChessSense** — identify tactical patterns, find best moves, and apply "
            "endgame techniques. Answer keys are Lichess-verified."
        ),
    },
}


def new_session(course: str) -> ArtsTracker:
    categories = COURSES[course]["categories"]()
    return ArtsTracker([CategoryState(name=n, rt_threshold_s=s["rt_threshold_s"]) for n, s in categories.items()])


def _create_map_figure(map_data: dict):
    """Create a Plotly map figure from map data."""
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add a marker at the location
        fig.add_trace(go.Scattermap(
            lat=[map_data["marker_lat"]],
            lon=[map_data["marker_lon"]],
            mode='markers',
            marker=go.scattermap.Marker(
                size=14,
                color='red',
                symbol='star',
            ),
            text=[map_data["marker_name"]],
            textposition="top center",
            textfont=dict(size=12, color="black"),
            name="Location",
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=map_data["center_lat"], lon=map_data["center_lon"]),
                zoom=map_data.get("zoom", 3),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=350,
            showlegend=False,
        )
        
        return fig
    except Exception as e:
        # Fallback: return empty figure
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text=f"Map error: {str(e)}", showarrow=False, x=0.5, y=0.5)],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig


def _download_image_from_url(url: str):
    """Download an image from URL and return PIL Image."""
    # Simple cache to avoid repeated downloads
    if not hasattr(_download_image_from_url, "_cache"):
        _download_image_from_url._cache = {}
    
    if url in _download_image_from_url._cache:
        return _download_image_from_url._cache[url]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(3):
        try:
            import requests
            from PIL import Image
            import io
            
            response = requests.get(url, timeout=10, stream=True, headers=headers)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            # Resize if too large
            if image.width > 800 or image.height > 800:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            _download_image_from_url._cache[url] = image
            return image
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(2 * (attempt + 1))
            else:
                # Fallback: create placeholder image
                from PIL import Image, ImageDraw
                image = Image.new('RGB', (400, 300), color='lightblue')
                draw = ImageDraw.Draw(image)
                draw.text((50, 100), "Image unavailable", fill='black')
                draw.text((50, 120), "Check your connection", fill='gray')
                return image


def _create_chess_board_image(fen: str):
    """Create a chess board image from FEN notation."""
    try:
        import chess
        import chess.svg
        from PIL import Image
        import io
        import base64
        
        # Create board from FEN
        board = chess.Board(fen)
        
        # Generate SVG
        svg_str = chess.svg.board(board, size=350, coordinates=True)
        
        # Convert SVG to PNG using cairosvg
        try:
            from cairosvg import svg2png
            png_data = svg2png(bytestring=svg_str.encode('utf-8'))
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(png_data))
            return image
        except ImportError:
            # Fallback: create a simple text image
            image = Image.new('RGB', (350, 350), color='white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Draw board representation
            draw.text((10, 10), "Chess Board", fill='black')
            draw.text((10, 30), f"FEN: {fen[:40]}...", fill='black')
            draw.text((10, 50), "Install cairosvg for full rendering", fill='gray')
            
            return image
    except Exception as e:
        # Fallback: create error image
        from PIL import Image, ImageDraw
        image = Image.new('RGB', (350, 350), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Chess Board Error", fill='red')
        draw.text((10, 30), str(e)[:50], fill='black')
        return image


def next_trial(course: str, tracker: ArtsTracker | None):
    if tracker is None:
        tracker = new_session(course)
    if tracker.all_retired() and tracker.trial > 0:
        done = "🎉 **All categories retired — session mastered!** Press *New session* to restart."
        return (
            tracker,
            None,
            gr.Image(visible=False),
            gr.Code(visible=False),
            gr.Plot(visible=False),
            done,
            *[gr.Button(visible=False)] * MAX_CHOICES,
            tracker.summary(),
            None,
        )
    
    cat = tracker.next_category()
    item = COURSES[course]["make_item"](cat)
    
    buttons = [
        gr.Button(value=item["choices"][i], visible=True, interactive=True) if i < len(item["choices"])
        else gr.Button(visible=False)
        for i in range(MAX_CHOICES)
    ]
    
    # Handle different stimulus types
    stimulus = item.get("stimulus", {})
    stimulus_type = stimulus.get("type", "text")
    
    # Default values
    img = gr.Image(visible=False)
    code = gr.Code(visible=False)
    plot = gr.Plot(visible=False)
    
    if stimulus_type == "image" and "image_url" in stimulus:
        # Geography - show image from URL
        image = _download_image_from_url(stimulus["image_url"])
        img = gr.Image(value=image, visible=True)
    elif stimulus_type == "map" and "map_data" in stimulus:
        # Geography - show map (legacy support)
        fig = _create_map_figure(stimulus["map_data"])
        plot = gr.Plot(value=fig, visible=True)
    elif stimulus_type == "code" and "content" in stimulus:
        # Python - show code with syntax highlighting
        code = gr.Code(value=stimulus["content"], language="python", visible=True)
    elif stimulus_type == "chess_board" and "fen" in stimulus:
        # Chess - show board image
        img = gr.Image(value=_create_chess_board_image(stimulus["fen"]), visible=True)
    elif stimulus_type == "pil_image":
        # Legacy PIL image support
        img = gr.Image(value=stimulus.get("image"), visible=True)
    
    return (
        tracker,
        item,
        img,
        code,
        plot,
        f"### {item['prompt']}",
        *buttons,
        tracker.summary(),
        time.time(),
    )


def answer(idx: int, course: str, tracker: ArtsTracker, item: dict, t0: float):
    if item is None or t0 is None:
        return tracker, "Press **Next trial** to begin.", tracker.summary() if tracker else []
    rt = time.time() - t0
    correct = idx == item["correct"]
    tracker.record(item["subcategory"], correct, rt)
    verdict = "✅ **Correct**" if correct else f"❌ **Incorrect** — answer: *{item['choices'][item['correct']]}*"
    threshold = COURSES[course]["categories"]()[item["subcategory"]]["rt_threshold_s"]
    speed = "⚡ fast enough to count toward mastery" if rt < threshold else f"🐢 over the {threshold:.0f}s fluency threshold"
    msg = f"{verdict}  ·  {rt:.1f}s ({speed})\n\n{item['feedback']}\n\n*Press **Next trial** to continue.*"
    return tracker, msg, tracker.summary()


def _switch_course(course: str):
    tr, item, img_u, code_u, plot_u, prompt_u, *btns, stats_u, t0_u = next_trial(course, new_session(course))
    return COURSES[course]["blurb"], tr, item, img_u, code_u, plot_u, prompt_u, *btns, stats_u, t0_u


with gr.Blocks() as student_demo:
    gr.Markdown("## 📚 Student Drill — Choose your domain and start training!")
    course_dd = gr.Dropdown(choices=list(COURSES), value="GEOGRAPHY", label="Course")
    blurb_md = gr.Markdown(COURSES["GEOGRAPHY"]["blurb"])
    gr.Markdown(
        "Categories retire after 4 consecutive fast-and-correct answers (ARTS adaptive "
        "sequencing — accuracy **and** response time; Kellman, Massey & Son 2010)."
    )
    tracker_s, item_s, t0_s = gr.State(None), gr.State(None), gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            # Visual components for different stimulus types
            img = gr.Image(visible=False, show_label=False)
            code = gr.Code(visible=False, label="Python Code", language="python")
            plot = gr.Plot(visible=False, label="Geography Map")
            
            prompt_md = gr.Markdown("Press **Next trial** to begin.")
            with gr.Row():
                btns = [gr.Button(visible=False) for _ in range(MAX_CHOICES)]
            feedback_md = gr.Markdown("")
            with gr.Row():
                next_btn = gr.Button("Next trial ▶", variant="primary")
                reset_btn = gr.Button("New session ⟳")
        with gr.Column(scale=2):
            gr.Markdown("### Session progress")
            stats = gr.Dataframe(interactive=False)

    trial_outputs = [tracker_s, item_s, img, code, plot, prompt_md, *btns, stats, t0_s]

    next_btn.click(next_trial, [course_dd, tracker_s], trial_outputs).then(lambda: "", None, feedback_md)
    reset_btn.click(lambda c: next_trial(c, new_session(c)), [course_dd], trial_outputs).then(lambda: "", None, feedback_md)
    course_dd.change(_switch_course, [course_dd], [blurb_md, *trial_outputs]).then(lambda: "", None, feedback_md)
    for i, b in enumerate(btns):
        b.click(
            lambda tr, c, it, t0, i=i: answer(i, c, tr, it, t0),
            [tracker_s, course_dd, item_s, t0_s],
            [tracker_s, feedback_md, stats],
        )


# ---------------------------------------------------------------------------
# Agent debug tab — live PLM Agent runs, gated to fixed dropdowns (see module
# docstring for why this isn't a free-text chat).
# ---------------------------------------------------------------------------

_COURSE_CONCEPTS = {course: list(cfg["categories"]().keys()) for course, cfg in COURSES.items()}

# Built once at Space startup, not per-click: each build_plm_agent() call constructs a
# fresh CodeAgent + retrieval sub-agent, which is wasted work if repeated every request.
_PLM_AGENTS = {course: build_plm_agent(course) for course in _COURSE_CONCEPTS}


def _update_concept_choices(course: str):
    concepts = _COURSE_CONCEPTS[course]
    return gr.Dropdown(choices=concepts, value=concepts[0])


def _summarize_trace(agent) -> str:
    lines = []
    for step in agent.memory.steps:
        if isinstance(step, ActionStep) and step.code_action:
            lines.append(f"--- step {step.step_number} ---\n{step.code_action}")
    return "\n\n".join(lines) or "(no steps captured)"


_IMAGE_PATH_RE = re.compile(r"data/items/\S+?\.png")


def run_plm_agent(course: str, concept: str):
    agent = _PLM_AGENTS[course]
    task = (
        f"Generate one {course} item for concept {concept!r}. Call the sanctioned "
        "generator tool, then report only image_path and correct_answer."
    )
    result = agent.run(task, reset=True)  # reset=True: no cross-visitor memory buildup
    trace = _summarize_trace(agent)

    # The final packaging step's exact shape isn't guaranteed by the prompt alone — observed
    # a dict, a plain [image_path, correct_answer] list, AND a free-form descriptive string
    # across different live runs/courses. Regexing the path out of the stringified result is
    # robust to all three rather than trying to enumerate every container shape.
    result_str = result if isinstance(result, str) else json.dumps(result, default=str)
    match = _IMAGE_PATH_RE.search(result_str)
    image_path = match.group(0) if match else None
    answer_json = result_str if isinstance(result, str) else json.dumps(result, indent=2, default=str)
    return image_path, answer_json, trace


with gr.Blocks() as agent_debug:
    gr.Markdown(
        "# 🤖 PLM Agent — live debug\n"
        "Each click runs a real `smolagents` `CodeAgent` (Qwen2.5-Coder-32B-Instruct via the "
        "HF Inference API) that decides the item parameters and calls the course's sanctioned "
        "generator tool — verified datasets compute the actual answer key. The code trace below "
        "shows exactly what the agent ran, so you can verify it never asserts an answer itself."
    )
    with gr.Row():
        course_dd = gr.Dropdown(choices=list(_COURSE_CONCEPTS), value="GEOGRAPHY", label="Course")
        concept_dd = gr.Dropdown(
            choices=_COURSE_CONCEPTS["GEOGRAPHY"], value=_COURSE_CONCEPTS["GEOGRAPHY"][0], label="Concept"
        )
    run_btn = gr.Button("Generate item ▶", variant="primary")
    with gr.Row():
        with gr.Column():
            out_img = gr.Image(label="Generated stimulus", type="filepath")
            out_json = gr.Code(label="Answer key", language="json")
        out_trace = gr.Code(label="Agent code trace", language="python")

    course_dd.change(_update_concept_choices, [course_dd], [concept_dd])
    run_btn.click(run_plm_agent, [course_dd, concept_dd], [out_img, out_json, out_trace])


with gr.Blocks(title="PLM Factory") as demo:
    gr.Markdown("# 🧠 PLM Factory — Adaptive Perceptual Learning Drills")
    gr.Markdown(
        "Train your intuition with timed drills across **Geography**, **Python**, and **Chess**. "
        "Categories retire as you master them (4 consecutive fast-and-correct answers)."
    )
    with gr.Tab("📚 Student Demo"):
        student_demo.render()
    with gr.Tab("🤖 Agent Debug (dev)"):
        agent_debug.render()

if __name__ == "__main__":
    demo.launch()
