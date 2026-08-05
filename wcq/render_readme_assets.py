"""Regenerate the static images used in the top-level README.

Kept as a script rather than a one-off so the README image can never drift
from what the code actually produces: rerun this and commit the result.

Requires `kaleido` for static image export, which is deliberately NOT in
requirements.txt since nothing at runtime needs it:

    pip install kaleido

Usage: python render_readme_assets.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def render_bracket() -> None:
    import streamlit_app as app
    from src.viz import charts

    rounds_data, champion, mc_survival, accuracy = app.build_real_bracket()
    fig = charts.bracket_chart(rounds_data, champion, mc_survival, dark_mode=False)
    fig.update_layout(
        width=2400, height=1300, paper_bgcolor="white", plot_bgcolor="white",
        title=dict(
            text="2026 World Cup knockout bracket: model's own picks vs. what actually happened"
                 f"<br><sub>{accuracy} correct · champion ({champion}) called correctly · "
                 f"trained only on pre-tournament data</sub>",
            x=0.5, xanchor="center", font=dict(size=26)),
        margin=dict(t=130, b=50, l=50, r=50),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "bracket.png"
    fig.write_image(str(out), scale=1.5)
    print(f"wrote {out}  ({accuracy}, champion={champion})")


if __name__ == "__main__":
    render_bracket()
