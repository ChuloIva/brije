"""
Matrix-style cascading probe visualization

Tokens and activations cascade down the screen Matrix-style
"""

from gpu_utils import configure_amd_gpu
configure_amd_gpu()

import sys
from pathlib import Path
from typing import List
import time
from dataclasses import dataclass

# Add paths
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
import rich.box

from streaming_probe_inference import StreamingProbeInferenceEngine
from visualization_utils import (
    confidence_to_color,
    format_confidence,
    truncate_text,
    get_emoji_for_confidence
)


@dataclass
class CascadeEntry:
    """Single entry in the cascade"""
    token_pos: int
    token_text: str
    activations: List[tuple]  # (action_name, confidence, layer)
    timestamp: float
    age: int = 0  # How many frames it's been visible


class MatrixCascadeViewer:
    """Matrix-style cascading visualization"""

    def __init__(self, max_history: int = 15, fade_speed: int = 1):
        """
        Initialize viewer

        Args:
            max_history: Maximum entries to keep visible
            fade_speed: How fast entries fade (higher = faster)
        """
        self.console = Console()
        self.max_history = max_history
        self.fade_speed = fade_speed
        self.cascade: List[CascadeEntry] = []
        self.paused = False
        self.speed = 1.0
        self.total_tokens = 0
        self.current_token = 0

    def add_entry(self, entry: CascadeEntry):
        """Add new entry to cascade"""
        self.cascade.insert(0, entry)
        # Remove old entries
        if len(self.cascade) > self.max_history:
            self.cascade = self.cascade[:self.max_history]
        self.current_token = entry.token_pos + 1

    def age_entries(self):
        """Age all entries"""
        for entry in self.cascade:
            entry.age += self.fade_speed

    def get_fade_style(self, age: int) -> str:
        """Get style based on age"""
        if age < 2:
            return "bold bright_white"
        elif age < 4:
            return "bright_white"
        elif age < 6:
            return "white"
        elif age < 8:
            return "dim white"
        else:
            return "dim"

    def render_cascade(self) -> Panel:
        """Render the cascade display"""
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True
        )

        table.add_column("Indicator", width=3, justify="left")
        table.add_column("Token", width=20)
        table.add_column("Activations", ratio=1)

        # Add entries
        for i, entry in enumerate(self.cascade):
            fade_style = self.get_fade_style(entry.age)

            # Indicator (arrow that fades)
            if i == 0:
                indicator = Text("▼", style="bold red")
            elif entry.age < 3:
                indicator = Text("↓", style="yellow")
            elif entry.age < 6:
                indicator = Text("⇣", style="dim yellow")
            else:
                indicator = Text("↓", style="dim")

            # Token info
            token_info = Text()
            token_info.append(f"[{entry.token_pos:3d}] ", style=f"dim {fade_style}")
            token_info.append(f"{truncate_text(entry.token_text, 12)}", style=f"bold {fade_style}")

            # Activations
            activations_text = Text()
            for j, (action, conf, layer_count) in enumerate(entry.activations[:3]):  # Top 3
                if j > 0:
                    activations_text.append(" | ")

                emoji = get_emoji_for_confidence(conf)
                color = confidence_to_color(conf)
                action_short = truncate_text(action, 16)

                activations_text.append(f"{emoji} ", style=color)
                activations_text.append(f"{action_short} ", style=f"{color} {fade_style}")
                activations_text.append(f"{format_confidence(conf, 5)}", style=f"bold {color}")
                activations_text.append(f" ({layer_count}L)", style=f"dim {fade_style}")

            # Add row
            table.add_row(indicator, token_info, activations_text)

        # Progress indicator
        if self.total_tokens > 0:
            progress_pct = (self.current_token / self.total_tokens) * 100
            progress_text = f"Token {self.current_token}/{self.total_tokens} ({progress_pct:.0f}%)"
        else:
            progress_text = f"Token {self.current_token}"

        title_style = "bold green" if not self.paused else "bold yellow"
        title_text = f"🌊 MATRIX CASCADE {progress_text}"

        return Panel(
            table,
            title=title_text,
            border_style=title_style,
            box=rich.box.DOUBLE
        )

    def render_controls(self) -> Panel:
        """Render control panel"""
        controls = Text()

        status = "⏸  PAUSED" if self.paused else "▶  PLAYING"
        status_style = "yellow" if self.paused else "green"
        controls.append(status, style=f"bold {status_style}")
        controls.append("  |  ")

        controls.append(f"Speed: {'█' * int(self.speed * 5)}{'░' * (5 - int(self.speed * 5))}")
        controls.append("  |  ")

        controls.append("[Space]", style="cyan")
        controls.append(" Pause  ")
        controls.append("[+/-]", style="cyan")
        controls.append(" Speed  ")
        controls.append("[Q]", style="cyan")
        controls.append(" Quit")

        return Panel(Align.center(controls), border_style="blue")

    def create_layout(self) -> Layout:
        """Create the overall layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="cascade", ratio=8),
            Layout(name="controls", size=3)
        )

        layout["cascade"].update(self.render_cascade())
        layout["controls"].update(self.render_controls())

        return layout


def animate_streaming_inference(
    engine: StreamingProbeInferenceEngine,
    text: str,
    threshold: float = 0.1,
    delay: float = 0.3
):
    """
    Animate probe inference with Matrix-style cascade

    Args:
        engine: Inference engine
        text: Text to analyze
        threshold: Activation threshold
        delay: Delay between tokens (seconds)
    """
    viewer = MatrixCascadeViewer(max_history=20, fade_speed=1)
    console = Console()

    # Get tokens
    inputs = engine.tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids'].to(engine.device)
    tokens = engine.tokenizer.convert_ids_to_tokens(input_ids[0])

    viewer.total_tokens = len(tokens)

    # Pre-compute all activations
    console.print("\n[bold cyan]Computing activations...[/bold cyan]")
    predictions = engine.predict_streaming(
        text,
        top_k=50,
        threshold=threshold,
        show_realtime=False
    )

    console.print(f"[bold green]✓[/bold green] Computed {len(predictions)} predictions\n")

    # Aggregate predictions by action
    aggregated_predictions = engine.aggregate_predictions(predictions, threshold=threshold)

    # Prepare animation data - show layer distribution for each token
    animation_data = []
    for token_pos in range(len(tokens)):
        # Collect activations for this token, grouped by action
        from collections import defaultdict
        action_activations = defaultdict(lambda: {'layers': [], 'max_conf': 0.0})

        for pred in predictions:
            if token_pos < len(pred.token_activations):
                tok_act = pred.token_activations[token_pos]
                if tok_act.is_active:
                    action_activations[pred.action_name]['layers'].append(tok_act.layer)
                    action_activations[pred.action_name]['max_conf'] = max(
                        action_activations[pred.action_name]['max_conf'],
                        tok_act.confidence
                    )

        # Format activations to show layer count
        activations = []
        for action_name, data in action_activations.items():
            layer_count = len(data['layers'])
            activations.append((
                action_name,
                data['max_conf'],
                layer_count  # Now passing layer count instead of single layer
            ))

        # Sort by layer count first, then confidence
        activations.sort(key=lambda x: (x[2], x[1]), reverse=True)

        animation_data.append({
            'token_pos': token_pos,
            'token_text': tokens[token_pos],
            'activations': activations,
            'timestamp': token_pos * delay
        })

    # Animate with keyboard control
    with Live(viewer.create_layout(), console=console, refresh_per_second=10) as live:
        paused = False

        def on_space():
            nonlocal paused
            paused = not paused
            viewer.paused = paused

        def on_plus():
            viewer.speed = min(viewer.speed + 0.2, 2.0)

        def on_minus():
            viewer.speed = max(viewer.speed - 0.2, 0.2)

        # Try to set up keyboard handlers (may not work in all terminals)
        try:
            import keyboard  # For keyboard input
            keyboard.on_press_key('space', lambda _: on_space())
            keyboard.on_press_key('+', lambda _: on_plus())
            keyboard.on_press_key('-', lambda _: on_minus())
        except ImportError:
            # Keyboard control not available
            pass

        # Animate
        for data in animation_data:
            if not paused:
                entry = CascadeEntry(
                    token_pos=data['token_pos'],
                    token_text=data['token_text'],
                    activations=data['activations'],
                    timestamp=data['timestamp']
                )
                viewer.add_entry(entry)

            # Age existing entries
            viewer.age_entries()

            # Update display
            live.update(viewer.create_layout())

            # Delay (adjusted by speed)
            time.sleep(delay / viewer.speed if not paused else 0.1)

    console.print("\n[bold green]✓ Animation complete![/bold green]\n")


def animate_simpler(
    engine: StreamingProbeInferenceEngine,
    text: str,
    threshold: float = 0.1,
    delay: float = 0.5
):
    """
    Simpler animation without keyboard controls

    Args:
        engine: Inference engine
        text: Text to analyze
        threshold: Activation threshold
        delay: Delay between tokens (seconds)
    """
    viewer = MatrixCascadeViewer(max_history=20, fade_speed=1)
    console = Console()

    # Get tokens
    inputs = engine.tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids'].to(engine.device)
    tokens = engine.tokenizer.convert_ids_to_tokens(input_ids[0])

    viewer.total_tokens = len(tokens)

    # Pre-compute all activations
    console.print("\n[bold cyan]Computing activations...[/bold cyan]")
    predictions = engine.predict_streaming(
        text,
        top_k=50,
        threshold=threshold,
        show_realtime=False
    )

    # Aggregate predictions by action
    aggregated_predictions = engine.aggregate_predictions(predictions, threshold=threshold)

    console.print(f"[bold green]✓[/bold green] Computed {len(predictions)} predictions")
    console.print(f"[bold cyan]✓[/bold cyan] Aggregated into {len(aggregated_predictions)} actions\n")
    console.print("[dim]Starting animation in 2 seconds...[/dim]")
    time.sleep(2)

    # Animate
    with Live(viewer.create_layout(), console=console, refresh_per_second=8) as live:
        for token_pos in range(len(tokens)):
            # Collect activations for this token, grouped by action
            from collections import defaultdict
            action_activations = defaultdict(lambda: {'layers': [], 'max_conf': 0.0})

            for pred in predictions:
                if token_pos < len(pred.token_activations):
                    tok_act = pred.token_activations[token_pos]
                    if tok_act.is_active:
                        action_activations[pred.action_name]['layers'].append(tok_act.layer)
                        action_activations[pred.action_name]['max_conf'] = max(
                            action_activations[pred.action_name]['max_conf'],
                            tok_act.confidence
                        )

            # Format activations to show layer count
            activations = []
            for action_name, data in action_activations.items():
                layer_count = len(data['layers'])
                activations.append((
                    action_name,
                    data['max_conf'],
                    layer_count  # Now passing layer count instead of single layer
                ))

            # Sort by layer count first, then confidence
            activations.sort(key=lambda x: (x[2], x[1]), reverse=True)

            # Add entry
            entry = CascadeEntry(
                token_pos=token_pos,
                token_text=tokens[token_pos],
                activations=activations,
                timestamp=token_pos * delay
            )
            viewer.add_entry(entry)

            # Age existing entries
            viewer.age_entries()

            # Update display
            live.update(viewer.create_layout())

            # Delay
            time.sleep(delay)

    console.print("\n[bold green]✓ Animation complete![/bold green]\n")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Matrix-style probe animation")
    parser.add_argument("--probes-dir", type=str, required=True, help="Probes directory")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Model name")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--threshold", type=float, default=0.1, help="Activation threshold")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between tokens (seconds)")

    args = parser.parse_args()

    # Initialize engine
    print("Loading model and probes...")
    engine = StreamingProbeInferenceEngine(
        probes_base_dir=Path(args.probes_dir),
        model_name=args.model,
        verbose=True
    )

    # Test text
    if args.text:
        text = args.text
    else:
        text = "After receiving feedback, I began reconsidering my approach and analyzing the problem more carefully."

    print(f"\nText: '{text}'")

    # Run animation (use simpler version without keyboard for compatibility)
    animate_simpler(engine, text, threshold=args.threshold, delay=args.delay)


if __name__ == "__main__":
    main()
