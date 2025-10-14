"""
Interactive Terminal UI for Probe Visualization

Full-screen TUI with navigable panels using Textual framework
"""

from gpu_utils import configure_amd_gpu
configure_amd_gpu()

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import asyncio

# Add paths
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, DataTable, Label
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table as RichTable
from rich.panel import Panel

from streaming_probe_inference import StreamingProbeInferenceEngine, StreamingPrediction, AggregatedPrediction
from visualization_utils import (
    confidence_to_color,
    format_confidence,
    truncate_text,
    create_sparkline,
    get_emoji_for_confidence,
    format_timestamp,
    get_activation_symbol,
    get_category_color
)
from action_categories import get_action_category, get_category_tag


class TokenDisplay(Static):
    """Display tokens with highlighting for selected token"""

    selected_token = reactive(0)

    def __init__(self, tokens: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = tokens
        self.predictions: Optional[List[StreamingPrediction]] = None
        self.display_threshold: float = 0.1

    def set_predictions(self, predictions: List[StreamingPrediction]):
        """Set predictions data"""
        self.predictions = predictions
        self.refresh()

    def render(self) -> Panel:
        """Render token stream with highlighting"""
        text = Text()

        for i, token in enumerate(self.tokens):
            # Check if this token has activations
            has_activation = False
            if self.predictions:
                for pred in self.predictions[:5]:  # Top 5
                    # Handle both AggregatedPrediction and StreamingPrediction
                    token_acts = pred.token_activations if hasattr(pred, 'token_activations') else (pred.layer_predictions[0].token_activations if pred.layer_predictions else [])
                    if i < len(token_acts):
                        if token_acts[i].confidence >= self.display_threshold:
                            has_activation = True
                            break

            # Style based on state
            if i == self.selected_token:
                # Selected: bold and highlighted
                text.append(f"[{token}]", style="bold black on yellow")
            elif has_activation:
                # Has activation: colored
                text.append(token, style="bold cyan")
            else:
                # Normal
                text.append(token, style="dim white")

            text.append(" ")

        return Panel(text, title="Token Stream (←/→ to navigate)", border_style="blue")


class ProbeDetails(Static):
    """Display details for selected token"""

    selected_token = reactive(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions: Optional[List[AggregatedPrediction]] = None
        self.tokens: List[str] = []
        self.display_threshold: float = 0.1

    def set_data(self, predictions: List[AggregatedPrediction], tokens: List[str]):
        """Set data"""
        self.predictions = predictions
        self.tokens = tokens
        self.refresh_details()

    def refresh_details(self):
        """Refresh the details display"""
        if not self.predictions or not self.tokens:
            self.update(Panel("No data", title="Probe Details"))
            return

        if self.selected_token >= len(self.tokens):
            self.update(Panel("Invalid token", title="Probe Details"))
            return

        token = self.tokens[self.selected_token]

        # Create table of activations at this token
        table = RichTable(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Probe", style="cyan", width=28)
        table.add_column("Layers", justify="center", width=5)
        table.add_column("Conf", justify="right", width=7)
        table.add_column("Layer Bar", width=12)
        table.add_column("", width=2)

        # Collect activations at this position from ALL predictions
        # Count how many layers are ACTIVE at this specific token position
        activations = []
        max_layers_at_token = 0
        
        for pred in self.predictions:
            # For AggregatedPrediction, check activations across all layers at this token
            if hasattr(pred, 'layer_predictions'):
                # Count active layers at this token position
                # Only check layers that are in the aggregated prediction's active layers list
                active_layers_at_token = 0
                max_conf_at_token = 0.0
                any_active = False
                
                for layer_pred in pred.layer_predictions:
                    # Only consider layers that passed the threshold in aggregation
                    if layer_pred.layer in pred.layers:
                        if self.selected_token < len(layer_pred.token_activations):
                            tok_act = layer_pred.token_activations[self.selected_token]
                            # Check if this layer is active at this specific token
                            if tok_act.confidence >= self.display_threshold:
                                active_layers_at_token += 1
                                any_active = True
                            max_conf_at_token = max(max_conf_at_token, tok_act.confidence)
                
                max_layers_at_token = max(max_layers_at_token, active_layers_at_token)
                
                # Add to activations list
                activations.append((pred.action_name, max_conf_at_token, any_active, active_layers_at_token))
            else:
                # Handle StreamingPrediction
                token_acts = pred.token_activations if hasattr(pred, 'token_activations') else []
                if self.selected_token < len(token_acts):
                    tok_act = token_acts[self.selected_token]
                    layer_count = 1
                    max_layers_at_token = max(max_layers_at_token, layer_count)
                    activations.append((pred.action_name, tok_act.confidence, tok_act.confidence >= self.display_threshold, layer_count))

        # Sort by layer count at this token (descending), then confidence
        activations.sort(key=lambda x: (x[3], x[1]), reverse=True)

        # Add ALL rows (not just top 15)
        for action, conf, is_active, layer_count_at_token in activations:
            color = confidence_to_color(conf)
            # Bar represents layer count at this token position, not overall
            bar_len = int((layer_count_at_token / max(max_layers_at_token, 1)) * 12) if max_layers_at_token > 0 else 0
            bar = "█" * bar_len
            marker = "✓" if is_active else ""

            # Category color and tag
            category = get_action_category(action)
            cat_color = get_category_color(category)
            cat_tag = get_category_tag(category)

            action_short = truncate_text(action, 23)
            table.add_row(
                f"[{cat_color}]{action_short}[/{cat_color}] [{cat_color}]{cat_tag}[/{cat_color}]",
                f"[yellow]{layer_count_at_token}[/yellow]",
                f"[{color}]{format_confidence(conf)}[/{color}]",
                f"[yellow]{bar}[/yellow]",
                f"[green]{marker}[/green]"
            )

        title_text = f"Token [{self.selected_token}]: '{token}' - All {len(activations)} Actions (sorted by layers)"
        self.update(Panel(table, title=title_text, border_style="green"))

    def watch_selected_token(self, new_value: int) -> None:
        """Called when selected_token changes"""
        self.refresh_details()


class ActivationHeatmap(Static):
    """Display heatmap of all activations across layers"""

    selected_token = reactive(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions: Optional[List[AggregatedPrediction]] = None
        self.layer_range = (21, 30)  # Default layer range
        self.display_threshold: float = 0.1

    def set_data(self, predictions: List[AggregatedPrediction], layer_range: Tuple[int, int]):
        """Set data"""
        self.predictions = predictions
        self.layer_range = layer_range
        self.refresh_heatmap()

    def refresh_heatmap(self):
        """Refresh the heatmap display based on selected token"""
        if not self.predictions:
            self.update(Panel("No data", title="Layer Activation Heatmap"))
            return

        # Create heatmap showing ALL actions (rows) vs layers (columns)
        # Show activation at the SELECTED TOKEN position
        table = RichTable(show_header=True, header_style="bold cyan", box=None, padding=(0, 0))

        # Header: layer numbers
        table.add_column("Action", style="cyan", width=30, no_wrap=True)
        table.add_column("Ct", width=3, justify="center", style="yellow")  # Layer count
        for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
            table.add_column(f"L{layer_idx}", width=3, justify="center", style="dim")

        # Group predictions by category
        grouped: Dict[str, List[AggregatedPrediction]] = {}
        for pred in self.predictions:
            cat = get_action_category(pred.action_name)
            grouped.setdefault(cat, []).append(pred)

        # Stable category order
        category_order = ["metacognitive", "analytical", "creative", "emotional", "memory", "other"]

        for category in category_order:
            preds_in_cat = grouped.get(category, [])
            if not preds_in_cat:
                continue

            # Sort within category by active layers at this token (desc)
            def active_layers_for(pred: AggregatedPrediction) -> int:
                count = 0
                for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
                    if layer_idx not in pred.layers:
                        continue
                    layer_pred = next((p for p in pred.layer_predictions if p.layer == layer_idx), None)
                    if layer_pred and self.selected_token < len(layer_pred.token_activations):
                        if layer_pred.token_activations[self.selected_token].confidence >= self.display_threshold:
                            count += 1
                return count

            preds_in_cat.sort(key=lambda p: active_layers_for(p), reverse=True)

            # Category header row
            cat_color = get_category_color(category)
            cat_tag = get_category_tag(category)
            header_cells = [f"[{cat_color}]{category.upper()}[/{cat_color}] [{cat_color}]{cat_tag}[/{cat_color}]", "", *([""] * (self.layer_range[1] - self.layer_range[0] + 1))]
            table.add_row(*header_cells)

            # Rows for predictions in this category
            for pred in preds_in_cat:
                # Action cell with category styling
                action_cell = f"[{cat_color}]{truncate_text(pred.action_name, 25)}[/{cat_color}] [{cat_color}]{cat_tag}[/{cat_color}]"

                # Count active layers at this token
                active_layers_at_token = 0
                layer_cells = []
                for layer_idx in range(self.layer_range[0], self.layer_range[1] + 1):
                    layer_pred = next((p for p in pred.layer_predictions if p.layer == layer_idx), None)
                    if layer_pred and self.selected_token < len(layer_pred.token_activations):
                        tok_act = layer_pred.token_activations[self.selected_token]
                        conf = tok_act.confidence
                        symbol = get_activation_symbol(conf)
                        color = confidence_to_color(conf)
                        layer_cells.append(f"[{color}]{symbol}[/{color}]")
                        if conf >= self.display_threshold and layer_idx in pred.layers:
                            active_layers_at_token += 1
                    else:
                        layer_cells.append("[dim]·[/dim]")

                row = [action_cell, f"[yellow]{active_layers_at_token}[/yellow]"]
                row.extend(layer_cells)
                table.add_row(*row)

        title_text = f"Layer Activation at Token [{self.selected_token}] - All {len(self.predictions)} Actions"
        self.update(Panel(table, title=title_text, border_style="magenta"))

    def watch_selected_token(self, new_value: int) -> None:
        """Called when selected_token changes"""
        self.refresh_heatmap()


class StatsPanel(Static):
    """Display statistics and controls"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions: Optional[List[StreamingPrediction]] = None
        self.tokens: List[str] = []
        self.processing_time: float = 0.0

    def set_data(self, predictions, tokens: List[str], processing_time: float):
        """Set data"""
        self.predictions = predictions
        self.tokens = tokens
        self.processing_time = processing_time
        self.refresh()

    def render(self) -> Panel:
        """Render stats"""
        if not self.predictions:
            return Panel("No data", title="Statistics")

        text = Text()

        # Summary stats
        text.append(f"Tokens: ", style="bold")
        text.append(f"{len(self.tokens)}\n")

        text.append(f"Actions: ", style="bold")
        text.append(f"{len(self.predictions)}\n")

        text.append(f"Time: ", style="bold")
        text.append(f"{self.processing_time:.2f}s\n\n")

        # Top 5 active probes - handle both AggregatedPrediction and StreamingPrediction
        text.append("Top 5 Active:\n", style="bold green")
        for i, pred in enumerate(self.predictions[:5], 1):
            # Check if it's aggregated or regular prediction
            is_aggregated = hasattr(pred, 'layer_count')

            if is_aggregated:
                if pred.is_active:
                    emoji = get_emoji_for_confidence(pred.max_confidence)
                    color = confidence_to_color(pred.max_confidence)
                    text.append(f"  {i}. ", style="dim")
                    text.append(f"{emoji} ", style=color)
                    text.append(f"{truncate_text(pred.action_name, 12)}", style=color)
                    text.append(f" ({pred.layer_count}L)\n", style="dim")
            else:
                if pred.is_active:
                    emoji = get_emoji_for_confidence(pred.confidence)
                    color = confidence_to_color(pred.confidence)
                    text.append(f"  {i}. ", style="dim")
                    text.append(f"{emoji} ", style=color)
                    text.append(f"{truncate_text(pred.action_name, 15)}\n", style=color)

        return Panel(text, title="Statistics", border_style="yellow")


class ProbeViewerApp(App):
    """Interactive TUI for exploring probe activations"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: auto 1fr;
    }

    #token-display {
        column-span: 2;
    }

    #probe-details {
        height: 100%;
        overflow-y: auto;
    }

    #heatmap {
        height: 100%;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("left", "previous_token", "← Prev"),
        Binding("right", "next_token", "→ Next"),
        Binding("home", "first_token", "⇤ First"),
        Binding("end", "last_token", "⇥ Last"),
        Binding("r", "reload", "↻ Reload"),
    ]

    def __init__(
        self,
        predictions: List[AggregatedPrediction],
        tokens: List[str],
        layer_range: Tuple[int, int],
        display_threshold: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.predictions = predictions
        self.tokens = tokens
        self.layer_range = layer_range
        self.display_threshold = display_threshold
        self.selected_token = 0

    def compose(self) -> ComposeResult:
        """Create UI components"""
        yield Header()

        # Token display (top, spans 2 columns)
        self.token_display = TokenDisplay(self.tokens, id="token-display")
        self.token_display.display_threshold = self.display_threshold
        self.token_display.set_predictions(self.predictions)
        yield self.token_display

        # Probe details (left column)
        self.probe_details = ProbeDetails(id="probe-details")
        self.probe_details.display_threshold = self.display_threshold
        self.probe_details.set_data(self.predictions, self.tokens)
        yield self.probe_details

        # Heatmap (right column) - showing all layers
        self.heatmap = ActivationHeatmap(id="heatmap")
        self.heatmap.display_threshold = self.display_threshold
        self.heatmap.set_data(self.predictions, self.layer_range)
        yield self.heatmap

        yield Footer()

    def action_next_token(self) -> None:
        """Move to next token"""
        if self.selected_token < len(self.tokens) - 1:
            self.selected_token += 1
            self._update_selection()

    def action_previous_token(self) -> None:
        """Move to previous token"""
        if self.selected_token > 0:
            self.selected_token -= 1
            self._update_selection()

    def action_first_token(self) -> None:
        """Jump to first token"""
        self.selected_token = 0
        self._update_selection()

    def action_last_token(self) -> None:
        """Jump to last token"""
        self.selected_token = len(self.tokens) - 1
        self._update_selection()

    def action_reload(self) -> None:
        """Reload display"""
        self._update_selection()

    def _update_selection(self) -> None:
        """Update all panels with new selection"""
        self.token_display.selected_token = self.selected_token
        self.probe_details.selected_token = self.selected_token
        self.heatmap.selected_token = self.selected_token


def launch_interactive_viewer(
    predictions: List[AggregatedPrediction],
    tokens: List[str],
    layer_range: Tuple[int, int],
    display_threshold: float
):
    """
    Launch the interactive TUI

    Args:
        predictions: List of aggregated predictions sorted by layer count
        tokens: List of token strings
        layer_range: Tuple of (start_layer, end_layer)
    """
    app = ProbeViewerApp(predictions, tokens, layer_range, display_threshold)
    app.run()


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive probe viewer")
    parser.add_argument("--probes-dir", type=str, required=True, help="Probes directory")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Model name")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--threshold", type=float, default=0.1, help="Activation threshold")
    parser.add_argument("--display-threshold", type=float, default=None, help="UI display threshold (defaults to --threshold)")

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
        text = "After receiving feedback, I began reconsidering my approach and analyzing the problem."

    print(f"\nAnalyzing: '{text}'\n")

    # Run inference
    import time
    start = time.time()
    all_predictions = engine.predict_streaming(
        text,
        top_k=len(engine.probes),  # Get ALL predictions for aggregation
        threshold=0.0,  # Get all, filter during aggregation
        show_realtime=False  # Don't show streaming output
    )
    processing_time = time.time() - start

    # Aggregate predictions by action across layers
    aggregated_predictions = engine.aggregate_predictions(all_predictions, threshold=args.threshold)

    # Sort by layer count (descending), then by max confidence (descending)
    aggregated_predictions.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)

    # Get tokens
    inputs = engine.tokenizer(text, return_tensors="pt")
    tokens = engine.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    print(f"✓ Processed in {processing_time:.2f}s")
    print(f"✓ Found {len(aggregated_predictions)} aggregated actions across {len(all_predictions)} layer predictions")
    print(f"✓ Layer range: {engine.layer_start}-{engine.layer_end}")
    print(f"\nLaunching interactive viewer...\n")

    # Launch TUI with aggregated predictions sorted by layer count
    launch_interactive_viewer(
        aggregated_predictions, 
        tokens, 
        (engine.layer_start, engine.layer_end),
        args.display_threshold if args.display_threshold is not None else args.threshold
    )


if __name__ == "__main__":
    main()
