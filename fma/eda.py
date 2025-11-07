import colorsys
from collections import Counter
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA

from fma.data import FMADataset
from fma.plain import console, with_status
from fma.types import PathLike


@with_status
def draw_genre_tree(
    dataset: FMADataset,
    save_file: PathLike | None = None,
    *,
    num_rows: int = 3,
    n_bins: int = 5,
    verbose: bool = True,
) -> None:
    """
    Visualize the hierarchical genre tree structure.
    Only plots genres that are present in dataset.genres.

    Args:
        dataset: FMADataset containing genre information
        save_file: Path to save the plot to
        verbose: Whether to print messages
        num_rows: Number of rows to display
    """

    NUM_ROWS = num_rows

    # Count number of genres for each genre id
    counter = Counter(
        [genre_id for genre_list in dataset.track_genres for genre_id in genre_list]
    )

    # Collect all genre IDs that appear in dataset.genres
    present_genre_ids = set()
    for genre_list in dataset.track_genres:
        present_genre_ids.update(genre_list)

    if not present_genre_ids:
        raise ValueError("No genres found in dataset")

    # Filter genres to only include present ones and their ancestors
    def add_ancestors(genre_id: int, genre_set: set) -> None:
        """Add a genre and all its ancestors to the set"""
        if genre_id in genre_set or genre_id not in dataset.id_to_genre:
            return
        genre_set.add(genre_id)
        genre = dataset.id_to_genre[genre_id]
        if genre.parent_id is not None:
            add_ancestors(genre.parent_id, genre_set)

    # Include all ancestors of present genres
    filtered_genre_ids = set()
    for genre_id in present_genre_ids:
        add_ancestors(genre_id, filtered_genre_ids)

    # Find root genres that have descendants in the filtered set
    root_genres = [
        g
        for g in dataset.id_to_genre.values()
        if g.parent_id is None and g.id in filtered_genre_ids
    ]

    if not root_genres:
        raise ValueError("No root genres found")

    # Helper function to get filtered children
    def get_filtered_children(genre_id: int) -> list[int]:
        """Get children that are in the filtered set"""
        genre = dataset.id_to_genre[genre_id]
        return [
            child_id for child_id in genre.children if child_id in filtered_genre_ids
        ]

    # Calculate tree layout
    def get_tree_depth(genre_id: int, depth: int = 0) -> int:
        """Get maximum depth of tree starting from this genre"""
        children = get_filtered_children(genre_id)
        if not children:
            return depth
        return max(get_tree_depth(child_id, depth + 1) for child_id in children)

    def count_leaves(genre_id: int) -> int:
        """Count number of leaf nodes in subtree"""
        children = get_filtered_children(genre_id)
        if not children:
            return 1
        return sum(count_leaves(child_id) for child_id in children)

    def layout_tree(
        genre_id: int,
        x_start: float,
        x_end: float,
        depth: int,
        positions: dict,
        edges: list,
        depth_offset: int = 0,
    ) -> None:
        """Recursively layout tree nodes with proper depth-based Y positioning"""
        x_mid = (x_start + x_end) / 2

        # Y position is determined by depth level, not recursive call
        y = -depth * VERTICAL_LEVEL_SPACING + depth_offset
        positions[genre_id] = (x_mid, y)

        children = get_filtered_children(genre_id)
        if children:
            # Calculate width for each child based on their leaf count
            child_leaves = [count_leaves(child_id) for child_id in children]
            total_leaves = sum(child_leaves)

            x_current = x_start
            for child_id, leaves in zip(children, child_leaves):
                child_width = (x_end - x_start) * (leaves / total_leaves)
                child_x_start = x_current
                child_x_end = x_current + child_width

                # Add edge from parent to child
                edges.append((genre_id, child_id))

                # Recursively layout child at next depth level
                layout_tree(
                    child_id,
                    child_x_start,
                    child_x_end,
                    depth + 1,  # Increment depth
                    positions,
                    edges,
                    depth_offset,
                )

                x_current = child_x_end

    def assign_trees_to_rows(
        root_genres,
        num_rows,
        node_width_multiplier,
        horizontal_spacing,
        width_variance_threshold: float | None = None,
    ):
        """
        Assign trees (root genres) to rows while balancing widths and minimizing total height.

        Strategy:
            * Compute a reasonable width-variance target from the width-only baseline when none
              is supplied.
            * Allow a soft overrun of that target during the improvement search so that height
              can be reduced even if an intermediate step widens a row.
            * Run several deterministic restarts (width-first, height-first, randomized) with
              hill climbing, then a tightening pass that reduces width variance while keeping
              height minimal.
            * Return the best layout that satisfies the hard threshold; if none do, fall back
              to the best layout found and report the overshoot.

        Args:
            root_genres: List of root genre objects.
            num_rows: Number of available rows.
            node_width_multiplier: Width multiplier per leaf node.
            horizontal_spacing: Spacing between trees within a row.
            width_variance_threshold: Optional hard cap on the acceptable width variance.
                                      If omitted, a data-driven default is used.

        Returns:
            List[List[Genre]] – each inner list is the set of root genres assigned to that row.
        """
        import random

        if num_rows <= 0:
            return []
        if not root_genres:
            return [[] for _ in range(num_rows)]

        tol = 1e-6
        rng = random.Random(42)

        # --- Pre-compute per-tree width and height contribution ---------------------------------
        tree_width_map: dict[int, float] = {}
        tree_depth_map: dict[int, int] = {}
        for root in root_genres:
            tree_width_map[root.id] = count_leaves(root.id) * node_width_multiplier
            tree_depth_map[root.id] = get_tree_depth(root.id)

        def compute_row_metrics(
            rows_local: list[list],
        ) -> tuple[list[float], list[int]]:
            """Return per-row widths and heights (depth levels; -1 means empty row)."""
            widths = []
            heights = []
            for row in rows_local:
                if row:
                    widths_in_row = [tree_width_map[item.id] for item in row]
                    row_width = sum(widths_in_row) + horizontal_spacing * (len(row) - 1)
                    row_height = max(tree_depth_map[item.id] for item in row)
                else:
                    row_width = 0.0
                    row_height = -1
                widths.append(row_width)
                heights.append(row_height)
            return widths, heights

        def compute_cost(rows_local: list[list]) -> tuple[float, float]:
            """Return (width variance, total height score in spacing units)."""
            widths, heights = compute_row_metrics(rows_local)
            width_variance = float(np.var(widths)) if any(widths) else 0.0
            total_height = sum(
                (h + 1) * VERTICAL_LEVEL_SPACING for h in heights if h >= 0
            )
            return width_variance, total_height

        def lex_height_then_width(
            cost_a: tuple[float, float], cost_b: tuple[float, float]
        ) -> bool:
            """True if cost_a is strictly better than cost_b (height first, then width)."""
            if cost_b is None:
                return True
            wv_a, height_a = cost_a
            wv_b, height_b = cost_b
            if height_a + tol < height_b:
                return True
            if height_b + tol < height_a:
                return False
            return wv_a + tol < wv_b

        # --- Baseline (width-only) assignment to derive defaults --------------------------------
        baseline_rows = [[] for _ in range(num_rows)]
        baseline_widths = [0.0] * num_rows
        for root in sorted(
            root_genres, key=lambda r: tree_width_map[r.id], reverse=True
        ):
            idx = min(range(num_rows), key=lambda i: baseline_widths[i])
            increment = tree_width_map[root.id]
            if baseline_rows[idx]:
                increment += horizontal_spacing
            baseline_rows[idx].append(root)
            baseline_widths[idx] += increment
        baseline_cost = compute_cost(baseline_rows)
        baseline_variance = baseline_cost[0]

        if width_variance_threshold is None:
            width_variance_threshold = baseline_variance * 10

        hard_threshold = float(width_variance_threshold)
        soft_threshold = np.inf

        def within_soft(cost: tuple[float, float]) -> bool:
            return cost[0] <= soft_threshold + tol

        def within_hard(cost: tuple[float, float]) -> bool:
            return cost[0] <= hard_threshold + tol

        # --- Move generators -------------------------------------------------------------------
        def try_all_single_moves(
            rows_local: list[list],
        ) -> list[tuple[list[list], tuple[float, float]]]:
            moves = []
            for src_idx, row in enumerate(rows_local):
                for item_idx, root in enumerate(row):
                    for dst_idx in range(num_rows):
                        if dst_idx == src_idx:
                            continue
                        candidate_rows = [list(r) for r in rows_local]
                        candidate_rows[src_idx].pop(item_idx)
                        candidate_rows[dst_idx].append(root)
                        cost = compute_cost(candidate_rows)
                        moves.append((candidate_rows, cost))
            return moves

        def try_all_swaps(
            rows_local: list[list],
        ) -> list[tuple[list[list], tuple[float, float]]]:
            swaps = []
            for i in range(num_rows):
                for j in range(i + 1, num_rows):
                    for idx_a, root_a in enumerate(rows_local[i]):
                        for idx_b, root_b in enumerate(rows_local[j]):
                            candidate_rows = [list(r) for r in rows_local]
                            candidate_rows[i][idx_a] = root_b
                            candidate_rows[j][idx_b] = root_a
                            cost = compute_cost(candidate_rows)
                            swaps.append((candidate_rows, cost))
            return swaps

        # --- Local improvement phases ----------------------------------------------------------
        def soft_height_pass(
            rows_local: list[list],
        ) -> tuple[list[list], tuple[float, float]]:
            rows = [list(r) for r in rows_local]
            current_cost = compute_cost(rows)
            improved = True
            while improved:
                improved = False
                best_candidate: tuple[list[list], tuple[float, float]] | None = None
                for candidate, cost in try_all_single_moves(rows) + try_all_swaps(rows):
                    if not within_soft(cost):
                        continue
                    # Prefer lower height; if equal height, prefer smaller variance
                    if cost[1] + tol < current_cost[1]:
                        if (
                            best_candidate is None
                            or cost[1] + tol < best_candidate[1][1]
                            or (
                                abs(cost[1] - best_candidate[1][1]) <= tol
                                and cost[0] + tol < best_candidate[1][0]
                            )
                        ):
                            best_candidate = (candidate, cost)
                    elif (
                        abs(cost[1] - current_cost[1]) <= tol
                        and cost[0] + tol < current_cost[0] - tol
                    ):
                        if (
                            best_candidate is None
                            or cost[0] + tol < best_candidate[1][0]
                        ):
                            best_candidate = (candidate, cost)
                if best_candidate:
                    rows, current_cost = best_candidate
                    improved = True
            return rows, current_cost

        def tighten_width(
            rows_local: list[list],
        ) -> tuple[list[list], tuple[float, float]]:
            rows = [list(r) for r in rows_local]
            current_cost = compute_cost(rows)
            improved = True
            while improved:
                improved = False
                best_candidate: tuple[list[list], tuple[float, float]] | None = None
                for candidate, cost in try_all_single_moves(rows) + try_all_swaps(rows):
                    if cost[1] > current_cost[1] + tol:
                        continue  # do not allow height increase
                    if not within_hard(cost):
                        continue
                    if (
                        cost[1] + tol < current_cost[1]
                        or cost[0] + tol < current_cost[0] - tol
                    ):
                        if best_candidate is None or lex_height_then_width(
                            cost, best_candidate[1]
                        ):
                            best_candidate = (candidate, cost)
                if best_candidate:
                    rows, current_cost = best_candidate
                    improved = True
            return rows, current_cost

        # --- Seeding strategies -----------------------------------------------------------------
        def greedy_height_seed(shuffle: bool = False) -> list[list]:
            items = list(root_genres)
            if shuffle:
                rng.shuffle(items)
            items.sort(
                key=lambda r: (
                    tree_depth_map[r.id],
                    tree_width_map[r.id],
                ),
                reverse=True,
            )
            rows = [[] for _ in range(num_rows)]
            for root in items:
                best_idx = 0
                best_cost = None
                for idx in range(num_rows):
                    candidate_rows = [list(r) for r in rows]
                    candidate_rows[idx].append(root)
                    cost = compute_cost(candidate_rows)
                    if best_cost is None or lex_height_then_width(cost, best_cost):
                        best_idx = idx
                        best_cost = cost
                rows[best_idx].append(root)
            return rows

        def random_seed() -> list[list]:
            rows = [[] for _ in range(num_rows)]
            for root in rng.sample(root_genres, len(root_genres)):
                idx = rng.randrange(num_rows)
                rows[idx].append(root)
            return rows

        seed_layouts: list[list[list]] = []
        seed_layouts.append(baseline_rows)
        seed_layouts.append(greedy_height_seed(False))
        for _ in range(min(4, len(root_genres))):
            seed_layouts.append(greedy_height_seed(True))
        for _ in range(max(0, len(root_genres) // 2)):
            seed_layouts.append(random_seed())

        # Deduplicate seeds
        seen_signatures = set()
        unique_seeds = []
        for seed in seed_layouts:
            signature = tuple(tuple(sorted(item.id for item in row)) for row in seed)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_seeds.append(seed)

        # --- Main search loop -------------------------------------------------------------------
        best_within_threshold: tuple[list[list], tuple[float, float]] | None = None
        best_overall: tuple[list[list], tuple[float, float]] | None = None

        for seed in unique_seeds:
            rows, cost = soft_height_pass(seed)
            rows, cost = tighten_width(rows)

            if best_overall is None or lex_height_then_width(cost, best_overall[1]):
                best_overall = (rows, cost)

            if within_hard(cost):
                if best_within_threshold is None or lex_height_then_width(
                    cost, best_within_threshold[1]
                ):
                    best_within_threshold = (rows, cost)

        if best_within_threshold is not None:
            chosen_rows, chosen_cost = best_within_threshold
        elif best_overall is not None:
            chosen_rows, chosen_cost = best_overall
        else:
            assert False, "Unreachable"

        # if verbose:
        #     wv, total_height = chosen_cost
        #     widths, heights = compute_row_metrics(chosen_rows)
        #     row_levels = [h + 1 if h >= 0 else 0 for h in heights]

        #     print(
        #         f"Width variance threshold (hard/soft): {hard_threshold:.3f} / "
        #         f"{('∞' if not np.isfinite(soft_threshold) else f'{soft_threshold:.3f}')}"
        #     )
        #     print(f"Baseline width variance: {baseline_variance:.3f}")
        #     print(f"Row widths after optimization: {[f'{w:.1f}' for w in widths]}")
        #     print(f"Width variance: {wv:.3f}")
        #     print(f"Row heights (levels): {row_levels}")
        #     print(f"Total height score (spacing units): {total_height:.3f}")

        #     if best_within_threshold is None and np.isfinite(hard_threshold):
        #         print(
        #             "Warning: Could not satisfy the requested width variance threshold. "
        #             "Returned the best overall layout instead."
        #         )

        return chosen_rows

    # Constants
    HORIZONTAL_SPACING = 0.5
    NODE_WIDTH_MULTIPLIER = 1
    VERTICAL_LEVEL_SPACING = 3.0
    ROW_SPACING = 0
    FONT_WIDTH_MULTIPLIER = 0.016
    BOX_HEIGHT = 0.5

    # Use optimized assignment algorithm
    rows = assign_trees_to_rows(
        root_genres, NUM_ROWS, NODE_WIDTH_MULTIPLIER, HORIZONTAL_SPACING
    )

    # Calculate positions for all genres with row-based layout
    positions = {}
    edges = []

    # Calculate max depth for each row
    row_max_depths = []
    for row_roots in rows:
        if row_roots:
            max_depth = max(get_tree_depth(root.id) for root in row_roots)
            row_max_depths.append(max_depth)
        else:
            row_max_depths.append(0)

    total_width = 0

    for row_idx, row_roots in enumerate(rows):
        if not row_roots:
            continue

        x_offset = 0

        # Calculate Y offset for this row based on previous rows' depths
        y_offset = 0
        for prev_idx in range(row_idx):
            # Add space for previous row's depth plus gap between rows
            y_offset -= (
                row_max_depths[prev_idx] + 1
            ) * VERTICAL_LEVEL_SPACING + ROW_SPACING

        for root in row_roots:
            width = count_leaves(root.id) * NODE_WIDTH_MULTIPLIER

            # Start each tree at depth 0, with y_offset determining row position
            layout_tree(
                root.id,
                x_offset,
                x_offset + width,
                0,  # Start at depth 0 for each root
                positions,
                edges,
                y_offset,  # Apply row offset
            )
            x_offset += width + HORIZONTAL_SPACING

        total_width = max(total_width, x_offset)

    # Calculate total height needed
    total_height = sum((depth + 1) * VERTICAL_LEVEL_SPACING for depth in row_max_depths)
    total_height += len(rows) * ROW_SPACING  # Add inter-row spacing

    # Create figure with appropriate sizing - more vertical, less horizontal
    fig_width = total_width * 0.45 + 2  # Add a small padding for colorbar
    fig_height = total_height * 0.55

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-1, total_width + 1)
    ax.set_ylim(-total_height - 1, 2)
    ax.axis("off")

    # Assign colors based on genre frequency using discrete bins
    # Get all genre frequencies and normalize
    max_freq = max(counter.values()) if counter else 1
    min_freq = min(counter.values()) if counter else 1

    # Create a lightened version of rainbow colormap for better text visibility
    def lighten_colormap(cmap, min_lightness=0.75, samples=256):
        """
        Create a lightened version of a colormap by adjusting lightness in HLS space.

        Args:
            cmap: The input colormap
            min_lightness: Minimum lightness value (0-1). Higher = lighter colors
            samples: Number of color samples to take from the colormap

        Returns:
            A new LinearSegmentedColormap with lightened colors
        """
        # Sample colors from the original colormap
        colors = cmap(np.linspace(0, 1, samples))

        # Convert to HLS and increase lightness
        lightened_colors = []
        for color in colors:
            r, g, b, a = color
            h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa

            # Increase lightness - blend towards white
            l = max(l, min_lightness)  # Ensure minimum lightness # noqa

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            lightened_colors.append((r, g, b, a))

        # Create new colormap from lightened colors
        return LinearSegmentedColormap.from_list(
            f"{cmap.name}_light", lightened_colors, N=samples
        )

    # Use a lightened rainbow colormap for better text visibility
    colormap = lighten_colormap(plt.cm.rainbow, min_lightness=0.8)  # type: ignore
    color_map = {}

    # Define discrete bins for frequencies (log-spaced)
    # Create bins that are evenly spaced in log space
    if max_freq > min_freq:
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        # Create bin boundaries in log space, then convert back
        boundaries = np.logspace(log_min, log_max, n_bins + 1)
    else:
        boundaries = np.array([min_freq, max_freq])
        n_bins = 1

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm

    # Create discrete normalization based on boundaries
    norm = BoundaryNorm(boundaries, colormap.N)

    for genre_id in filtered_genre_ids:
        freq = counter.get(genre_id, 0)
        # Add 1 to avoid log(0) for genres with 0 count
        if freq == 0:
            freq = 1
        # Map frequency to discrete color using BoundaryNorm
        color_map[genre_id] = colormap(norm(freq))

    # Add discrete colorbar to show frequency bins
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, spacing="uniform")
    cbar.set_label(
        "Genre Frequency (log scale, discrete bins)",
        rotation=270,
        labelpad=20,
        fontsize=10,
    )

    # All root nodes are horizontal (top level always horizontal)
    horizontal_nodes = {root.id for root in root_genres}

    # Calculate box dimensions for each node (needed for edge drawing)
    box_dimensions = {}

    for genre_id in positions.keys():
        genre = dataset.id_to_genre[genre_id]
        has_children = bool(get_filtered_children(genre_id))
        is_root = genre.parent_id is None
        is_horizontal = genre_id in horizontal_nodes

        # Smaller font sizes
        if is_root:
            fontsize = 8.5
        elif has_children:
            fontsize = 7.5
        else:
            fontsize = 7

        title = genre.title
        char_width = fontsize * FONT_WIDTH_MULTIPLIER
        padding = 0.1  # Reduced padding

        if is_horizontal:
            box_width = len(title) * char_width + padding * 2
            box_width = max(box_width, 1.3)
            box_width = min(box_width, 10.0)
            box_height = BOX_HEIGHT  # Reduced height
        else:
            box_width = BOX_HEIGHT  # Reduced width
            box_height = len(title) * char_width + padding * 2
            box_height = max(box_height, 1.3)
            box_height = min(box_height, 5.5)

        box_dimensions[genre_id] = (box_width, box_height)

    # Draw edges with proper connection points (bottom center to top center)
    for parent_id, child_id in edges:
        px, py = positions[parent_id]
        cx, cy = positions[child_id]

        # Get box dimensions
        p_width, p_height = box_dimensions[parent_id]
        c_width, c_height = box_dimensions[child_id]

        # Parent connection point: bottom center
        p_connect_x = px
        p_connect_y = py - p_height / 2

        # Child connection point: top center
        c_connect_x = cx
        c_connect_y = cy + c_height / 2

        ax.plot(
            [p_connect_x, c_connect_x],
            [p_connect_y, c_connect_y],
            "k-",
            alpha=0.3,
            linewidth=1.5,
            zorder=1,
        )

    # Draw each node
    for genre_id, (x, y) in positions.items():
        genre = dataset.id_to_genre[genre_id]

        # Determine node properties
        has_children = bool(get_filtered_children(genre_id))
        is_root = genre.parent_id is None
        is_horizontal = genre_id in horizontal_nodes

        # Font sizes
        if is_root:
            fontsize = 8.5
        elif has_children:
            fontsize = 7.5
        else:
            fontsize = 7

        title = genre.title
        box_width, box_height = box_dimensions[genre_id]

        # Draw box
        color = color_map.get(genre_id, "lightgray")

        # Add subtle shadow
        shadow = FancyBboxPatch(
            (x - box_width / 2 + 0.03, y - box_height / 2 - 0.03),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            facecolor="gray",
            edgecolor="none",
            alpha=0.15,
            zorder=2,
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(box)

        # Add text
        rotation = 0 if is_horizontal else 90

        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="normal",
            rotation=rotation,
            zorder=4,
        )

    plt.title("Genre Hierarchy Tree", fontsize=20, fontweight="bold", pad=30)
    plt.tight_layout()

    if save_file:
        save_file = Path(save_file).absolute()
        plt.savefig(save_file)
        plt.close()
        if verbose:
            console.done(f"Saved figure to {save_file}")
    else:
        plt.show()
        plt.close()


@with_status
def describe_tracks(
    dataset: FMADataset,
    save_file: PathLike | None = None,
    *,
    n_bins: int = 15,
    verbose: bool = True,
) -> None:
    # Set seaborn style for professional appearance
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Set up the figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(
        "Description of Tracks",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Define a color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D", "#C73E1D", "#6A4C93"]

    # Helper function to format numbers
    def format_func(value, tick_number):
        if value >= 1e6:
            return f"{value / 1e6:.0f}M"
        elif value >= 1e3:
            return f"{value / 1e3:.0f}K"
        else:
            return f"{value:.0f}"

    # Plot 1: Listens
    listens_data = dataset.tracks["listens"][dataset.tracks["listens"] > 0]
    log_listens = np.log10(listens_data)
    axes[0].hist(
        log_listens,
        bins=n_bins,
        color=colors[0],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Listens", fontsize=11, fontweight="semibold")
    axes[0].set_ylabel("Frequency", fontsize=11, fontweight="semibold")
    axes[0].set_title(
        "(a) Distribution of Listens", fontsize=12, fontweight="bold", pad=10
    )
    axes[0].set_xticks(np.arange(int(log_listens.min()), int(log_listens.max()) + 1))
    axes[0].set_xticklabels([format_func(10**x, None) for x in axes[0].get_xticks()])
    axes[0].grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Plot 2: Favorites
    favorites_data = dataset.tracks["favorites"][dataset.tracks["favorites"] > 0]
    log_favorites = np.log10(favorites_data)
    axes[1].hist(
        log_favorites,
        bins=n_bins,
        color=colors[1],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Favorites", fontsize=11, fontweight="semibold")
    axes[1].set_ylabel("Frequency", fontsize=11, fontweight="semibold")
    axes[1].set_title(
        "(b) Distribution of Favorites", fontsize=12, fontweight="bold", pad=10
    )
    axes[1].set_xticks(
        np.arange(int(log_favorites.min()), int(log_favorites.max()) + 1)
    )
    axes[1].set_xticklabels([format_func(10**x, None) for x in axes[1].get_xticks()])
    axes[1].grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Plot 3: Interest
    interest_data = dataset.tracks["interest"][dataset.tracks["interest"] > 0]
    log_interest = np.log10(interest_data)
    axes[2].hist(
        log_interest,
        bins=n_bins,
        color=colors[2],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Interest", fontsize=11, fontweight="semibold")
    axes[2].set_ylabel("Frequency", fontsize=11, fontweight="semibold")
    axes[2].set_title(
        "(c) Distribution of Interest", fontsize=12, fontweight="bold", pad=10
    )
    axes[2].set_xticks(np.arange(int(log_interest.min()), int(log_interest.max()) + 1))
    axes[2].set_xticklabels([format_func(10**x, None) for x in axes[2].get_xticks()])
    axes[2].grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Plot 4: Durations
    duration_data = dataset.tracks["duration"][dataset.tracks["duration"] > 0]
    log_durations = np.log10(duration_data)
    axes[3].hist(
        log_durations,
        bins=n_bins,
        color=colors[3],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
    )
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Duration (seconds)", fontsize=11, fontweight="semibold")
    axes[3].set_ylabel("Frequency", fontsize=11, fontweight="semibold")
    axes[3].set_title(
        "(d) Distribution of Duration", fontsize=12, fontweight="bold", pad=10
    )
    axes[3].set_xticks(
        np.arange(int(log_durations.min()), int(log_durations.max()) + 1)
    )
    axes[3].set_xticklabels([format_func(10**x, None) for x in axes[3].get_xticks()])
    axes[3].grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Plot 5: Bit Rate Distribution (Pie Chart)
    # Filter out invalid bit rates and get top values
    bit_rate_data = dataset.tracks["bit_rate"][dataset.tracks["bit_rate"] > 0]
    bit_rate_counts = bit_rate_data.value_counts().head(5)

    # Group smaller values into "Others"
    total_count = len(bit_rate_data)
    other_count = total_count - bit_rate_counts.sum()
    if other_count > 0:
        bit_rate_counts["Others"] = other_count

    # Define colors for pie chart
    pie_colors = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#06A77D",
        "#C73E1D",
        "#6A4C93",
        "#E0E0E0",
    ]

    # Create pie chart
    wedges, texts, autotexts = axes[4].pie(
        bit_rate_counts.values,
        labels=[
            f"{int(br / 1000)} kbps" if br != "Others" else br
            for br in bit_rate_counts.index
        ],
        autopct="%1.1f%%",
        colors=pie_colors[: len(bit_rate_counts)],
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "semibold"},
    )

    # Make percentage text white and bold for better visibility
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    axes[4].set_title("(e) Bit Rates", fontsize=12, fontweight="bold", pad=10)

    # Plot 6: Genre Distribution (Bar Chart)
    # Flatten all genres and count occurrences
    all_genres: list[int] = []
    for genre_list in dataset.track_genres:
        all_genres.extend(genre_list)

    genre_counts = Counter(all_genres)
    top_genres = dict(
        sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    # Create bar chart
    bars = axes[5].barh(
        range(len(top_genres)),
        list(top_genres.values()),
        color=colors[5],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
    )
    axes[5].set_ylabel("Genre ID", fontsize=11, fontweight="semibold")
    axes[5].set_xlabel("Frequency", fontsize=11, fontweight="semibold")
    axes[5].set_title("(f) Top 10 Genre", fontsize=12, fontweight="bold", pad=10)
    axes[5].set_yticks(range(len(top_genres)))
    axes[5].set_yticklabels([dataset.id_to_genre[g].title for g in top_genres.keys()])
    axes[5].grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
    axes[5].invert_yaxis()  # Highest count at top

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[5].text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="semibold",
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_file:
        save_file = Path(save_file).absolute()
        plt.savefig(save_file)
        plt.close()
        if verbose:
            console.done(f"Saved figure to {save_file}")
    else:
        plt.show()
        plt.close()


@with_status
def plot_pca(
    dataset: FMADataset,
    save_file: PathLike | None = None,
    *,
    max_components: int | None = None,
    variance_thresholds: list[float] | None = None,
    verbose: bool = True,
) -> None:
    """
    Visualize PCA explained variance to show how many dimensions are needed
    to retain a certain amount of variance.

    Args:
        dataset: FMADataset containing feature information
        save_file: Path to save the plot to
        max_components: Maximum number of components to plot (default: all)
        variance_thresholds: List of variance thresholds to mark on the plot
                           (default: [0.80, 0.90, 0.95, 0.99])
        verbose: Whether to print messages
    """
    if variance_thresholds is None:
        variance_thresholds = [0.80, 0.90, 0.95, 0.99]

    # Get features from dataset
    X = dataset.features.values

    # Determine number of components
    n_components = min(X.shape) if max_components is None else max_components
    n_components = min(n_components, min(X.shape))

    # Fit PCA
    if verbose:
        console.log(f"Fitting PCA with {n_components} components...")

    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Get explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Set up the figure
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "PCA Explained Variance Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Plot 1: Cumulative explained variance
    ax1.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        linewidth=2.5,
        color="#2E86AB",
        marker="o",
        markersize=3,
        markevery=max(1, len(cumulative_variance) // 20),
    )

    # Add variance threshold lines and annotations
    colors_thresh = ["#F18F01", "#A23B72", "#06A77D", "#C73E1D"]
    for i, threshold in enumerate(variance_thresholds):
        if threshold <= cumulative_variance[-1]:
            # Find the number of components needed
            n_needed = np.argmax(cumulative_variance >= threshold) + 1

            ax1.axhline(
                y=threshold,
                color=colors_thresh[i % len(colors_thresh)],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"{threshold * 100:.0f}% variance",
            )
            ax1.axvline(
                x=n_needed,
                color=colors_thresh[i % len(colors_thresh)],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

            # Add annotation
            ax1.annotate(
                f"n={n_needed}",
                xy=(n_needed, threshold),
                xytext=(10, -10),
                textcoords="offset points",
                fontsize=9,
                color=colors_thresh[i % len(colors_thresh)],
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=colors_thresh[i % len(colors_thresh)],
                    alpha=0.8,
                ),
            )

    ax1.set_xlabel("Number of Components", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Cumulative Explained Variance", fontsize=11, fontweight="semibold")
    ax1.set_title(
        "(a) Cumulative Explained Variance",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Individual explained variance (bar chart for first N components)
    n_bars = min(50, n_components)
    ax2.bar(
        range(1, n_bars + 1),
        explained_variance_ratio[:n_bars],
        color="#A23B72",
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
    )

    ax2.set_xlabel("Component Number", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Explained Variance Ratio", fontsize=11, fontweight="semibold")
    ax2.set_title(
        f"(b) Individual Explained Variance (First {n_bars} Components)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Add summary statistics as text
    summary_text = (
        f"Total components: {n_components}\n"
        f"Total features: {X.shape[1]}\n"
        f"Samples: {X.shape[0]}"
    )
    ax2.text(
        0.95,
        0.95,
        summary_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_file:
        save_file = Path(save_file).absolute()
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close()
        if verbose:
            console.done(f"Saved PCA plot to {save_file}")
    else:
        plt.show()
        plt.close()

    # Print summary if verbose
    if verbose:
        console.log("\n[bold]PCA Summary:[/bold]")
        for threshold in variance_thresholds:
            if threshold <= cumulative_variance[-1]:
                n_needed = np.argmax(cumulative_variance >= threshold) + 1
                reduction = (1 - n_needed / X.shape[1]) * 100
                console.log(
                    f"  • {threshold * 100:.0f}% variance: {n_needed} components "
                    f"({reduction:.1f}% reduction)"
                )
