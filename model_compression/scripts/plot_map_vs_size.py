import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# CVPR-style plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 22,
    'text.usetex': False,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
})

# Results with model families and manual label offsets (x_off, y_off, ha, va)
results = [
    # SAM2 family
    {"name": "SAM2.1 Large", "family": "SAM2", "coco_map": 0.2555, "lvis_map": 0.2994, "size": 857.0, "offset": (8, 0, 'left', 'center')},
    {"name": "SAM2.1 Base+", "family": "SAM2", "coco_map": 0.2870, "lvis_map": 0.3014, "size": 309.0, "offset": (8, 0, 'left', 'center')},
    {"name": "SAM2.1 Small", "family": "SAM2", "coco_map": 0.2566, "lvis_map": 0.2884, "size": 176.0, "offset": (8, 0, 'left', 'center')},
    {"name": "SAM2.1 Tiny", "family": "SAM2", "coco_map": 0.2533, "lvis_map": 0.2756, "size": 149.0, "offset": (-8, 0, 'right', 'center')},
    # SAM3 family
    {"name": "SAM3", "family": "SAM3", "coco_map": 0.2213, "lvis_map": 0.2582, "size": 1200.0, "offset": (8, 0, 'left', 'center')},
    
    # PicoSAM2 variants (4.84 MB float32, 1.21 MB int8)
    {"name": "PicoSAM2 (Scratch)", "family": "PicoSAM2", "coco_map": 0.2268, "lvis_map": 0.2120, "size": 4.84, "offset": (0, 12, 'center', 'bottom')},
    {"name": "PicoSAM2 (D-SAM2)", "family": "PicoSAM2", "coco_map": 0.4052, "lvis_map": 0.4012, "size": 4.84, "offset": (0, -12, 'center', 'top')},
    {"name": "PicoSAM2 (D-SAM3)", "family": "PicoSAM2", "coco_map": 0.4113, "lvis_map": 0.4094, "size": 4.84, "offset": (0, 12, 'center', 'bottom')}, 

    # PicoSAM3 variants (5.26 MB float32, 1.31 MB int8)
    {"name": "PicoSAM3 (D-SAM3)", "family": "PicoSAM3", "coco_map": 0.4377, "lvis_map": 0.4334, "size": 5.26, "offset": (8, 0, 'left', 'center')},
    {"name": "PicoSAM3 (DQ-SAM3)", "family": "PicoSAM3", "coco_map": 0.4364, "lvis_map": 0.4331, "size": 1.31, "offset": (8, 0, 'left', 'center')},
]

# Color palette - visually distinct and colorblind-friendly
family_colors = {
    'SAM2': '#2E86AB',      # Steel blue
    'SAM3': '#A23B72',      # Berry
    'PicoSAM2': '#F18F01',  # Orange
    'PicoSAM3': '#C73E1D',  # Red-orange
}

family_markers = {
    'SAM2': 'o',       # Circle
    'SAM3': 's',       # Square
    'PicoSAM2': '^',   # Triangle up
    'PicoSAM3': 'D',   # Diamond
}

def create_plot(results, x_key, x_label, y_label, title, filename):
    fig, ax = plt.subplots(figsize=(10, 7))

    plotted_families = set()

    # Filter valid results
    valid_results = [r for r in results if r[x_key] is not None]

    for r in valid_results:
        x_value = r[x_key]
        y_value = r["size"]
        family = r["family"]

        # Plot with family-based styling
        label = family if family not in plotted_families else None
        plotted_families.add(family)

        ax.scatter(
            x_value, y_value,
            color=family_colors[family],
            marker=family_markers[family],
            s=200,
            edgecolor='white',
            linewidth=2,
            label=label,
            zorder=5,
            alpha=0.9
        )

        # Get offset for this point
        x_off, y_off, ha, va = r.get("offset", (8, 0, 'left', 'center'))

        # Add text label using annotate for precise offset control
        ax.annotate(
            r['name'],
            xy=(x_value, y_value),
            xytext=(x_off, y_off),
            textcoords='offset points',
            fontsize=11,
            fontweight='medium',
            ha=ha,
            va=va,
            bbox=dict(
                facecolor='white',
                alpha=0.9,
                edgecolor='#cccccc',
                linewidth=0.5,
                pad=3,
                boxstyle='round,pad=0.3'
            ),
            zorder=10
        )

    # Styling
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontweight='bold', labelpad=10)

    # Grid styling
    ax.grid(True, which='major', ls='-', alpha=0.3, color='gray')
    ax.grid(True, which='minor', ls=':', alpha=0.2, color='gray')

    # Set axis limits with padding
    all_x = [r[x_key] for r in valid_results]
    all_y = [r["size"] for r in valid_results]

    x_margin = (max(all_x) - min(all_x)) * 0.2
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin * 2.5)
    ax.set_ylim(min(all_y) * 0.3, max(all_y) * 3)

    # Legend with better styling
    ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=13,
        markerscale=1.2,
        title='Model Family',
        title_fontsize=14,
        labelspacing=1.2,  # Vertical space between legend items
    )

    # Title
    ax.set_title(title, fontweight='bold', pad=15)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

    plt.tight_layout()

    # Save in multiple formats
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "images"), exist_ok=True)

    # PDF for publication
    pdf_path = os.path.join(script_dir, filename.replace('.png', '.pdf'))
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300, format='pdf')

    # PNG for preview
    png_path = os.path.join(script_dir, filename)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, format='png')

    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

# Create plots
create_plot(
    results,
    "coco_map",
    "mAP@[0.5:0.95] on COCO Dataset",
    "Model Size (MB)",
    "",
    "images/coco_map_vs_size.png"
)

create_plot(
    results,
    "lvis_map",
    "mAP@[0.5:0.95] on LVIS Dataset",
    "Model Size (MB)",
    "",
    "images/lvis_map_vs_size.png"
)
