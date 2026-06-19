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
    # SAM1 family (for reference)
    {"name": "SAM2.1 Large", "family": "SAM2", "coco_miou": 0.4179, "lvis_miou": 0.4630, "size": 857.0, "offset": (8, 0, 'left', 'center')},
    {"name": "SAM2.1 Base+", "family": "SAM2", "coco_miou": 0.4530, "lvis_miou": 0.4761, "size": 309.0, "offset": (8, 0, 'left', 'center')},
    {"name": "SAM2.1 Small", "family": "SAM2", "coco_miou": 0.4266, "lvis_miou": 0.4634, "size": 176.0, "offset": (8, 0, 'left', 'center')},
    {"name": "SAM2.1 Tiny", "family": "SAM2", "coco_miou": 0.4217, "lvis_miou": 0.4505, "size": 149.0, "offset": (-8, 0, 'right', 'center')},
    # SAM3 family
    {"name": "SAM3", "family": "SAM3", "coco_miou": 0.3641, "lvis_miou": 0.4053, "size": 1200.0, "offset": (8, 0, 'left', 'center')},
    # PicoSAM2 variants
    {"name": "PicoSAM2 (Scratch)", "family": "PicoSAM2", "coco_miou": 0.5094, "lvis_miou": 0.4844, "size": 4.84, "offset": (-8, 0, 'right', 'center')},
    {"name": "PicoSAM2 (D-SAM2)", "family": "PicoSAM2", "coco_miou": 0.6311, "lvis_miou": 0.6180, "size": 4.84, "offset": (0, -12, 'center', 'top')},
    {"name": "PicoSAM2 (D-SAM3)", "family": "PicoSAM2", "coco_miou": 0.6351, "lvis_miou": 0.6231, "size": 4.84, "offset": (0, 12, 'center', 'bottom')},
    # PicoSAM3 variants
    {"name": "PicoSAM3 (D-SAM3)", "family": "PicoSAM3", "coco_miou": 0.6545, "lvis_miou": 0.6401, "size": 5.26, "offset": (8, 0, 'left', 'center')},
    {"name": "PicoSAM3 (DQ-SAM3)", "family": "PicoSAM3", "coco_miou": 0.6534, "lvis_miou": 0.6398, "size": 1.31, "offset": (8, 0, 'left', 'center')},
]

# Color palette - visually distinct and colorblind-friendly
family_colors = {
    'SAM1': '#7B7B7B',      # Gray (legacy)
    'EfficientSAM': '#6B4C9A',  # Purple
    'SAM2': '#2E86AB',      # Steel blue
    'SAM3': '#A23B72',      # Berry
    'PicoSAM2': '#F18F01',  # Orange
    'PicoSAM3': '#C73E1D',  # Red-orange
}

family_markers = {
    'SAM1': 'p',       # Pentagon
    'EfficientSAM': 'h',  # Hexagon
    'SAM2': 'o',       # Circle
    'SAM3': 's',       # Square
    'PicoSAM2': '^',   # Triangle up
    'PicoSAM3': 'D',   # Diamond
}

def create_plot(results, x_key, x_label, y_label, title, filename):
    fig, ax = plt.subplots(figsize=(11, 7))

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
            fontsize=10,
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

    x_margin = (max(all_x) - min(all_x)) * 0.15
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin * 3)
    ax.set_ylim(min(all_y) * 0.3, max(all_y) * 3)

    # Legend with better styling
    ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=11,
        markerscale=1.2,
        title='Model Family',
        title_fontsize=12,
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
    "coco_miou",
    "mIoU on COCO Dataset",
    "Model Size (MB)",
    "",
    "images/coco_miou_vs_size.png"
)

create_plot(
    results,
    "lvis_miou",
    "mIoU on LVIS Dataset",
    "Model Size (MB)",
    "",
    "images/lvis_miou_vs_size.png"
)
