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
})

# Results with model families
results = [
    # SAM1 family
    {"name": "SAM-H", "family": "SAM1", "size": 2420.3},
    {"name": "SAM-L", "family": "SAM1", "size": 1174.9},
    {"name": "SAM-B", "family": "SAM1", "size": 343.3},
    # Efficient SAM variants
    {"name": "EfficientViT-L0", "family": "EfficientSAM", "size": 118.2},
    {"name": "MobileSAM", "family": "EfficientSAM", "size": 37.0},
    {"name": "EdgeSAM", "family": "EfficientSAM", "size": 37.0},
    {"name": "LiteSAM", "family": "EfficientSAM", "size": 16.0},
    # SAM2 family
    {"name": "SAM2.1 Large", "family": "SAM2", "size": 857.0},
    {"name": "SAM2.1 Base+", "family": "SAM2", "size": 309.0},
    {"name": "SAM2.1 Small", "family": "SAM2", "size": 176.0},
    {"name": "SAM2.1 Tiny", "family": "SAM2", "size": 149.0},
    # SAM3 family
    {"name": "SAM3", "family": "SAM3", "size": 1200.0},
    # PicoSAM variants (PicoSAM2: 4.84/1.21 MB, PicoSAM3: 5.26/1.31 MB)
    {"name": "PicoSAM2", "family": "PicoSAM2", "size": 4.84},
    {"name": "PicoSAM2 Quant", "family": "PicoSAM2", "size": 1.21},
    {"name": "PicoSAM3", "family": "PicoSAM3", "size": 5.26},
    {"name": "PicoSAM3 Quant", "family": "PicoSAM3", "size": 1.31},
]

# Color palette
family_colors = {
    'SAM1': '#7B7B7B',
    'EfficientSAM': '#6B4C9A',
    'SAM2': '#2E86AB',
    'SAM3': '#A23B72',
    'PicoSAM2': '#F18F01',
    'PicoSAM3': '#C73E1D',
}

def plot_bar_chart(results, title, filename):
    # Sort by size descending (largest first, will appear on left)
    results_sorted = sorted(results, key=lambda x: -x["size"])

    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = np.arange(len(results_sorted))
    names = [r['name'] for r in results_sorted]
    sizes = [r['size'] for r in results_sorted]
    colors = [family_colors[r['family']] for r in results_sorted]

    # Create vertical bars with gradient effect
    bars = ax.bar(x_pos, sizes, color=colors, edgecolor='white', linewidth=2, width=0.75)

    # Add size labels on top of bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        # Position label above bar
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.15,
                f'{size:.1f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333',
                rotation=0)

    # Use log scale for y-axis
    ax.set_yscale('log')
    ax.set_ylabel('Model Size (MB)', fontweight='bold', labelpad=10)

    # Set x-axis labels with rotation
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)

    # Grid styling - only horizontal
    ax.grid(True, which='major', axis='y', ls='-', alpha=0.3, color='gray')
    ax.grid(True, which='minor', axis='y', ls=':', alpha=0.2, color='gray')
    ax.set_axisbelow(True)

    # Extend y-axis to fit labels
    ax.set_ylim(0.5, max(sizes) * 2.5)

    # Add legend for families
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='white', linewidth=1.5, label=family)
                       for family, color in family_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', title='Model Family',
              fontsize=11, title_fontsize=12, framealpha=0.95, labelspacing=1.0,
              edgecolor='#cccccc')

    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('#333333')

    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "images"), exist_ok=True)

    pdf_path = os.path.join(script_dir, "images", filename.replace('.png', '.pdf'))
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300, format='pdf')

    png_path = os.path.join(script_dir, "images", filename)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, format='png')

    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_bubble_chart(results, title, filename):
    # Sort by size descending
    results_sorted = sorted(results, key=lambda x: -x["size"])

    fig, ax = plt.subplots(figsize=(16, 5))

    x_positions = np.arange(len(results_sorted))
    sizes = [r['size'] for r in results_sorted]
    max_size = max(sizes)

    # Scale bubble sizes (area proportional to size)
    bubble_sizes = [(s / max_size) * 4000 for s in sizes]
    colors = [family_colors[r['family']] for r in results_sorted]

    for i, (r, bubble_size, color) in enumerate(zip(results_sorted, bubble_sizes, colors)):
        ax.scatter(x_positions[i], 0, s=bubble_size, color=color,
                   alpha=0.85, edgecolor='white', linewidth=2, zorder=5)
        # Name label
        ax.text(x_positions[i], 0.55, r["name"], ha='center', va='bottom',
                fontsize=10, fontweight='medium', rotation=35)
        # Size label
        ax.text(x_positions[i], -0.35, f"{r['size']:.1f} MB", ha='center', va='top',
                fontsize=9, color='#444444')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.8, 1.0)
    ax.set_xlim(-1, len(results_sorted))
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)

    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('white')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='white', label=family)
                       for family, color in family_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', title='Model Family',
              fontsize=10, title_fontsize=11, framealpha=0.95, ncol=2)

    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "images"), exist_ok=True)

    pdf_path = os.path.join(script_dir, "images", filename.replace('.png', '.pdf'))
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300, format='pdf')

    png_path = os.path.join(script_dir, "images", filename)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, format='png')

    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# Create plots
plot_bar_chart(results, "Model Size Comparison", "model_size_bar_chart.png")
plot_bubble_chart(results, "Model Size Comparison", "model_size_bubble_chart.png")

# Compact version (smaller models only)
tiny_size = next(r['size'] for r in results if r['name'] == 'SAM2.1 Tiny')
results_compact = [r for r in results if r['size'] <= tiny_size]
plot_bubble_chart(results_compact, "Compact Models Size Comparison", "model_size_bubble_compact.png")
