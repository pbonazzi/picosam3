import os
import matplotlib.pyplot as plt
import matplotlib
from adjustText import adjust_text
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

# Results with model families and latency
results = [
    # SAM1 family
    {"name": "SAM-H", "family": "SAM1", "size": 2420.3, "latency": 2392},
    {"name": "SAM-L", "family": "SAM1", "size": 1174.9, "latency": 1146},
    {"name": "SAM-B", "family": "SAM1", "size": 343.3, "latency": 368.8},
    # Efficient SAM variants
    {"name": "FastSAM", "family": "EfficientSAM", "size": 275.6, "latency": 153.6},
    {"name": "EfficientSAM-Ti", "family": "EfficientSAM", "size": 118.2, "latency": 81},
    {"name": "SlimSAM-77", "family": "EfficientSAM", "size": 51.0, "latency": 110},
    {"name": "MobileSAM", "family": "EfficientSAM", "size": 37.0, "latency": 38.4},
    {"name": "TinySAM", "family": "EfficientSAM", "size": 37.0, "latency": 38.4},
    {"name": "Q-TinySAM", "family": "EfficientSAM", "size": 16.0, "latency": 24},
    # SAM2 family
    {"name": "SAM2.1 Large", "family": "SAM2", "size": 857.0, "latency": None},
    {"name": "SAM2.1 Tiny", "family": "SAM2", "size": 149.0, "latency": None},
    
    # PicoSAM variants (PicoSAM2: 4.84/1.21 MB, PicoSAM3: 5.26/1.31 MB)
    {"name": "PicoSAM2", "family": "PicoSAM2", "size": 4.84, "latency": 2.54},
    {"name": "PicoSAM2 Quant", "family": "PicoSAM2", "size": 1.21, "latency": None},
    {"name": "PicoSAM3", "family": "PicoSAM3", "size": 5.26, "latency": None},
    {"name": "PicoSAM3 Quant", "family": "PicoSAM3", "size": 1.31, "latency": None},
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

family_markers = {
    'SAM1': 'p',
    'EfficientSAM': 'h',
    'SAM2': 'o',
    'SAM3': 's',
    'PicoSAM2': '^',
    'PicoSAM3': 'D',
}

def create_latency_vs_size_plot(results, title, filename):
    fig, ax = plt.subplots(figsize=(10, 7))

    texts = []
    plotted_families = set()

    for r in results:
        size = r["size"]
        latency = r["latency"]
        family = r["family"]

        if latency is None:
            continue

        # Plot with family-based styling
        label = family if family not in plotted_families else None
        plotted_families.add(family)

        ax.scatter(
            size, latency,
            color=family_colors[family],
            marker=family_markers[family],
            s=200,
            edgecolor='white',
            linewidth=2,
            label=label,
            zorder=5,
            alpha=0.9
        )

        # Add text label
        text = ax.text(
            size, latency, f"  {r['name']}",
            fontsize=10,
            ha='left',
            va='center',
            fontweight='medium',
            bbox=dict(
                facecolor='white',
                alpha=0.85,
                edgecolor='none',
                pad=2,
                boxstyle='round,pad=0.3'
            )
        )
        texts.append(text)

    # Styling
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model Size (MB)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Latency (ms)', fontweight='bold', labelpad=10)

    # Grid styling
    ax.grid(True, which='major', ls='-', alpha=0.3, color='gray')
    ax.grid(True, which='minor', ls=':', alpha=0.2, color='gray')

    # Add trend line
    valid_data = [(r['size'], r['latency']) for r in results if r['latency'] is not None]
    if len(valid_data) > 2:
        sizes, latencies = zip(*valid_data)
        z = np.polyfit(np.log10(sizes), np.log10(latencies), 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
        y_line = 10 ** p(np.log10(x_line))
        ax.plot(x_line, y_line, '--', color='#999999', alpha=0.6, linewidth=2, zorder=1)

    # Legend
    legend = ax.legend(
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=12,
        markerscale=1.2,
        title='Model Family',
        title_fontsize=13
    )

    # Adjust text labels
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='->', color='#666666', lw=1, connectionstyle='arc3,rad=0.2'),
        expand_points=(1.5, 1.5),
        force_text=(0.5, 0.8)
    )

    # Title
    ax.set_title(title, fontweight='bold', pad=15)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

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

# Create plot
create_latency_vs_size_plot(
    results,
    "Latency vs. Model Size",
    "latency_vs_size.png"
)
