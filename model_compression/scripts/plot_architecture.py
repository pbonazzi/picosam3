import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# CVPR-style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'text.usetex': False,
})

def draw_picosam3_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    # Colors
    colors = {
        'input': '#4CAF50',      # Green
        'encoder': '#2196F3',    # Blue
        'down': '#1976D2',       # Dark blue
        'bottleneck': '#9C27B0', # Purple
        'decoder': '#FF9800',    # Orange
        'skip': '#607D8B',       # Gray
        'eca': '#E91E63',        # Pink
        'output': '#F44336',     # Red
    }

    box_height = 0.6

    # Helper function to draw a block
    def draw_block(x, y, width, height, color, label, fontsize=9):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white', wrap=True)
        return x + width/2, y + height/2

    # Helper for arrows
    def draw_arrow(start, end, color='#333333', style='->', connectionstyle='arc3,rad=0'):
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=style, color=color, linewidth=2,
            connectionstyle=connectionstyle, mutation_scale=15
        )
        ax.add_patch(arrow)

    # Title
    ax.text(8, 9.5, 'PicoSAM3 Architecture', fontsize=20, fontweight='bold',
            ha='center', va='center', color='#333333')
    ax.text(8, 9.0, '1.37M parameters | 5.26 MB', fontsize=12,
            ha='center', va='center', color='#666666')

    # ========== INPUT ==========
    x_input = 0.5
    y_base = 5
    cx, cy = draw_block(x_input, y_base - 0.3, 1.0, 0.6, colors['input'], 'Input\n3×96×96', 8)

    # ========== ENCODER (left side, going down) ==========
    encoder_x = 2.0
    encoder_stages = [
        ('Enc 1\nDWConv\n3→48', 48),
        ('Enc 2\nDWConv\n48→96', 96),
        ('Enc 3\nDWConv\n96→160', 160),
        ('Enc 4\nDWConv\n160→256', 256),
    ]

    encoder_positions = []
    y_enc = 7.5
    for i, (label, ch) in enumerate(encoder_stages):
        cx_enc, cy_enc = draw_block(encoder_x, y_enc, 1.4, 1.0, colors['encoder'], label, 8)
        encoder_positions.append((cx_enc, cy_enc, y_enc))

        # Down arrow between encoder stages
        if i < len(encoder_stages) - 1:
            # Draw down arrow with "↓2" label
            ax.annotate('', xy=(encoder_x + 0.7, y_enc - 0.3),
                       xytext=(encoder_x + 0.7, y_enc),
                       arrowprops=dict(arrowstyle='->', color=colors['down'], lw=2))
            ax.text(encoder_x + 1.0, y_enc - 0.15, '↓2', fontsize=8, color=colors['down'], fontweight='bold')

        y_enc -= 1.8

    # Arrow from input to encoder
    draw_arrow((x_input + 1.0, y_base), (encoder_x, 7.5 + 0.5))

    # ========== BOTTLENECK ==========
    bottleneck_x = 4.5
    bottleneck_y = 1.5

    # Draw bottleneck components
    draw_block(bottleneck_x, bottleneck_y + 1.2, 1.8, 0.8, colors['bottleneck'], 'DWConv\n256→320', 8)
    draw_block(bottleneck_x, bottleneck_y, 1.8, 0.8, colors['bottleneck'], 'Dilated DW\nd=2', 8)
    draw_block(bottleneck_x, bottleneck_y - 1.2, 1.8, 0.8, colors['bottleneck'], '1×1 Conv\n320→320', 8)

    # Bottleneck label
    ax.text(bottleneck_x + 0.9, bottleneck_y + 2.3, 'Bottleneck', fontsize=10,
            fontweight='bold', ha='center', color=colors['bottleneck'])

    # Arrow from last encoder to bottleneck
    draw_arrow((encoder_x + 1.4, encoder_positions[-1][2] + 0.5),
               (bottleneck_x, bottleneck_y + 1.6))

    # ========== DECODER (right side, going up) ==========
    decoder_x = 7.5
    decoder_stages = [
        ('Dec 1\nUp+DW\n320→192', 192),
        ('Dec 2\nUp+DW\n192→128', 128),
        ('Dec 3\nUp+DW\n128→80', 80),
        ('Dec 4\nUp+DW\n80→40', 40),
    ]

    decoder_positions = []
    y_dec = 1.5
    for i, (label, ch) in enumerate(decoder_stages):
        cx_dec, cy_dec = draw_block(decoder_x, y_dec, 1.4, 1.0, colors['decoder'], label, 8)
        decoder_positions.append((cx_dec, cy_dec, y_dec))

        # Up arrow between decoder stages
        if i < len(decoder_stages) - 1:
            ax.annotate('', xy=(decoder_x + 0.7, y_dec + 1.3),
                       xytext=(decoder_x + 0.7, y_dec + 1.0),
                       arrowprops=dict(arrowstyle='->', color=colors['decoder'], lw=2))
            ax.text(decoder_x + 1.0, y_dec + 1.15, '↑2', fontsize=8, color=colors['decoder'], fontweight='bold')

        y_dec += 1.8

    # Arrow from bottleneck to decoder
    draw_arrow((bottleneck_x + 1.8, bottleneck_y + 0.4),
               (decoder_x, 1.5 + 0.5))

    # ========== SKIP CONNECTIONS ==========
    skip_labels = ['256→192', '160→128', '96→80', '48→40']
    for i, (enc_pos, dec_pos) in enumerate(zip(reversed(encoder_positions), decoder_positions)):
        enc_x, enc_y, enc_ypos = enc_pos
        dec_x, dec_y, dec_ypos = dec_pos

        # Draw curved skip connection
        mid_x = (encoder_x + 1.4 + decoder_x) / 2 + 2

        # Skip connection box
        skip_y = (enc_ypos + dec_ypos) / 2 + 0.5
        draw_block(10.0, skip_y, 1.2, 0.5, colors['skip'], f'1×1\n{skip_labels[i]}', 7)

        # Arrows for skip connection
        ax.annotate('', xy=(10.0, skip_y + 0.25),
                   xytext=(encoder_x + 1.4, enc_ypos + 0.5),
                   arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.5,
                                 connectionstyle='arc3,rad=-0.2'))
        ax.annotate('', xy=(decoder_x + 1.4, dec_ypos + 0.5),
                   xytext=(10.0 + 1.2, skip_y + 0.25),
                   arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.5,
                                 connectionstyle='arc3,rad=-0.2'))

    # ========== ECA BLOCK ==========
    eca_x = 12.0
    eca_y = 7.0
    draw_block(eca_x, eca_y, 1.5, 1.2, colors['eca'], 'ECA\nAttention\n40ch', 9)

    # Arrow from last decoder to ECA
    draw_arrow((decoder_x + 1.4, decoder_positions[-1][2] + 0.5),
               (eca_x, eca_y + 0.6))

    # ========== REFINEMENT HEAD ==========
    refine_x = 14.0
    draw_block(refine_x, eca_y + 0.6, 1.4, 0.8, colors['output'], 'DWConv\n40→40', 8)
    draw_block(refine_x, eca_y - 0.4, 1.4, 0.8, colors['output'], '1×1 Conv\n40→1', 8)

    ax.text(refine_x + 0.7, eca_y + 1.6, 'Refine', fontsize=10,
            fontweight='bold', ha='center', color=colors['output'])

    # Arrow from ECA to refinement
    draw_arrow((eca_x + 1.5, eca_y + 0.6), (refine_x, eca_y + 1.0))

    # Arrow between refine blocks
    ax.annotate('', xy=(refine_x + 0.7, eca_y - 0.4),
               xytext=(refine_x + 0.7, eca_y + 0.6),
               arrowprops=dict(arrowstyle='->', color=colors['output'], lw=2))

    # ========== OUTPUT ==========
    draw_block(refine_x, eca_y - 1.6, 1.4, 0.6, colors['input'], 'Output\n1×96×96', 8)
    ax.annotate('', xy=(refine_x + 0.7, eca_y - 1.6),
               xytext=(refine_x + 0.7, eca_y - 0.4),
               arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # ========== LEGEND ==========
    legend_elements = [
        mpatches.Patch(facecolor=colors['encoder'], label='Encoder (DWConv)', edgecolor='white'),
        mpatches.Patch(facecolor=colors['bottleneck'], label='Bottleneck (Dilated)', edgecolor='white'),
        mpatches.Patch(facecolor=colors['decoder'], label='Decoder (Upsample+DWConv)', edgecolor='white'),
        mpatches.Patch(facecolor=colors['skip'], label='Skip Connection (1×1)', edgecolor='white'),
        mpatches.Patch(facecolor=colors['eca'], label='ECA Attention', edgecolor='white'),
        mpatches.Patch(facecolor=colors['output'], label='Refinement Head', edgecolor='white'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              framealpha=0.95, edgecolor='#cccccc', title='Components', title_fontsize=11)

    # Key features annotation
    features_text = (
        "Key Features:\n"
        "• Depthwise Separable Convolutions\n"
        "• Dilated Conv in Bottleneck (d=2)\n"
        "• ECA Channel Attention\n"
        "• U-Net Skip Connections\n"
        "• Refinement Head for Sharp Masks"
    )
    ax.text(0.5, 2.5, features_text, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()

    # Save
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "images"), exist_ok=True)

    pdf_path = os.path.join(script_dir, "images", "picosam3_architecture.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300, format='pdf')

    png_path = os.path.join(script_dir, "images", "picosam3_architecture.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300, format='png')

    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

if __name__ == "__main__":
    draw_picosam3_architecture()
