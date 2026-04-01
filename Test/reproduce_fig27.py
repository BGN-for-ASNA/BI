import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

def draw_posterior_link(ax, node1_pos, node2_pos, max_curvature_offset, num_lines, cmap, global_curvature=0.0):
    """
    Draws a dense bundle of "fiber-like" lines (hair effect).
    global_curvature: offsets the entire bundle from the straight line.
    """
    base_teal = '#0D506E'
    fiber_count = max(num_lines, 400)
    
    for i in range(fiber_count):
        # Distribution around a global curvature offset
        curvature = global_curvature + np.random.uniform(-max_curvature_offset, max_curvature_offset)
        
        is_highlight = (np.random.random() < 0.25)
        # Distance to the *center* of the bundle (which is now global_curvature)
        dist_to_bundle_center = abs((curvature - global_curvature) / (max_curvature_offset + 1e-9))
        
        if is_highlight:
            color_idx = dist_to_bundle_center * 0.8
            line_color = cmap(color_idx)
            alpha = (1.0 - dist_to_bundle_center**1.5) * 0.6 + 0.1
            lw = 0.5
        else:
            line_color = base_teal
            alpha = (1.0 - dist_to_bundle_center**0.8) * 0.3 + 0.05
            lw = 0.35
            
        mid_point = (node1_pos + node2_pos) / 2
        line_vec = node2_pos - node1_pos
        perp_vec = np.array([-line_vec[1], line_vec[0]])
        if np.linalg.norm(perp_vec) > 0:
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
        
        jitter_amp = max_curvature_offset * 0.08
        ctrl_jitter = (np.random.random() - 0.5) * jitter_amp
        control_point = mid_point + curvature * perp_vec + ctrl_jitter * perp_vec
        
        t = np.linspace(0, 1, 80) # Slightly fewer points for performance
        curve_points = np.array([(1-t_i)**2 * node1_pos + 2*(1-t_i)*t_i * control_point + t_i**2 * node2_pos for t_i in t])
        
        ax.plot(curve_points[:, 0], curve_points[:, 1], color=line_color, linewidth=lw, alpha=alpha, zorder=1)

# --- 1. Main Setup ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 12))
background_color = '#1A1A2E' # Updated background color
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# Very subtle grid matching Solarized Dark palette
ax.grid(color='#1b1f27', linestyle='-', linewidth=0.5, alpha=0.15)

# --- 2. Create Custom Colormap (Full Spectrum Neon) ---
# Transition across the full neon spectrum for vibrancy
colors = [
    (0.0, '#00F5FF'), # Cyan
    (0.2, '#1E90FF'), # Dodger Blue
    (0.4, '#7B68EE'), # Medium Slate Blue
    (0.6, '#9400D3'), # Dark Violet
    (0.8, '#FF1493'), # Deep Pink
    (1.0, '#FFD700')  # Gold/Orange
]
custom_cmap = LinearSegmentedColormap.from_list('neon_spectrum', colors)

# --- 3. Define Tree Topology and Manual Positions ---
# Nodes: Root=0, Internal=1,2,3,4,5,6, Taxa=7-16
node_positions = {
    0: np.array([0.0, 0.0]), # Root
    
    1: np.array([0.3, 0.2]), # Upper branch 1
    2: np.array([0.3, -0.2]),# Lower branch 1
    
    3: np.array([0.6, 0.35]),# Clade A parent
    4: np.array([0.6, 0.0]), # Middle internal
    5: np.array([0.6, -0.3]),# Clade D parent
    
    # Clade A
    6: np.array([0.8, 0.45]), # Clade A junction
}

# Taxa positions (right side)
taxa_x = 1.0
taxa_y = np.linspace(0.5, -0.5, 10)
for i in range(10):
    node_positions[7+i] = np.array([taxa_x, taxa_y[i]])

# Define edges with uncertainty levels and global curvature for organic "sweep"
# (node1, node2, uncertainty, global_curvature)
# Uncertainty (spread) increased to occupy more space
edges_data = [
    (0, 1, 0.12, 0.08), (1, 3, 0.08, 0.06), (3, 6, 0.06, 0.03),
    # Clade A taxa
    (6, 7, 0.06, 0.03), (6, 8, 0.07, 0.015), (6, 9, 0.08, -0.015), (6, 10, 0.09, -0.03), 
    
    (0, 2, 0.18, -0.12), (2, 5, 0.14, -0.1), # Lower (wide spread)
    (1, 4, 0.20, -0.04), # Cross branch (very wide)
    
    (4, 11, 0.12, 0.08), (4, 12, 0.14, 0.0), (4, 13, 0.18, -0.08), # Middle taxa
    
    (5, 14, 0.08, 0.06), (5, 15, 0.1, 0.0), (5, 16, 0.12, -0.06)  # Lower taxa
]

# --- 4. Draw Edges ---
for n1, n2, unc, gc in edges_data:
    draw_posterior_link(
        ax=ax,
        node1_pos=node_positions[n1],
        node2_pos=node_positions[n2],
        max_curvature_offset=unc,
        num_lines=45,
        cmap=custom_cmap,
        global_curvature=gc
    )

# --- 5. Draw Nodes with Glow ---
for i, pos in node_positions.items():
    # Glow effect
    ax.scatter(pos[0], pos[1], s=400, color='#61AFFF', alpha=0.1, zorder=3)
    ax.scatter(pos[0], pos[1], s=200, color='#61AFFF', alpha=0.2, zorder=3)
    # Core
    ax.scatter(pos[0], pos[1], s=80, color='white', edgecolors='#61AFFF', linewidth=1.5, zorder=4)

    # Clade Labels
    if i == 6:
        ax.text(pos[0], pos[1] + 0.03, 'Clade A', color='white', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    elif i == 4:
        ax.text(pos[0], pos[1] + 0.03, 'Clade B', color='white', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    elif i == 5:
        ax.text(pos[0], pos[1] - 0.03, 'Clade C', color='white', 
                ha='center', va='top', fontsize=11, fontweight='bold')

    # Taxa Labels
    if i >= 7:
        ax.text(pos[0] + 0.02, pos[1], f'Taxon {i-6}', color='white', 
                va='center', fontsize=12, fontweight='light', fontfamily='sans-serif')

# Probability labels near some nodes
prob_labels = [
    (3, 'P=0.99'), (4, 'P=0.58'), (5, 'P=0.70'), (1, 'P=0.99')
]
for node_idx, txt in prob_labels:
    pos = node_positions[node_idx]
    ax.text(pos[0] - 0.05, pos[1] + 0.02, txt, color='#94a3b8', 
            fontsize=10, fontstyle='italic', ha='right')

# --- 6. Finalize ---
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('Test/reproduce_fig27.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Successfully generated Test/reproduce_fig27.png")
