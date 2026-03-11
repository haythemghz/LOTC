import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set Academic Formatting
plt.rcParams.update({
    "text.usetex": False,  # If true, requires local LaTeX installation
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
})

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.axis('off')

# Colors
color_data_space = '#7FB3D5'
color_latent_data = '#1F618D'
color_prototypes = '#E74C3C'
color_transport = '#D68910'
color_arrow = '#34495E'

# 1. Input Data Space \mathcal{X}
circle_x = patches.Circle((-4, 0), 2, fill=True, color='#F2F3F4', ec='#BDC3C7', lw=2)
ax.add_patch(circle_x)
ax.text(-4, 2.2, r"Input Space $\mathcal{X}$", ha='center', va='bottom', fontsize=16, fontweight='bold')

np.random.seed(42)
# Generate 3 clusters of input data
n_points_per_cluster = 30
for cx, cy in [(-4.8, 0.8), (-3.2, 0.5), (-4.2, -1.0)]: # 3 clusters
    pts_x = np.random.normal(cx, 0.3, n_points_per_cluster)
    pts_y = np.random.normal(cy, 0.3, n_points_per_cluster)
    ax.scatter(pts_x, pts_y, s=15, c=color_data_space, alpha=0.7, edgecolors='white', linewidth=0.5)

# 2. Encoder Arrow \phi_\theta
arrow = patches.FancyArrowPatch((-1.5, 0), (1.5, 0), 
                                connectionstyle="arc3,rad=0.2", 
                                color=color_arrow, 
                                arrowstyle="simple,tail_width=2,head_width=8,head_length=8",
                                alpha=0.8)
ax.add_patch(arrow)
ax.text(0, 0.8, r"Encoder $\phi_\theta$", ha='center', va='center', fontsize=16, color=color_arrow)

# 3. Latent Space \mathcal{Z} (Hypersphere)
circle_z = patches.Circle((4, 0), 2, fill=False, color='#BDC3C7', lw=3, linestyle='--')
ax.add_patch(circle_z)
# Fill the inside lightly
circle_z_fill = patches.Circle((4, 0), 2, fill=True, color='#EAEDED', alpha=0.4)
ax.add_patch(circle_z_fill)
ax.text(4, 2.2, r"Latent Hypersphere $\mathcal{Z}$", ha='center', va='bottom', fontsize=16, fontweight='bold')

# 4. Latent Data Distribution \hat{\mu}
# Generate points on the sphere/circle boundary ( LOTC projection )
angles_c1 = np.random.normal(np.pi/4, 0.2, n_points_per_cluster)
angles_c2 = np.random.normal(3*np.pi/4, 0.2, n_points_per_cluster)
angles_c3 = np.random.normal(3*np.pi/2, 0.3, n_points_per_cluster * 2) # Larger cluster

all_angles = np.concatenate([angles_c1, angles_c2, angles_c3])
latent_pts_x = 4 + 2 * np.cos(all_angles)
latent_pts_y = 2 * np.sin(all_angles)

ax.scatter(latent_pts_x, latent_pts_y, s=30, c=color_latent_data, edgecolors='white', linewidth=0.5, zorder=3, label=r"Data $\hat{\mu}$")

# 5. Prototypes \nu (with mass \alpha_j)
proto_angles = [np.pi/4, 3*np.pi/4, 3*np.pi/2]
proto_masses = [0.25, 0.25, 0.5] # alpha_j
proto_sizes = [m * 1500 for m in proto_masses]

proto_x = [4 + 2 * np.cos(a) for a in proto_angles]
proto_y = [2 * np.sin(a) for a in proto_angles]

# Plot transport plan (sinkhorn lines)
for i, a_data in enumerate(all_angles):
    dx = 4 + 2 * np.cos(a_data)
    dy = 2 * np.sin(a_data)
    
    # Connect to the nearest prototype for visualization of OT
    dists = [np.sqrt((dx - px)**2 + (dy - py)**2) for px, py in zip(proto_x, proto_y)]
    closest_p = np.argmin(dists)
    
    # Faint lines to all, stronger line to closest
    for j, (px, py) in enumerate(zip(proto_x, proto_y)):
        alpha = 0.3 if j == closest_p else 0.05
        lw = 1.0 if j == closest_p else 0.5
        ax.plot([dx, px], [dy, py], color=color_transport, alpha=alpha, lw=lw, zorder=1)

ax.scatter(proto_x, proto_y, s=proto_sizes, c=color_prototypes, marker='*', edgecolors='black', linewidth=1.0, zorder=4, label=r"Prototypes $\nu$ ($\alpha_j$)")

# Labels for Measures
ax.text(proto_x[0]+0.3, proto_y[0]+0.3, r"$c_1 (\alpha_1)$", fontsize=14, color='black')
ax.text(proto_x[1]-0.4, proto_y[1]+0.3, r"$c_2 (\alpha_2)$", fontsize=14, color='black')
ax.text(proto_x[2]-0.3, proto_y[2]-0.4, r"$c_3 (\alpha_3)$", fontsize=14, color='black')

# Add legend nicely
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=14)

# Bounding box limits to crop tightly
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-2.5, 3.0)

# Formatting: Tight layout and no title
plt.tight_layout()

# Save as standard academic format High Res PNG
output_path = r"d:\Haythem\Temp\AAAA-New Research\Learned Optimal Transport Clustering A Differentiable Wasserstein Framework for Prototype Learning\paper\figures\fig1_conceptual.png"
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
plt.savefig(output_path, dpi=400, bbox_inches='tight', transparent=True)
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', transparent=True)
print(f"Successfully generated and saved academic figure successfully to {output_path}")

