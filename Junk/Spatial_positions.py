import matplotlib.pyplot as plt
deep_plum ="#5B3758"       # Primary – dark purple / plum
tropical_green ="#00916E"  # Secondary – teal / tropical green
rose_pink = "#DE6C83"       # Accent – rose pink
peach_orange = "#FCB97D"    # Contrast – peach / soft orange
pastel_green = "#D4E4BC"    # Contrast – pastel green
room_dim = [8.12, 7.35, 3.00]

spatial_positions = [
    [          1.3,          6.05, room_dim[2]/2],  # 0 — upper left
    [room_dim[0]/2,          6.05, room_dim[2]/2],  # 1 — upper middle
    [         6.82,          6.05, room_dim[2]/2],  # 2 — upper right

    [          1.3, room_dim[1]/2, room_dim[2]/2],  # 3 — mid left
    [room_dim[0]/2, room_dim[1]/2, room_dim[2]/2],  # 4 — center
    [         6.82, room_dim[1]/2, room_dim[2]/2],  # 5 — mid right

    [          1.3,           1.3, room_dim[2]/2],  # 6 — bottom left
    [room_dim[0]/2,           1.3, room_dim[2]/2],  # 7 — bottom middle
    [         6.82,           1.3, room_dim[2]/2],  # 8 — bottom right
]

fig, ax = plt.subplots(figsize=(7, 6))

for i, pos in enumerate(spatial_positions):
    x, y = pos[0], pos[1]

    # Small filled dot
    dot_radius = 0.15
    dot = plt.Circle((x, y), dot_radius, color=pastel_green, ec=tropical_green, lw=1.2)
    ax.add_artist(dot)

    # Number inside the dot
    ax.text(x, y, str(i), color=tropical_green, fontsize=9, ha='center', va='center', weight='bold')

    # Outer circle with radius = 1
    outer_circle = plt.Circle((x, y), 1.0, fill=False, color=tropical_green, lw=1.2, ls='--')
    ax.add_artist(outer_circle)

ax.set_xlim(0, room_dim[0])
ax.set_ylim(0, room_dim[1])
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
