import numpy as np
import jax.numpy as jnp
from typing import Tuple

def create_homogeneous_mask(H: int, W: int) -> jnp.ndarray:
    """Create open homogeneous grid mask (all 1.0)."""
    return jnp.ones((H, W), dtype=jnp.float32)

def create_wall_obstacle_mask(
    H: int = 256,
    W: int = 256,
    num_barriers: int = 2,
    wall_thickness: int = 8,
    passage_width: int = 20
) -> jnp.ndarray:
    """
    Create static wall obstacle mask with narrow passages/corridors.
    1.0 = open/passable, 0.0 = static wall obstacle.
    """
    mask = np.ones((H, W), dtype=np.float32)
    
    spacing = W // (num_barriers + 1)
    
    for b in range(1, num_barriers + 1):
        x_wall = b * spacing
        x_start = max(0, x_wall - wall_thickness // 2)
        x_end = min(W, x_wall + wall_thickness // 2)
        
        # Build vertical wall
        mask[:, x_start:x_end] = 0.0
        
        # Create 2 narrow passages in the wall
        p1_center = H // 3
        p2_center = (2 * H) // 3
        
        p1_start = max(0, p1_center - passage_width // 2)
        p1_end = min(H, p1_center + passage_width // 2)
        
        p2_start = max(0, p2_center - passage_width // 2)
        p2_end = min(H, p2_center + passage_width // 2)
        
        mask[p1_start:p1_end, x_start:x_end] = 1.0
        mask[p2_start:p2_end, x_start:x_end] = 1.0
        
    return jnp.array(mask, dtype=jnp.float32)

def create_topological_maze_mask(
    H: int = 256,
    W: int = 256,
    border_thickness: int = 8,
    corridor_width: int = 28,
    wall_thickness: int = 8
) -> jnp.ndarray:
    """
    Create a 2D topological labyrinth with:
    - Impassable outer perimeter borders (to prevent periodic toroidal shortcuts).
    - Start Chamber (Left) and Goal Chamber (Right).
    - Short Path (Upper Corridor).
    - Long Meandering Detour (Lower Corridor).
    - Two Dead-End Cul-de-Sacs (Dead End 1 North, Dead End 2 South).
    """
    mask = np.zeros((H, W), dtype=np.float32)
    sy, sx = H / 256.0, W / 256.0
    
    def r_y(y1, y2): return int(round(y1 * sy)), int(round(y2 * sy))
    def r_x(x1, x2): return int(round(x1 * sx)), int(round(x2 * sx))
    
    # 1. Carve Start Chamber (Left)
    y1, y2 = r_y(90, 166); x1, x2 = r_x(12, 55); mask[y1:y2, x1:x2] = 1.0
    
    # 2. Carve Goal Chamber (Right)
    y1, y2 = r_y(90, 166); x1, x2 = r_x(201, 244); mask[y1:y2, x1:x2] = 1.0
    
    # 3. Main Central Horizontal Hub Corridor from Start
    y1, y2 = r_y(114, 142); x1, x2 = r_x(50, 100); mask[y1:y2, x1:x2] = 1.0
    
    # 4. Short Upper Path: Junction at x=95 -> Up to y=60 -> Across to Goal
    y1, y2 = r_y(50, 120); x1, x2 = r_x(85, 115); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(48, 78); x1, x2 = r_x(100, 210); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(65, 120); x1, x2 = r_x(190, 215); mask[y1:y2, x1:x2] = 1.0
    
    # 5. Long Lower Meandering Detour: Junction at x=95 -> Down to y=190 -> Across to Goal
    y1, y2 = r_y(135, 205); x1, x2 = r_x(85, 115); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(180, 208); x1, x2 = r_x(100, 150); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(180, 230); x1, x2 = r_x(135, 165); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(202, 230); x1, x2 = r_x(150, 200); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(135, 215); x1, x2 = r_x(185, 215); mask[y1:y2, x1:x2] = 1.0
    
    # 6. Dead-End Cul-de-Sac 1 (Branching North from Upper Path)
    y1, y2 = r_y(14, 60); x1, x2 = r_x(140, 168); mask[y1:y2, x1:x2] = 1.0
    
    # 7. Dead-End Cul-de-Sac 2 (Branching East from Start Hub)
    y1, y2 = r_y(114, 142); x1, x2 = r_x(95, 138); mask[y1:y2, x1:x2] = 1.0
    
    # Enforce strict outer border
    b = max(2, int(round(border_thickness * (sy + sx) / 2.0)))
    mask[:b, :] = 0.0
    mask[-b:, :] = 0.0
    mask[:, :b] = 0.0
    mask[:, -b:] = 0.0
    
    return jnp.array(mask, dtype=jnp.float32)

def create_dynamic_rerouting_mask(
    H: int = 256,
    W: int = 256,
    gate_closed: bool = False,
    border_thickness: int = 8
) -> jnp.ndarray:
    """
    Create a dual-corridor environment with a dynamic switchable gate.
    - Start Chamber at Left, Goal Chamber at Right.
    - Corridor North (y ≈ 65, primary faster route).
    - Corridor South (y ≈ 190, backup route).
    - At x=128 in Corridor North, a switchable gate can close dynamically mid-simulation.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    sy, sx = H / 256.0, W / 256.0
    
    def r_y(y1, y2): return int(round(y1 * sy)), int(round(y2 * sy))
    def r_x(x1, x2): return int(round(x1 * sx)), int(round(x2 * sx))
    
    # Start and Goal Chambers
    y1, y2 = r_y(85, 170); x1, x2 = r_x(12, 55); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(85, 170); x1, x2 = r_x(201, 244); mask[y1:y2, x1:x2] = 1.0
    
    # Left Bifurcation Junction
    y1, y2 = r_y(55, 200); x1, x2 = r_x(48, 80); mask[y1:y2, x1:x2] = 1.0
    
    # Corridor North (Primary)
    y1, y2 = r_y(50, 80); x1, x2 = r_x(75, 210); mask[y1:y2, x1:x2] = 1.0
    
    # Corridor South (Secondary Backup)
    y1, y2 = r_y(175, 205); x1, x2 = r_x(75, 210); mask[y1:y2, x1:x2] = 1.0
    
    # Right Convergence Junction
    y1, y2 = r_y(55, 200); x1, x2 = r_x(185, 215); mask[y1:y2, x1:x2] = 1.0
    
    # Dynamic Gate in North Corridor
    if gate_closed:
        y1, y2 = r_y(50, 80); x1, x2 = r_x(120, 138); mask[y1:y2, x1:x2] = 0.0
        
    # Enforce strict outer borders
    b = max(2, int(round(border_thickness * (sy + sx) / 2.0)))
    mask[:b, :] = 0.0
    mask[-b:, :] = 0.0
    mask[:, :b] = 0.0
    mask[:, -b:] = 0.0
    
    return jnp.array(mask, dtype=jnp.float32)

def create_multi_terminal_city_mask(
    H: int = 256,
    W: int = 256,
    border_thickness: int = 6
) -> jnp.ndarray:
    """
    Create a multi-terminal territory inspired by the Tokyo Rail experiment (Tero et al. 2010):
    - Central colony spawn area around (128, 128).
    - 4 distributed regional city hubs (NW, NE, SE, SW).
    - 8 complex mountain barriers & archipelago islands between hubs forcing multi-stage branching vein synthesis.
    """
    mask = np.ones((H, W), dtype=np.float32)
    sy, sx = H / 256.0, W / 256.0
    
    b = max(2, int(round(border_thickness * (sy + sx) / 2.0)))
    mask[:b, :] = 0.0
    mask[-b:, :] = 0.0
    mask[:, :b] = 0.0
    mask[:, -b:] = 0.0
    
    def r_y(y1, y2): return int(round(y1 * sy)), int(round(y2 * sy))
    def r_x(x1, x2): return int(round(x1 * sx)), int(round(x2 * sx))
    
    # 4 Cardinal mountain barriers
    y1, y2 = r_y(25, 85); x1, x2 = r_x(105, 151); mask[y1:y2, x1:x2] = 0.0 # North
    y1, y2 = r_y(171, 231); x1, x2 = r_x(105, 151); mask[y1:y2, x1:x2] = 0.0 # South
    y1, y2 = r_y(105, 151); x1, x2 = r_x(25, 85); mask[y1:y2, x1:x2] = 0.0 # West
    y1, y2 = r_y(105, 151); x1, x2 = r_x(171, 231); mask[y1:y2, x1:x2] = 0.0 # East
    
    # 4 Diagonal archipelago islands (forcing narrow bottleneck passes)
    y1, y2 = r_y(65, 90); x1, x2 = r_x(65, 90); mask[y1:y2, x1:x2] = 0.0 # NW Island
    y1, y2 = r_y(65, 90); x1, x2 = r_x(166, 191); mask[y1:y2, x1:x2] = 0.0 # NE Island
    y1, y2 = r_y(166, 191); x1, x2 = r_x(65, 90); mask[y1:y2, x1:x2] = 0.0 # SW Island
    y1, y2 = r_y(166, 191); x1, x2 = r_x(166, 191); mask[y1:y2, x1:x2] = 0.0 # SE Island
    
    return jnp.array(mask, dtype=jnp.float32)

def create_swarm_funnel_mask(
    H: int = 256,
    W: int = 256,
    bottleneck_width: int = 26,
    border_thickness: int = 8
) -> jnp.ndarray:
    """
    Create a dual-chamber converging funnel environment for swarm rendezvous:
    - Chamber 1 (North-West) and Chamber 2 (South-West).
    - Converging angled funnel towards a central narrow bottleneck.
    - Opens into Large East Destination Chamber.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    sy, sx = H / 256.0, W / 256.0
    
    def r_y(y1, y2): return int(round(y1 * sy)), int(round(y2 * sy))
    def r_x(x1, x2): return int(round(x1 * sx)), int(round(x2 * sx))
    
    # North-West and South-West Spawn Chambers
    y1, y2 = r_y(35, 105); x1, x2 = r_x(14, 75); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(150, 220); x1, x2 = r_x(14, 75); mask[y1:y2, x1:x2] = 1.0
    
    # Angled converging funnels into bottleneck
    y1, y2 = r_y(60, 135); x1, x2 = r_x(70, 128); mask[y1:y2, x1:x2] = 1.0
    y1, y2 = r_y(120, 195); x1, x2 = r_x(70, 128); mask[y1:y2, x1:x2] = 1.0
    
    # Bottleneck corridor
    bw = int(round(bottleneck_width * sy))
    by_start = int(round(128 * sy)) - bw // 2
    by_end = int(round(128 * sy)) + bw // 2
    x1, x2 = r_x(120, 148)
    mask[by_start:by_end, x1:x2] = 1.0
    
    # Destination Chamber (East)
    y1, y2 = r_y(45, 210); x1, x2 = r_x(145, 242); mask[y1:y2, x1:x2] = 1.0
    
    # Enforce borders
    b = max(2, int(round(border_thickness * (sy + sx) / 2.0)))
    mask[:b, :] = 0.0
    mask[-b:, :] = 0.0
    mask[:, :b] = 0.0
    mask[:, -b:] = 0.0
    
    return jnp.array(mask, dtype=jnp.float32)

def solve_laplace_corridor_potential(
    wall_mask: np.ndarray,
    goal_mask: np.ndarray,
    max_iters: int = 350,
    decay: float = 0.9995
) -> np.ndarray:
    """
    Solve steady-state Laplace/Poisson diffusion in open corridors (wall_mask == 1.0):
    ∇² C(x,y) = 0 with Dirichlet boundary condition C = 1.0 at goal_mask and C = 0.0 at walls.
    Produces a smooth harmonic potential field whose gradient ∇C points along corridors toward the goal.
    """
    H, W = wall_mask.shape
    C = np.zeros((H, W), dtype=np.float32)
    goal_bool = (goal_mask > 0.5)
    passable = (wall_mask > 0.5)
    
    # Set goal potential
    C[goal_bool] = 1.0
    
    for _ in range(max_iters):
        # 4-neighbor average
        c_up = np.roll(C, shift=1, axis=0)
        c_down = np.roll(C, shift=-1, axis=0)
        c_left = np.roll(C, shift=1, axis=1)
        c_right = np.roll(C, shift=-1, axis=1)
        
        c_new = 0.25 * (c_up + c_down + c_left + c_right) * decay
        # Zero out inside walls
        c_new[~passable] = 0.0
        # Fix Dirichlet source at goal
        c_new[goal_bool] = 1.0
        C = c_new
        
    # Final normalization
    max_val = np.max(C)
    if max_val > 1e-6:
        C = C / max_val
        
    return C.astype(np.float32)

def solve_geodesic_corridor_potential(
    wall_mask: np.ndarray,
    goal_mask: np.ndarray
) -> np.ndarray:
    """
    Compute exact geodesic distance potential field D(x,y) inside open corridors:
    Uses BFS wavefront propagation across 8-connected passable pixels (wall_mask > 0.5) from goal seeds.
    Returns normalized potential field C(x,y) = 1.0 - D(x,y) / D_max in [0.0, 1.0].
    Guarantees steady, non-zero gradient vector pointing along shortest corridor paths toward the goal.
    """
    from collections import deque
    H, W = wall_mask.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    queue = deque()
    
    passable = (wall_mask > 0.5)
    goal_cells = (goal_mask > 0.5) & passable
    
    for y in range(H):
        for x in range(W):
            if goal_cells[y, x]:
                dist[y, x] = 0.0
                queue.append((y, x))
                
    while queue:
        cy, cx = queue.popleft()
        cd = dist[cy, cx]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and passable[ny, nx]:
                step_cost = 1.414 if (dy != 0 and dx != 0) else 1.0
                if cd + step_cost < dist[ny, nx]:
                    dist[ny, nx] = cd + step_cost
                    queue.append((ny, nx))
                    
    valid_dists = dist[passable & np.isfinite(dist)]
    max_d = np.max(valid_dists) if len(valid_dists) > 0 else 1.0
    
    pot = np.where(passable & np.isfinite(dist), 1.0 - dist / max_d, 0.0).astype(np.float32)
    return pot

