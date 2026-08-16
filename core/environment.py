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
