import unittest
import numpy as np
import jax
import jax.numpy as jnp

from core.environment import (
    create_topological_maze_mask,
    create_dynamic_rerouting_mask,
    create_multi_terminal_city_mask,
    create_swarm_funnel_mask,
    solve_laplace_corridor_potential
)
from experiments.supplementary.run_topological_transport import (
    run_scenario_maze,
    run_scenario_dynamic_reroute,
    run_scenario_tokyo_rail,
    run_scenario_swarm_channeling
)

class TestTopologicalTransport(unittest.TestCase):
    def setUp(self):
        self.H, self.W = 128, 128
        
    def test_topological_maze_mask_geometry(self):
        mask = create_topological_maze_mask(self.H, self.W, border_thickness=4)
        self.assertEqual(mask.shape, (self.H, self.W))
        # Borders must be solid wall (0.0)
        self.assertTrue(jnp.all(mask[:4, :] == 0.0))
        self.assertTrue(jnp.all(mask[-4:, :] == 0.0))
        self.assertTrue(jnp.all(mask[:, :4] == 0.0))
        self.assertTrue(jnp.all(mask[:, -4:] == 0.0))
        # Open corridors must exist
        self.assertGreater(float(jnp.sum(mask == 1.0)), 100)
        
    def test_dynamic_rerouting_mask(self):
        mask_open = create_dynamic_rerouting_mask(self.H, self.W, gate_closed=False)
        mask_closed = create_dynamic_rerouting_mask(self.H, self.W, gate_closed=True)
        self.assertEqual(mask_open.shape, (self.H, self.W))
        self.assertEqual(mask_closed.shape, (self.H, self.W))
        # Closed mask must have fewer open cells than open mask
        self.assertGreater(float(jnp.sum(mask_open)), float(jnp.sum(mask_closed)))
        
    def test_laplace_potential_solver(self):
        wall_mask = np.ones((self.H, self.W), dtype=np.float32)
        wall_mask[:4, :] = 0.0
        wall_mask[-4:, :] = 0.0
        wall_mask[:, :4] = 0.0
        wall_mask[:, -4:] = 0.0
        
        goal_mask = np.zeros((self.H, self.W), dtype=np.float32)
        goal_mask[64, 100] = 1.0
        
        pot = solve_laplace_corridor_potential(wall_mask, goal_mask, max_iters=50)
        self.assertEqual(pot.shape, (self.H, self.W))
        self.assertAlmostEqual(float(pot[64, 100]), 1.0, places=3)
        # Point far from goal should have lower potential
        self.assertLess(float(pot[64, 20]), float(pot[64, 90]))
        # Walls must be 0.0
        self.assertAlmostEqual(float(pot[0, 0]), 0.0, places=5)
        
    def test_smoke_scenario_maze(self):
        # Quick 20-step smoke test
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_scenario_maze(seed=42, grid_size=64, steps=20, sample_interval=5, output_dir=tmp_dir)
            self.assertIn("scenario", res)
            self.assertEqual(res["scenario"], "continuous_maze")
            self.assertIn("mass_preservation_ratio", res)

if __name__ == "__main__":
    unittest.main()
