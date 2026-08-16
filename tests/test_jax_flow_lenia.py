import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts,
    initialize_multi_patch_state, flow_lenia_step_single, run_flow_lenia_rollout
)
from core.metrics import (
    compute_evolutionary_activity, compute_compression_complexity,
    compute_multi_scale_entropy, evaluate_run_metrics
)
from core.imgep import IMGEPArchive, sample_random_config, evaluate_single_config_jax
from core.environment import create_wall_obstacle_mask

class TestJAXFlowLenia(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(42)
        self.H, self.W = 128, 128
        self.K = 3
        self.radii = jnp.array([6.0, 9.0, 12.0], dtype=jnp.float32)
        self.kernel_ffts = precompute_kernel_ffts(self.radii, self.H, self.W)

    def test_precompute_fft_shape(self):
        # rfft2 of (H, W) real image has shape (H, W // 2 + 1)
        expected_shape = (self.K, self.H, self.W // 2 + 1)
        self.assertEqual(self.kernel_ffts.shape, expected_shape)

    def test_localized_multi_patch_initialization(self):
        state = initialize_multi_patch_state(
            self.rng_key, self.H, self.W, C=1, K=self.K, n_patches=3, kernel_radii=self.radii
        )
        mass = state.mass[0]
        
        # Grid must NOT be filled with noise everywhere
        zero_cells = jnp.sum(mass == 0.0)
        total_cells = self.H * self.W
        # At least 70% of grid should be empty vacuum
        self.assertGreater(float(zero_cells / total_cells), 0.70)
        
        # Total mass must be strictly positive
        self.assertGreater(float(jnp.sum(mass)), 1.0)

    def test_mass_conservation_in_step(self):
        state = initialize_multi_patch_state(
            self.rng_key, self.H, self.W, C=1, K=self.K, n_patches=2, kernel_radii=self.radii
        )
        params = FlowLeniaParams(
            mu=jnp.array([0.15, 0.15, 0.15]),
            sigma=jnp.array([0.015, 0.015, 0.015]),
            weights=jnp.array([0.33, 0.33, 0.33])
        )
        
        init_mass_sum = float(jnp.sum(state.mass[0]))
        
        # Run 5 steps
        curr_state = state
        key = self.rng_key
        for _ in range(5):
            key, subk = random.split(key)
            curr_state = flow_lenia_step_single(curr_state, self.kernel_ffts, params, subk)
            
        step_mass_sum = float(jnp.sum(curr_state.mass[0]))
        
        # Mass must be conserved within float32 precision (< 1e-4 relative error)
        self.assertAlmostEqual(init_mass_sum, step_mass_sum, places=3)

    def test_gene_wise_mixing_preserves_bounds(self):
        state = initialize_multi_patch_state(
            self.rng_key, self.H, self.W, C=1, K=self.K, n_patches=3, kernel_radii=self.radii
        )
        params = FlowLeniaParams(
            mu=jnp.array([0.15, 0.15, 0.15]),
            sigma=jnp.array([0.015, 0.015, 0.015]),
            weights=jnp.array([0.33, 0.33, 0.33])
        )
        
        next_state = flow_lenia_step_single(
            state, self.kernel_ffts, params, self.rng_key, mixing_rule='gene_wise'
        )
        
        # Parameter bounds checking
        self.assertTrue(jnp.all(next_state.mu_map >= 0.01))
        self.assertTrue(jnp.all(next_state.mu_map <= 1.0))
        self.assertTrue(jnp.all(next_state.sigma_map >= 0.005))
        self.assertTrue(jnp.all(next_state.sigma_map <= 0.30))

    def test_3d_metrics_computation(self):
        mass_grid = np.random.uniform(0, 1, (128, 128)).astype(np.float32)
        sampled_mass = np.random.uniform(0, 1, (5, 1, 128, 128)).astype(np.float32)
        sampled_gid = np.random.randint(0, 3, (5, 128, 128))
        
        metrics = evaluate_run_metrics(mass_grid, sampled_mass, sampled_gid, total_steps=250, n_genomes=3)
        
        self.assertIn("ea_raw", metrics)
        self.assertIn("complexity_raw", metrics)
        self.assertIn("entropy_raw", metrics)
        self.assertGreaterEqual(metrics["ea_raw"], 0.0)
        self.assertGreater(metrics["complexity_raw"], 0)
        self.assertGreater(metrics["entropy_raw"], 0.0)

    def test_imgep_archive_and_fps(self):
        archive = IMGEPArchive()
        for i in range(10):
            cfg = {"id": i}
            m = {
                "com_displacement": float(i * 2.0),
                "ea_raw": float(i * 1.5),
                "complexity_raw": float(100 + i * 50),
                "entropy_raw": float(0.1 * i)
            }
            archive.add_trial(cfg, m)
            
        self.assertEqual(archive.size(), 10)
        
        # Test nearest neighbor search
        goal = np.array([0.5, 0.5, 0.5])
        nearest = archive.find_nearest(goal)
        self.assertTrue(0 <= nearest < 10)
        
        # Test Farthest-Point Sampling
        fps_indices = archive.select_farthest_point_sampling(k=3)
        self.assertEqual(len(fps_indices), 3)
        self.assertEqual(len(set(fps_indices)), 3) # distinct indices

    def test_wall_obstacle_mask(self):
        wall_mask = create_wall_obstacle_mask(H=128, W=128)
        self.assertEqual(wall_mask.shape, (128, 128))
        # Verify walls (0.0) and passages (1.0) exist
        self.assertTrue(jnp.any(wall_mask == 0.0))
        self.assertTrue(jnp.any(wall_mask == 1.0))

if __name__ == "__main__":
    unittest.main()
