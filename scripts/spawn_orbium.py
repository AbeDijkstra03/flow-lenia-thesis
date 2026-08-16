#!/usr/bin/env python3
"""
Standalone Physics Engine Verification Script (JAX).
Spawns the classic Orbium unicaudatus glider in a JAX Flow-Lenia grid and records a broadcast-quality MP4 video and filmstrip.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts, run_flow_lenia_rollout
)
from core.visualization import save_rollout_mp4, extract_trajectory_filmstrip

DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}

def ch2val(c):
    if c in '.b': return 0
    elif c == 'o': return 255
    elif len(c) == 1: return ord(c)-ord('A')+1
    else: return (ord(c[0])-ord('p')) * 24 + (ord(c[1])-ord('A')+25)

def _append_stack(list1, list2, count, is_repeat=False):
    list1.append(list2)
    if count != '':
        repeated = list2 if is_repeat else []
        list1.extend([repeated] * (int(count)-1))

def _recur_get_max_lens(dim, list1, max_lens):
    max_lens[dim] = max(max_lens[dim], len(list1))
    if dim < 1:
        for list2 in list1:
            _recur_get_max_lens(dim+1, list2, max_lens)

def _recur_cubify(dim, list1, max_lens):
    more = max_lens[dim] - len(list1)
    if dim < 1:
        list1.extend([[]] * more)
        for list2 in list1:
            _recur_cubify(dim+1, list2, max_lens)
    else:
        list1.extend([0] * more)

def rle2cells(st):
    stacks = [[] for dim in range(2)]
    last, count = '', ''
    delims = list(DIM_DELIM.values())
    st = st.rstrip('!') + DIM_DELIM[1]
    for ch in st:
        if ch.isdigit(): count += ch
        elif ch in 'pqrstuvwxy@': last = ch
        else:
            if last+ch not in delims:
                _append_stack(stacks[0], ch2val(last+ch)/255.0, count, is_repeat=True)
            else:
                dim = delims.index(last+ch)
                for d in range(dim):
                    _append_stack(stacks[d+1], stacks[d], count, is_repeat=False)
                    stacks[d] = []
            last, count = '', ''
    A = stacks[1]
    max_lens = [0 for dim in range(2)]
    _recur_get_max_lens(0, A, max_lens)
    _recur_cubify(0, A, max_lens)
    return np.asarray(A)

def main():
    print(f"Spawning Orbium unicaudatus in JAX Flow Lenia ({jax.devices()})...")
    grid_size = 256
    mu = 0.15
    sigma = 0.015
    rle = '7.MD6.qL$6.pKqEqFURpApBRAqQ$5.VqTrSsBrOpXpWpTpWpUpCrQ$4.CQrQsTsWsApITNPpGqGvL$3.IpIpWrOsGsBqXpJ4.LsFrL$A.DpKpSpJpDqOqUqSqE5.ExD$qL.pBpTT2.qCrGrVrWqM5.sTpP$.pGpWpD3.qUsMtItQtJ6.tL$.uFqGH3.pXtOuR2vFsK5.sM$.tUqL4.GuNwAwVxBwNpC4.qXpA$2.uH5.vBxGyEyMyHtW4.qIpL$2.wV5.tIyG3yOxQqW2.FqHpJ$2.tUS4.rM2yOyJyOyHtVpPMpFqNV$2.HsR4.pUxAyOxLxDxEuVrMqBqGqKJ$3.sLpE3.pEuNxHwRwGvUuLsHrCqTpR$3.TrMS2.pFsLvDvPvEuPtNsGrGqIP$4.pRqRpNpFpTrNtGtVtStGsMrNqNpF$5.pMqKqLqRrIsCsLsIrTrFqJpHE$6.RpSqJqPqVqWqRqKpRXE$8.OpBpIpJpFTK!'
    
    cells = rle2cells(rle)
    h, w = cells.shape
    
    H, W = grid_size, grid_size
    K = 1
    radii = jnp.array([13.0], dtype=jnp.float32)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    mass = np.zeros((1, H, W), dtype=np.float32)
    y_start = (H - h) // 2
    x_start = (W - w) // 2
    mass[0, y_start:y_start+h, x_start:x_start+w] = cells
    
    mass_jnp = jnp.array(mass, dtype=jnp.float32)
    mu_map = jnp.full((K, H, W), mu, dtype=jnp.float32)
    sigma_map = jnp.full((K, H, W), sigma, dtype=jnp.float32)
    weights_map = jnp.full((K, H, W), 1.0, dtype=jnp.float32)
    resource_map = jnp.ones((H, W), dtype=jnp.float32)
    gid_map = jnp.zeros((H, W), dtype=jnp.int32)
    
    state = FlowLeniaState(
        mass=mass_jnp,
        mu_map=mu_map,
        sigma_map=sigma_map,
        weights_map=weights_map,
        resource_map=resource_map,
        genome_id_map=gid_map
    )
    
    params = FlowLeniaParams(
        mu=jnp.array([mu]),
        sigma=jnp.array([sigma]),
        weights=jnp.array([1.0]),
        dt=0.05
    )
    
    rng_key = random.PRNGKey(42)
    print("Simulating Orbium glider for 300 steps...")
    final_state, sampled_mass, _ = run_flow_lenia_rollout(
        state, kernel_ffts, params, rng_key,
        num_steps=300, sample_interval=2,
        mixing_rule='constant', enable_mutation=False
    )
    
    sampled_mass_np = np.array(sampled_mass)
    os.makedirs("results", exist_ok=True)
    video_path = "results/spawned_orbium.mp4"
    filmstrip_path = "results/spawned_orbium_filmstrip.png"
    
    save_rollout_mp4(sampled_mass_np, video_path, fps=20, dual_panel=True)
    extract_trajectory_filmstrip(sampled_mass_np, filmstrip_path, num_frames=6, dual_panel=True)
    
    print(f"Saved Orbium glider MP4 video to: {video_path}")
    print(f"Saved Orbium glider filmstrip to: {filmstrip_path}")

if __name__ == '__main__':
    main()
