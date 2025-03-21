import os
from collections import defaultdict
from functools import partial

import flax.linen as nn
import gymnasium as gym
import gymnax
from gymnax.wrappers import LogWrapper
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
# Tensorboard: Prevent tf from allocating full GPU memory
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from tdmpc2_jax import TDMPC2, WorldModel
from tdmpc2_jax.common.activations import mish, simnorm
from tdmpc2_jax.data import SequentialReplayBuffer
from tdmpc2_jax.envs.dmcontrol import make_dmc_env
from tdmpc2_jax.networks import NormedLinear

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']

  seed = cfg.seed
  max_steps = cfg.max_steps
  log_interval_steps = cfg.log_interval_steps
  save_interval_steps = cfg.save_interval_steps

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  env, env_params = gymnax.make(env_config.env_id)
  env = LogWrapper(env)

  np.random.seed(seed)
  rng = jax.random.PRNGKey(seed)

  rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
  env_reset = jax.vmap(env.reset, in_axes=(0, None))
  env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
  env_action_space_sample = jax.vmap(env.action_space(env_params).sample)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, model_key, encoder_key = jax.random.split(rng, 3)
  encoder_module = nn.Sequential([
      NormedLinear(encoder_config.encoder_dim, activation=mish, dtype=dtype)
      for _ in range(encoder_config.num_encoder_layers-1)] + [
      NormedLinear(
          model_config.latent_dim,
          activation=partial(simnorm, simplex_dim=model_config.simnorm_dim),
          dtype=dtype)
  ])

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(encoder_module.tabulate(jax.random.key(0),
          env.observation_space.sample(), compute_flops=True))

  ##############################
  # Replay buffer setup
  ##############################
  rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
  dummy_obs, dummy_state = env_reset(jnp.array(env_rngs), env_params)
  rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
  dummy_action = env_action_space_sample(jnp.array(env_rngs))
  rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
  dummy_next_obs, dummy_next_state, dummy_reward, dummy_term, _ = \
    env_step(jnp.array(env_rngs), dummy_state, dummy_action, env_params)
  dummy_trunc = jnp.zeros_like(dummy_term)
  dummy_replay_data = dict(
    observation=dummy_obs,
    action=dummy_action,
    reward=dummy_reward,
    next_observation=dummy_next_obs,
    terminated=dummy_term,
    truncated=dummy_trunc)
  replay_buffer = SequentialReplayBuffer(
    capacity=max_steps // env_config.num_envs,
    num_envs=env_config.num_envs,
    seed=seed,
    dummy_input=dummy_replay_data
  )
  
  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, dummy_obs)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config.max_grad_norm),
          optax.adam(encoder_config.learning_rate),
      ))

  model = WorldModel.create(
      action_dim=int(np.prod(dummy_action.shape) / env_config.num_envs),
      encoder=encoder,
      **model_config,
      key=model_key)
  if model.action_dim >= 20:
    tdmpc_config.mppi_iterations += 2

  agent = TDMPC2.create(world_model=model, **tdmpc_config)
  global_step = 0

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps'])
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path, options=options, item_names=(
          'agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(mngr.latest_step(),
                              args=ocp.args.Composite(
          agent=ocp.args.StandardRestore(agent),
          global_step=ocp.args.JsonRestore(),
          buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
      )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    prev_plan = (
        jnp.zeros((env_config.num_envs, agent.horizon, agent.model.action_dim)),
        jnp.full((env_config.num_envs, agent.horizon,
                  agent.model.action_dim), agent.max_plan_std)
    )
    rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
    observation, state = env_reset(jnp.array(env_rngs), env_params)
  
    T = 500
    seed_steps = int(max(5*T, 1000) * env_config.num_envs *
                     env_config.utd_ratio)
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(env_config.num_envs, dtype=bool)
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if global_step <= seed_steps:
        rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
        action = env_action_space_sample(jnp.array(env_rngs))
      else:
        rng, action_key = jax.random.split(rng)
        prev_plan = (prev_plan[0],
                     jnp.full_like(prev_plan[1], agent.max_plan_std))
        action, prev_plan = agent.act(
            observation, prev_plan=prev_plan, train=True, key=action_key)

      rng, *env_rngs = jax.random.split(rng, env_config.num_envs + 1)
      next_observation, state, reward, terminated, info = env_step(jnp.array(env_rngs), state, action, env_params)
      truncated = jnp.zeros_like(terminated)

      if np.any(~done):
        replay_buffer.insert(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated
            ),
            env_mask=~done
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        prev_plan = (
            prev_plan[0].at[done].set(0),
            prev_plan[1].at[done].set(agent.max_plan_std)
        )
        if "returned_episode_returns" in info:
          rer = info["returned_episode_returns"]
          rel = info["returned_episode_lengths"]

          for ienv in range(rer.shape[0]):
            re = float(rer[ienv])
            le = float(rel[ienv])
            print(f"Episode {ep_count[ienv]}: {re}, {le}")
            writer.scalar(f'episode/return', re, global_step + ienv)
            writer.scalar(f'episode/length', le, global_step + ienv)
            ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size, agent.horizon)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate])

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      pbar.update(env_config.num_envs)
    pbar.close()


if __name__ == '__main__':
  train()
