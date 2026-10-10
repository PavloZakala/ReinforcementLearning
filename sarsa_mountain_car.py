import argparse
import os
import random

from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from gymnasium.wrappers import RecordVideo
from matplotlib.colors import ListedColormap


# Mountain Car has a continuous observation space (position, velocity),
# so we discretize it into a fixed-size grid to reuse a tabular SARSA update.
#
# Trick: bin count is a critical hyperparameter. Too few bins (<10) throw away
# the velocity resolution needed to swing the car; too many bins (>50) explode
# the state space and slow learning. 20x20 is a good balance for MountainCar.

# Observation Space
# Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
NUM_BINS = (20, 20)

# 0: Accelerate to the left
# 1: Don’t accelerate
# 2: Accelerate to the right



def build_bins(env, num_bins):
    low = env.observation_space.low
    high = env.observation_space.high
    # np.digitize expects interior edges only; slicing [1:-1] avoids off-by-one
    # bin indices at the observation-space boundaries.
    return [np.linspace(low[i], high[i], num_bins[i] + 1)[1:-1] for i in range(len(num_bins))]


def discretize(obs, bins):
    return tuple(int(np.digitize(obs[i], bins[i])) for i in range(len(bins)))


def reset_to(env, position, velocity):
    """Reset the env then overwrite its internal state to (position, velocity).

    Gymnasium's MountainCar samples position uniformly in [-0.6, -0.4] and
    velocity=0, so this bypass is the only way to seed rollouts from arbitrary
    points in the state space (e.g. up on the hills or moving fast).
    """
    env.reset()
    env.unwrapped.state = np.array([position, velocity], dtype=np.float64)
    return np.array([position, velocity], dtype=np.float64)


def uniform_start_states(env, num_bins=NUM_BINS):
    """Return the (position, velocity) center of every discretization bin."""
    low = env.observation_space.low
    high = env.observation_space.high
    pos_edges = np.linspace(low[0], high[0], num_bins[0] + 1)
    vel_edges = np.linspace(low[1], high[1], num_bins[1] + 1)
    pos_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])
    vel_centers = 0.5 * (vel_edges[:-1] + vel_edges[1:])
    return [(float(p), float(v)) for p in pos_centers for v in vel_centers]


# Reward shaping potential based on the car's mechanical energy.
#
# MountainCar's hill profile is height(x) = sin(3x), so potential energy is
# proportional to sin(3x) (mass & g absorbed into the scale). Kinetic energy
# is 0.5 * v^2. Summing them yields the total mechanical energy the car has
# accumulated -- a natural progress signal that grows as the car swings
# higher on either side of the valley.
#
# Trick (Ng, Harada & Russell 1999, potential-based reward shaping):
#     r_shaped = r_env + (gamma * phi(s') - phi(s))
# This form provably preserves the optimal policy of the original MDP while
# giving the agent a dense, non-sparse learning signal so it doesn't have to
# hit the flag by pure chance before learning anything.
def potential(obs):
    position, velocity = obs
    pe = np.sin(3.0 * position)
    ke = 0.5 * velocity * velocity
    return pe + ke


def epsilon_greedy(env, Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(range(env.action_space.n), key=lambda a: Q[(state, a)])


def generate_policy(env, Q, bins):
    policy = defaultdict(int)
    for i in range(NUM_BINS[0]):
        for j in range(NUM_BINS[1]):
            state = (i, j)
            policy[state] = max(range(env.action_space.n), key=lambda a: Q[(state, a)])
    return policy


# On-policy SARSA with state discretization
def sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon,
          epsilon_min=0.05, epsilon_decay=0.995,
          use_shaping=False, shaping_scale=1.0):
    bins = build_bins(env, NUM_BINS)
    # Trick: defaultdict(float) initializes unseen Q-values to 0. Since every
    # step reward is -1, any unseen (s,a) looks *better* than what we've tried,
    # which acts as a mild "optimism in the face of uncertainty" bonus and
    # encourages the agent to explore unvisited states.
    Q = defaultdict(float)

    episode_rewards = []

    for i in tqdm.tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state = discretize(obs, bins)
        action = epsilon_greedy(env, Q, state, epsilon)
        phi = potential(obs) if use_shaping else 0.0

        total_reward = 0.0
        for t in range(num_timesteps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs, bins)
            done = terminated or truncated

            # Potential-based shaping: F = gamma * phi(s') - phi(s).
            # Only the shaped reward is used for the Q update; total_reward
            # still tracks the true environment return for logging.
            if use_shaping:
                next_phi = potential(next_obs)
                shaping = shaping_scale * (gamma * next_phi - phi)
                shaped_reward = reward + shaping
            else:
                next_phi = 0.0
                shaped_reward = reward

            # Trick: distinguish `terminated` (goal reached) from `truncated`
            # (200-step time limit). On a real terminal, drop the bootstrap so
            # the goal reward isn't diluted by a spurious gamma*Q(next). On
            # truncation we still bootstrap because the episode was cut short,
            # not actually finished -- otherwise the agent learns that time-out
            # is "the end of the world" and value backups get poisoned.
            if done and terminated:
                target = shaped_reward
            else:
                next_action = epsilon_greedy(env, Q, next_state, epsilon)
                # Q_i = \sum(r + gamma * Q_i+1)
                target = shaped_reward + gamma * Q[(next_state, next_action)]
            
            Q[(state, action)] += alpha * (target - Q[(state, action)])

            total_reward += reward
            state = next_state
            phi = next_phi
            if done:
                break
            action = next_action

        episode_rewards.append(total_reward)
        # Trick: multiplicative epsilon decay with a floor. Start fully random
        # (epsilon=1.0) so early episodes hit the flag by chance and inject
        # non-zero signal into Q; decay geometrically toward a small floor so
        # exploration never fully stops and the agent can escape bad grooves.
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if (i + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            tqdm.tqdm.write(f"episode {i + 1} | avg_reward(last 500)={avg:.2f} | epsilon={epsilon:.3f}")

    return Q, bins


def test(env, Q, bins, render=False):
    obs, _ = env.reset()
    state = discretize(obs, bins)
    if render:
        env.render()

    total_reward = 0.0
    done = False
    while not done:
        action = max(range(env.action_space.n), key=lambda a: Q[(state, a)])
        obs, reward, terminated, truncated, _ = env.step(action)
        state = discretize(obs, bins)

        if render:
            env.render()

        total_reward += reward
        done = terminated or truncated

    return total_reward


def show_agent(env, Q, bins):
    obs, _ = env.reset()
    state = discretize(obs, bins)
    env.render()

    done = False
    while not done:
        action = max(range(env.action_space.n), key=lambda a: Q[(state, a)])
        obs, _, terminated, truncated, _ = env.step(action)
        state = discretize(obs, bins)
        env.render()
        done = terminated or truncated

    env.close()


def record_video(Q, bins, video_folder='videos', num_episodes=1, name_prefix='sarsa-mountain-car'):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda ep: ep < num_episodes,
    )

    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins)
        done = False
        while not done:
            action = max(range(env.action_space.n), key=lambda a: Q[(state, a)])
            obs, _, terminated, truncated, _ = env.step(action)
            state = discretize(obs, bins)
            done = terminated or truncated

    env.close()
    print(f"saved {num_episodes} episode video(s) to {os.path.abspath(video_folder)}")


def plot_policy(env, Q, save_path='policy.png'):
    n_actions = env.action_space.n
    pos_bins, vel_bins = NUM_BINS

    policy_grid = np.zeros((vel_bins, pos_bins), dtype=int)
    value_grid = np.zeros((vel_bins, pos_bins), dtype=float)
    for i in range(pos_bins):
        for j in range(vel_bins):
            q_vals = [Q[((i, j), a)] for a in range(n_actions)]
            policy_grid[j, i] = int(np.argmax(q_vals))
            value_grid[j, i] = float(np.max(q_vals))

    low = env.observation_space.low
    high = env.observation_space.high
    extent = [low[0], high[0], low[1], high[1]]

    action_labels = ['left', 'none', 'right']
    cmap = ListedColormap(['#e74c3c', '#bdc3c7', '#2ecc71'][:n_actions])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(policy_grid, origin='lower', extent=extent, aspect='auto',
                     cmap=cmap, vmin=-0.5, vmax=n_actions - 0.5)
    ax1.set_title('Greedy policy')
    ax1.set_xlabel('position')
    ax1.set_ylabel('velocity')
    cbar1 = fig.colorbar(im1, ax=ax1, ticks=range(n_actions))
    cbar1.ax.set_yticklabels(action_labels[:n_actions])

    im2 = ax2.imshow(value_grid, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax2.set_title('State value  V(s) = max_a Q(s, a)')
    ax2.set_xlabel('position')
    ax2.set_ylabel('velocity')
    fig.colorbar(im2, ax=ax2)

    fig.suptitle('SARSA on MountainCar-v0')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved policy visualization to {os.path.abspath(save_path)}")


def analyze_states(env, Q, bins, num_episodes=500, start_states=None):
    """Run greedy rollouts and estimate P(success | state visited) and
    the mean number of remaining steps to finish (successful episodes only).

    If ``start_states`` is provided, each entry is a (position, velocity)
    tuple used as an explicit reset via reset_to. The total number of rollouts
    is then len(start_states) * num_episodes. Otherwise the env's default
    reset distribution is used for ``num_episodes`` rollouts.
    """
    visits = defaultdict(int)
    successes = defaultdict(int)
    steps_sum = defaultdict(float)
    steps_n = defaultdict(int)
    total_success = 0
    total_rollouts = 0

    if start_states is None:
        rollout_starts = [None] * num_episodes
    else:
        rollout_starts = [(p, v) for (p, v) in start_states for _ in range(num_episodes)]

    for start in rollout_starts:
        if start is None:
            obs, _ = env.reset()
        else:
            obs = reset_to(env, start[0], start[1])
        state = discretize(obs, bins)
        # Record the first step index at which each state is visited so we
        # can compute steps-to-finish once the episode length is known.
        first_visit = {state: 0}

        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            action = max(range(env.action_space.n), key=lambda a: Q[(state, a)])
            obs, _, terminated, truncated, _ = env.step(action)
            state = discretize(obs, bins)
            step += 1
            first_visit.setdefault(state, step)

        success = bool(terminated)
        total_success += int(success)
        total_rollouts += 1
        episode_len = step
        for s, first in first_visit.items():
            visits[s] += 1
            if success:
                successes[s] += 1
                steps_sum[s] += episode_len - first
                steps_n[s] += 1

    stats = {}
    for s in visits:
        mean_steps = steps_sum[s] / steps_n[s] if steps_n[s] > 0 else float('nan')
        stats[s] = (visits[s], successes[s], successes[s] / visits[s], mean_steps)

    print(f"state analysis: {len(stats)} unique states visited over {total_rollouts} rollouts | "
          f"episode success rate = {total_success / total_rollouts:.2%}")
    return stats


def plot_state_success(env, stats, save_path='state_success.png'):
    pos_bins, vel_bins = NUM_BINS

    # NaN = never visited; imshow will render it as the cmap 'bad' color.
    prob_grid = np.full((vel_bins, pos_bins), np.nan, dtype=float)
    visit_grid = np.zeros((vel_bins, pos_bins), dtype=int)
    steps_grid = np.full((vel_bins, pos_bins), np.nan, dtype=float)
    for (i, j), (v, _s, p, mean_steps) in stats.items():
        if 0 <= i < pos_bins and 0 <= j < vel_bins:
            prob_grid[j, i] = p
            visit_grid[j, i] = v
            steps_grid[j, i] = mean_steps

    low = env.observation_space.low
    high = env.observation_space.high
    extent = [low[0], high[0], low[1], high[1]]

    cmap_prob = plt.cm.RdYlGn.copy()
    cmap_prob.set_bad('#222222')
    cmap_steps = plt.cm.viridis_r.copy()
    cmap_steps.set_bad('#222222')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    im1 = ax1.imshow(prob_grid, origin='lower', extent=extent, aspect='auto',
                     cmap=cmap_prob, vmin=0.0, vmax=1.0)
    ax1.set_title('P(success | state visited)')
    ax1.set_xlabel('position')
    ax1.set_ylabel('velocity')
    fig.colorbar(im1, ax=ax1, label='success probability')

    # Log scale on visit counts because a handful of states dominate.
    im2 = ax2.imshow(np.log1p(visit_grid), origin='lower', extent=extent, aspect='auto', cmap='magma')
    ax2.set_title('log(1 + visit count)')
    ax2.set_xlabel('position')
    ax2.set_ylabel('velocity')
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(steps_grid, origin='lower', extent=extent, aspect='auto', cmap=cmap_steps)
    ax3.set_title('Mean steps to finish  (successful episodes)')
    ax3.set_xlabel('position')
    ax3.set_ylabel('velocity')
    fig.colorbar(im3, ax=ax3, label='steps remaining')

    fig.suptitle('State analysis of greedy policy')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved state-analysis visualization to {os.path.abspath(save_path)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Tabular SARSA on MountainCar-v0.')

    parser.add_argument('--alpha', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.05, help='minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='multiplicative epsilon decay per episode')

    parser.add_argument('--num-episodes', type=int, default=10000, help='number of training episodes')
    parser.add_argument('--num-timesteps', type=int, default=200, help='max steps per training episode')
    parser.add_argument('--seed', type=int, default=None, help='seed for python/numpy RNGs')

    # Reward shaping options
    parser.add_argument('--use-shaping', action='store_true',
                        help='enable potential-based reward shaping using mechanical energy')
    parser.add_argument('--shaping-scale', type=float, default=1.0,
                        help='multiplier on the shaping term (only used with --use-shaping)')

    # Evaluation / artifacts
    parser.add_argument('--num-eval', type=int, default=100, help='greedy evaluation episodes')
    parser.add_argument('--policy-plot', type=str, default='policy.png', help='where to save policy visualization')
    parser.add_argument('--state-plot', type=str, default='state_success.png',
                        help='where to save the per-state success-probability heatmap')
    parser.add_argument('--analysis-episodes', type=int, default=500,
                        help='greedy episodes per starting state used for the analysis')
    parser.add_argument('--uniform-start', action='store_true',
                        help='iterate uniformly over every discretized starting state instead of using the env default reset')
    parser.add_argument('--start-pos', type=float, default=None,
                        help='manual initial position for analysis rollouts (requires --start-vel; overrides --uniform-start)')
    parser.add_argument('--start-vel', type=float, default=None,
                        help='manual initial velocity for analysis rollouts (requires --start-pos)')
    parser.add_argument('--video-folder', type=str, default='videos', help='where to save recorded videos')
    parser.add_argument('--video-episodes', type=int, default=3, help='number of greedy episodes to record (0 to skip)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    env = gym.make('MountainCar-v0')

    # Hyperparameter tricks:
    # - alpha=0.1 is small enough to average over noisy per-episode returns
    #   but large enough to actually move Q within a reasonable budget.
    # - gamma=0.99 keeps long-horizon credit assignment alive -- the goal is
    #   ~100+ steps away, so a smaller gamma would prevent reward propagation
    #   back to the start state.
    # - epsilon=1.0 (pure random start) is essential: the -1-per-step reward
    #   gives *no* gradient until the flag is hit at least once by chance.
    # - --use-shaping toggles potential-based reward shaping; with shaping on,
    #   convergence is typically 5-10x faster.
    print(f"config: alpha={args.alpha} gamma={args.gamma} epsilon={args.epsilon} "
          f"episodes={args.num_episodes} shaping={args.use_shaping} scale={args.shaping_scale}")

    Q, bins = sarsa(
        env,
        num_episodes=args.num_episodes,
        num_timesteps=args.num_timesteps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        use_shaping=args.use_shaping,
        shaping_scale=args.shaping_scale,
    )

    policy = generate_policy(env, Q, bins)
    print(f"learned policy over {len(policy)} discretized states")

    env.close()

    plot_policy(gym.make('MountainCar-v0'), Q, save_path=args.policy_plot)

    # Evaluate the greedy policy without exploration
    eval_env = gym.make('MountainCar-v0')
    total = 0.0
    for _ in range(args.num_eval):
        total += test(eval_env, Q, bins, render=False)
    print(f"average reward over {args.num_eval} eval episodes: {total / args.num_eval:.2f}")
    eval_env.close()

    # Per-state success-probability analysis under the greedy policy
    analysis_env = gym.make('MountainCar-v0')

    if (args.start_pos is None) != (args.start_vel is None):
        raise SystemExit("--start-pos and --start-vel must be given together")

    if args.start_pos is not None:
        start_states = [(args.start_pos, args.start_vel)]
        print(f"analysis: manual start state ({args.start_pos}, {args.start_vel}) x {args.analysis_episodes} episodes")
    elif args.uniform_start:
        start_states = uniform_start_states(analysis_env)
        print(f"analysis: uniform coverage {len(start_states)} start states x {args.analysis_episodes} episodes each")
    else:
        start_states = None
        print(f"analysis: env-default reset x {args.analysis_episodes} episodes")

    stats = analyze_states(analysis_env, Q, bins,
                           num_episodes=args.analysis_episodes,
                           start_states=start_states)
    analysis_env.close()

    top = sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True)[:10]
    print("top 10 most-visited states  (state -> visits, successes, P(success), mean_steps_to_finish):")
    for s, (v, sc, p, ms) in top:
        ms_str = f"{ms:.1f}" if ms == ms else "n/a"  # NaN check
        print(f"  {s}: visits={v} successes={sc} P(success)={p:.2%} mean_steps={ms_str}")

    plot_state_success(gym.make('MountainCar-v0'), stats, save_path=args.state_plot)

    if args.video_episodes > 0:
        record_video(Q, bins, video_folder=args.video_folder, num_episodes=args.video_episodes)

    # Uncomment to visualize a rollout of the learned agent in a window
    # render_env = gym.make('MountainCar-v0', render_mode='human')
    # show_agent(render_env, Q, bins)
