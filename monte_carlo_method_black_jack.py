import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

# Blackjack-v1 spaces:
#   Observation: Tuple(Discrete(32), Discrete(11), Discrete(2))
#   Action:      Discrete(2)
STATE_SHAPE = (32, 11, 2)
N_ACTIONS = 2
EPS_MIN = 0.05


def _sidx(state):
    """Turn a Blackjack state tuple into an ndarray index tuple."""
    return (int(state[0]), int(state[1]), int(state[2]))


def epsilon_greedy_policy(state, Q, N_state):

    if state[0] >= 20:
        return 0

    si = _sidx(state)
    # GLIE decay with a floor so rare states keep exploring a bit.
    eps = max(EPS_MIN, 1.0 / (1 + N_state[si]))

    if random.uniform(0, 1) < eps:
        return env.action_space.sample()
    q_row = Q[si]
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(random.choice(best))


def generate_episode(num_timesteps, Q, N_state):
    episode = []
    state, _ = env.reset()

    for t in range(num_timesteps):

        # select the action according to the epsilon-greedy policy
        action = epsilon_greedy_policy(state, Q, N_state)

        # perform the selected action and store the next state information
        next_state, reward, done, info, _ = env.step(action)

        # store the state, action, reward in the episode list
        episode.append((state, action, reward))

        # if the next state is a final state then break the loop else update the next state to the current
        # state
        if done:
            break

        state = next_state

    return episode


def generate_policy(Q, N):
    """Greedy policy as an ndarray of shape STATE_SHAPE; -1 for unvisited states."""
    visited = N.sum(axis=-1) > 0
    # Mask unvisited actions with -inf so they never win argmax.
    masked_Q = np.where(N > 0, Q, -np.inf)
    max_vals = masked_Q.max(axis=-1, keepdims=True)
    ties = (masked_Q == max_vals) & (N > 0)

    policy = np.full(STATE_SHAPE, -1, dtype=np.int64)
    # Random tie-break: pick uniformly among argmax actions.
    rand = np.random.rand(*STATE_SHAPE, N_ACTIONS) * ties
    policy_visited = rand.argmax(axis=-1)
    policy[visited] = policy_visited[visited]
    return policy


def test_policy(policy, env):

    num_episodes = 50000
    num_timesteps = 10000
    total_reward = 0

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(num_timesteps):
            # action = env.action_space.sample()
            action = int(policy[_sidx(state)])
            if action < 0:
                # Unvisited during training: fall back to a safe default.
                action = 0
            next_state, reward, done, info, _ = env.step(action)
            episode_reward += reward

            if done:
                break

            state = next_state

        total_reward += episode_reward

    return total_reward / num_episodes


def plot_visit_distribution(N):
    """Visualize the distribution of visit counts in the N array."""
    counts = N[N > 0].ravel()
    if counts.size == 0:
        print("N is empty, nothing to plot.")
        return

    unique_pairs = counts.size
    total_visits = int(counts.sum())
    print(f"Unique (state, action) pairs: {unique_pairs}")
    print(f"Total visits: {total_visits}")
    print(f"Min / Median / Mean / Max visits: "
          f"{counts.min()} / {np.median(counts):.1f} / "
          f"{counts.mean():.1f} / {counts.max()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of visit counts (log scale on y)
    axes[0, 0].hist(counts, bins=50, color="steelblue", edgecolor="black")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Histogram of visit counts N[s, a]")
    axes[0, 0].set_xlabel("Visit count")
    axes[0, 0].set_ylabel("Number of (state, action) pairs (log)")

    # 2. Histogram of log10(visit count) - reveals the tail
    axes[0, 1].hist(np.log10(counts), bins=50, color="darkorange", edgecolor="black")
    axes[0, 1].set_title("Histogram of log10(visit count)")
    axes[0, 1].set_xlabel("log10(N)")
    axes[0, 1].set_ylabel("Number of (state, action) pairs")

    # 3. Sorted visit counts (rank plot) on log-log scale
    sorted_counts = np.sort(counts)[::-1]
    axes[1, 0].plot(np.arange(1, sorted_counts.size + 1), sorted_counts, color="seagreen")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Rank plot of visit counts (log-log)")
    axes[1, 0].set_xlabel("Rank")
    axes[1, 0].set_ylabel("Visit count")

    # 4. Per-action comparison
    for action in range(N.shape[-1]):
        ac = N[..., action].ravel()
        ac = ac[ac > 0]
        if ac.size == 0:
            continue
        axes[1, 1].hist(
            ac,
            bins=50,
            alpha=0.5,
            label=f"action={action} (n={ac.size})",
            edgecolor="black",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Visit count distribution per action")
    axes[1, 1].set_xlabel("Visit count")
    axes[1, 1].set_ylabel("Number of states (log)")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig("N_distribution.png", dpi=120)
    print("Saved plot to N_distribution.png")
    plt.show()


def plot_visit_by_state_dims(N):
    """Show how the visit count depends on each state component.

    Blackjack state = (player_sum, dealer_showing, usable_ace); N has shape
    (player_sum, dealer, usable_ace, action).
    """
    # Sum over the action axis first, then reduce state dims.
    N_state = N.sum(axis=-1)  # (player_sum, dealer, usable_ace)

    by_player_sum = N_state.sum(axis=(1, 2))  # (32,)
    by_dealer = N_state.sum(axis=(0, 2))      # (11,)
    by_ace = N_state.sum(axis=(0, 1))         # (2,)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Marginals as bar plots
    xs = np.arange(by_player_sum.size)
    axes[0, 0].bar(xs, by_player_sum, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Visits vs player sum (state[0])")
    axes[0, 0].set_xlabel("player sum")
    axes[0, 0].set_ylabel("total visits")

    xs = np.arange(by_dealer.size)
    axes[0, 1].bar(xs, by_dealer, color="darkorange", edgecolor="black")
    axes[0, 1].set_title("Visits vs dealer showing (state[1])")
    axes[0, 1].set_xlabel("dealer card")
    axes[0, 1].set_ylabel("total visits")

    axes[0, 2].bar(
        [str(x) for x in range(by_ace.size)],
        by_ace,
        color=["seagreen", "crimson"][: by_ace.size],
        edgecolor="black",
    )
    axes[0, 2].set_title("Visits vs usable ace (state[2])")
    axes[0, 2].set_xlabel("usable ace")
    axes[0, 2].set_ylabel("total visits")

    # Heatmaps of (player_sum, dealer) for each usable_ace value
    for col, ace in enumerate([0, 1]):
        grid = N_state[:, :, ace]  # (player_sum, dealer)
        if grid.sum() == 0:
            axes[1, col].set_title(f"No data for usable_ace={ace}")
            axes[1, col].axis("off")
            continue
        im = axes[1, col].imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        axes[1, col].set_title(f"Visits heatmap, usable_ace={ace}")
        axes[1, col].set_xlabel("dealer card (state[1])")
        axes[1, col].set_ylabel("player sum (state[0])")
        fig.colorbar(im, ax=axes[1, col], label="visits")

    # Log-scaled version of the ace=0 heatmap to reveal low-visit cells
    grid = N_state[:, :, 0]
    if grid.sum() > 0:
        im = axes[1, 2].imshow(
            np.log10(grid + 1),
            aspect="auto",
            origin="lower",
            cmap="magma",
        )
        axes[1, 2].set_title("log10(visits+1), usable_ace=0")
        axes[1, 2].set_xlabel("dealer card")
        axes[1, 2].set_ylabel("player sum")
        fig.colorbar(im, ax=axes[1, 2], label="log10(visits+1)")
    else:
        axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig("N_by_state_dims.png", dpi=120)
    print("Saved plot to N_by_state_dims.png")
    plt.show()


if __name__ == '__main__':

    # Action Space Discrete(2)
    # Observation Space Tuple(Discrete(32), Discrete(11), Discrete(2))

    env = gym.make('Blackjack-v1', natural=True, sab=False)

    Q = np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.float64)
    N = np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.int64)
    N_state = np.zeros(STATE_SHAPE, dtype=np.int64)

    num_iterations = 1000000
    for i in tqdm.tqdm(range(num_iterations)):
        episode = generate_episode(100, Q, N_state)

        # Bump per-state visit counter for GLIE epsilon decay.
        for (state, _, _) in episode:
            N_state[_sidx(state)] += 1

        # First-visit indices: keep the earliest t at which each (s, a) appears.
        first_visit = {}
        for t, (state, action, _) in enumerate(episode):
            key = (_sidx(state), int(action))
            if key not in first_visit:
                first_visit[key] = t

        # Compute returns G_t in a single backward pass (undiscounted).
        G = 0.0
        returns = [0.0] * len(episode)
        for t in range(len(episode) - 1, -1, -1):
            G = episode[t][2] + G
            returns[t] = G

        # Incremental-mean update, only at the first-visit step of each (s, a).
        for (si, action), t in first_visit.items():
            idx = si + (action,)
            N[idx] += 1
            Q[idx] += (returns[t] - Q[idx]) / N[idx]

    policy = generate_policy(Q, N)
    print(test_policy(policy, env))

    plot_visit_distribution(N)
    plot_visit_by_state_dims(N)
