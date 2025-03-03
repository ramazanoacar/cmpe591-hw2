import time
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from homework2 import Hw2Env


class QNetwork(nn.Module):
    """
    A simple MLP with input_dim=6 (EE_x, EE_y, OBJ_x, OBJ_y, GOAL_x, GOAL_y),
    hidden_dim=64, and output_dim=8 discrete actions.
    """

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=8):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """A simple replay buffer with uniform random sampling."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


def huber_loss(input, target, delta=1.0):
    error = target - input
    cond = error.abs() < delta
    loss = torch.where(cond, 0.5 * error.pow(2), delta * (error.abs() - 0.5 * delta))
    return loss.mean()


def update_network(
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    batch_size,
    gamma=0.99,
    device="cpu",
):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    q_values = policy_net(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = huber_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def smooth_data(data, window_size=50):
    """Apply moving-average smoothing to the data list."""
    if len(data) < window_size:
        return data

    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2)
        window = data[start_idx:end_idx]
        smoothed.append(sum(window) / len(window))

    return smoothed


def save_to_csv(episode_rewards, episode_rps, filename="dqn_training_data.csv"):
    """Save training results (rewards, RPS) to CSV."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Reward", "RewardPerStep"])
        for i in range(len(episode_rewards)):
            writer.writerow([i, episode_rewards[i], episode_rps[i]])
    print(f"Data saved to {filename}")


def save_model(model, filename="dqn_model.pt"):
    """Save the trained model weights."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def train(
    num_episodes=3800,
    update_frequency=10,
    target_update_frequency=200,
    epsilon=1.0,
    epsilon_decay=0.997,
    epsilon_min=0.03,
    batch_size=64,
    gamma=0.99,
    buffer_size=100000,
    learning_rate=1e-3,
    device="cpu",
    save_checkpoints=True,
):
    """
    Parameters used in the training loop.
    """
    env = Hw2Env(n_actions=8, render_mode="offscreen")

    # Initialize networks
    policy_net = QNetwork(input_dim=6, hidden_dim=64, output_dim=8).to(device)
    target_net = QNetwork(input_dim=6, hidden_dim=64, output_dim=8).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_size)

    # Tracking
    global_step = 0
    episode_rewards = []
    episode_rps = []

    for episode in range(num_episodes):
        env.reset()
        obs = env.high_level_state()  # shape = (6,)
        done = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0

        while not (done or truncated):
            global_step += 1
            episode_steps += 1

            # Epsilon-greedy selection
            if random.random() < epsilon:
                action = np.random.randint(8)
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    q_vals = policy_net(obs_tensor)
                    action = q_vals.argmax(dim=1).item()

            # Take step in environment
            state, reward, is_terminal, is_truncated = env.step(action)
            next_obs = env.high_level_state()

            # Store transition
            replay_buffer.push(
                obs, action, reward, next_obs, float(is_terminal or is_truncated)
            )

            obs = next_obs
            episode_reward += reward
            done = is_terminal
            truncated = is_truncated

            # Update the policy network
            if global_step % update_frequency == 0:
                update_network(
                    policy_net,
                    target_net,
                    replay_buffer,
                    optimizer,
                    batch_size,
                    gamma,
                    device,
                )

            # Update the target network
            if global_step % target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

        episode_rewards.append(episode_reward)
        rps = episode_reward / episode_steps if episode_steps > 0 else 0
        episode_rps.append(rps)

        print(
            f"Episode {episode} finished: "
            f"Reward={episode_reward:.3f}, RPS={rps:.3f}, Epsilon={epsilon:.3f}"
        )

        if save_checkpoints and (episode % 100 == 0) and (episode > 0):
            csv_file = f"dqn_training_checkpoint_ep{episode}.csv"
            model_file = f"dqn_model_checkpoint_ep{episode}.pt"
            save_to_csv(episode_rewards, episode_rps, csv_file)
            save_model(policy_net, model_file)

    # Save final results
    save_to_csv(episode_rewards, episode_rps, "dqn_training_data.csv")
    save_model(policy_net, "dqn_model.pt")

    return policy_net, episode_rewards, episode_rps


def test(
    model=None,
    model_path="dqn_model.pt",
    num_episodes=10,
    device="cpu",
    render_mode="offscreen",
):
    """
    If 'model' is provided, we test that model.
    If 'model' is None, we instantiate a QNetwork, load state_dict from `model_path`, and test it.

    Args:
        model:          A QNetwork instance or None (default). If None, load from model_path.
        model_path:     Path to the .pt file with the model weights (only used if model=None).
        num_episodes:   How many test episodes to run.
        device:         "cpu" or "cuda".
        render_mode:    "offscreen" or "gui" for the Hw2Env environment.
    """
    if model is None:
        print(f"[Test] No model supplied. Loading from {model_path} ...")
        model = QNetwork(input_dim=6, hidden_dim=64, output_dim=8).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Test] Successfully loaded model from {model_path}")

    model.eval()

    env = Hw2Env(n_actions=8, render_mode=render_mode)

    all_rewards = []
    global_step = 0
    for episode in range(num_episodes):
        env.reset()
        obs = env.high_level_state()
        done = False
        truncated = False
        step = 0
        episode_reward = 0.0
        while not (done or truncated):
            step += 1
            global_step += 1
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q_vals = model(obs_tensor)
                action = q_vals.argmax(dim=1).item()

            _, reward, is_terminal, is_truncated = env.step(action)
            obs = env.high_level_state()

            episode_reward += reward
            done = is_terminal
            truncated = is_truncated

        all_rewards.append(episode_reward)
        print(
            f"[Test] Episode {episode} - Reward = {episode_reward:.3f} - Reward Per Step = {episode_reward / step:.3f}"
        )

    avg_reward = np.mean(all_rewards)
    print(
        f"[Test] Average reward over {num_episodes} episodes: {avg_reward:.3f} - Average Reward Per Step = {np.sum(all_rewards) / global_step:.3f}"
    )
    return avg_reward


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    policy_net, rewards, rps = train(
        num_episodes=3800,
        device=device,
        save_checkpoints=False,
    )

    window_size = min(50, len(rewards))
    smoothed_rewards = smooth_data(rewards, window_size)
    smoothed_rps = smooth_data(rps, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.3)
    plt.plot(rps, label="Reward Per Step", alpha=0.3)
    plt.plot(smoothed_rewards, label="Smoothed Episode Reward", linewidth=2)
    plt.plot(smoothed_rps, label="Smoothed Reward Per Step", linewidth=2)
    plt.title("DQN Training on Hw2Env")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Reward Per Step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("dqn_training_plot.png")
    print("Saved training plot to dqn_training_plot.png")

    test(policy_net, num_episodes=5, device=device)


if __name__ == "__main__":
    main()
