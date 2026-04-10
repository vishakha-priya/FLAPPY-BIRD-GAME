import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import argparse
import itertools

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]

        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):

        if render:
            env = gym.make("FlappyBird-v0", render_mode="human")
        else:
            env = gym.make("FlappyBird-v0")

        num_states = env.observation_space.shape[0] #input dim
        num_actions = env.action_space.n #output dim

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
             
            # target network 
            target_dqn = DQN(num_states, num_actions).to(device)
            #copy the wt and bias from policy=>target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            steps = 0
            best_reward = float("-inf")
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            epsilon=0

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            episode_rewards = 0
            terminated = False

            while not terminated and episode_rewards < self.reward_threshold:

                # ε-greedy policy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() #explore
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax().item() #exploit

                # Take step
                new_state, reward, terminated, _, _ = env.step(action)

                next_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
                action_tensor = torch.tensor(action, dtype=torch.long, device=device)

                episode_rewards += reward

                if is_training:
                    memory.append((state, action_tensor, next_state, reward_tensor, terminated))
                    steps += 1

                state = next_state
            if is_training:
                print(f"Episode={episode+1}, Reward={episode_rewards}, Epsilon={epsilon}")

            if is_training:
                # Epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save best model
                if episode_rewards > best_reward:
                    best_reward = episode_rewards
                    with open(self.LOG_FILE, "a") as f:
                        f.write(f"Best reward={episode_rewards} at episode={episode+1}\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                # Training step
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Sync networks
                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

       # env.close() to manually stop (ctrl+c)

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        #here we can train it one by one but that will mmake the training more slower

        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        # Target Q-values (y true)
        with torch.no_grad():
            max_next_q = target_dqn(next_states).max(dim=1)[0]
            target_q = rewards + (1 - terminations) * self.gamma * max_next_q

        # Current Q-values (y pred)
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    #parse command line input
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("hyperparameters")
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    agent = Agent(param_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)