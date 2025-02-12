import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import cv2
import pygame


class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # e.g., (84, 84, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99           # discount factor
        self.epsilon = 0.7          # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.target_update_freq = 1000  # update target network every fixed number of training steps
        self.train_step = 0

        # Build the online and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=self.state_size),
            keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from the online network to the target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience tuple in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """Train the online network using a batch sampled from memory with the Double DQN update."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([exp[0] for exp in minibatch])
        next_states = np.vstack([exp[3] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * target_next[i][next_actions[i]]

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()


class SpaceInvadersEnv:
    def __init__(self, env_name='SpaceInvaders-v4', window_width=800, window_height=600):
        self.env = gym.make(env_name)
        self.window_width = window_width
        self.window_height = window_height

        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Space Invaders AI')
        self.clock = pygame.time.Clock()
        self.fps = 60

        self.MISS_SHOT_PENALTY = -5
        self.SUCCESS_REWARD = 10
        self.LIFE_LOSS_PENALTY = -20
        self.DODGE_REWARD = 2
        self.WIN_SCORE = 1000
        self.WIN_REWARD = 50

        self.last_score = 0
        self.last_lives = 3

    def step(self, action):
        fired_shot = False
        action_mapped = action

        if action == 2:
            fired_shot = True
            action_mapped = 1
        elif action == 0:
            action_mapped = 3
        elif action == 1:
            action_mapped = 2

        result = self.env.step(action_mapped)
        if len(result) == 5:
            next_state, env_reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, env_reward, done, info = result

        reward = env_reward
        current_score = info.get('score', self.last_score)
        current_lives = info.get('lives', self.last_lives)

        if fired_shot:
            reward += 1
            if current_score > self.last_score:
                reward += self.SUCCESS_REWARD
            else:
                reward += self.MISS_SHOT_PENALTY
        elif action in [0, 1] and current_lives == self.last_lives:
            reward += self.DODGE_REWARD

        if current_lives < self.last_lives:
            reward += self.LIFE_LOSS_PENALTY

        if done and current_score >= self.WIN_SCORE:
            reward += self.WIN_REWARD

        self.last_score = current_score
        self.last_lives = current_lives

        if isinstance(next_state, tuple):
            next_state = next_state[0]
        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        self.last_score = 0
        self.last_lives = 3
        return state

    def close(self):
        self.env.close()
        pygame.quit()


def preprocess_frame(frame):
    if isinstance(frame, tuple):
        frame = frame[0]
    frame = np.array(frame)
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame / 255.0


def stack_frames(stacked_frames, state, is_new):
    frame = preprocess_frame(state)
    if is_new or stacked_frames is None:
        stacked_frames = np.stack([frame] * 4, axis=2)
    else:
        stacked_frames = np.append(stacked_frames[:, :, 1:], np.expand_dims(frame, axis=2), axis=2)
    return stacked_frames


def main():
    env_name = 'SpaceInvaders-v4'
    env = SpaceInvadersEnv(env_name)
    state_size = (84, 84, 4)
    action_size = env.env.action_space.n
    agent = DoubleDQNAgent(state_size, action_size)

    batch_size = 32
    episodes = 50

    for e in range(episodes):
        state = env.reset()
        processed_frame = preprocess_frame(state)
        stacked_frames = np.stack([processed_frame] * 4, axis=2)
        state = np.expand_dims(stacked_frames, axis=0)
        total_reward = 0

        for time in range(5000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            processed_next_frame = preprocess_frame(next_state)
            stacked_frames = stack_frames(stacked_frames, processed_next_frame, is_new=False)
            next_state_expanded = np.expand_dims(stacked_frames, axis=0)

            agent.remember(state, action, reward, next_state_expanded, done)
            state = next_state_expanded
            total_reward += reward

            if done:
                agent.update_target_model()
                break

            agent.replay(batch_size)

    env.close()


if __name__ == "__main__":
    main()
