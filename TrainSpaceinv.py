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
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.7  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.target_update = 1000   # target network update frequency
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=self.state_size))
        model.add(keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
        model.add(keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def preprocess_frame(frame):
    if isinstance(frame, tuple):
        frame = frame[0]  # Extract the first element if frame is a tuple
    frame = np.array(frame)  # Ensure frame is a NumPy array
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame / 255.0
    return frame

def stack_frames(stacked_frames, state, is_new):
    frame = preprocess_frame(state)
    if is_new:
        stacked_frames = np.stack([frame] * 4, axis=2)
    else:
        stacked_frames = np.append(stacked_frames[:, :, 1:], np.expand_dims(frame, axis=2), axis=2)
    return stacked_frames

class SpaceInvadersEnv:
    def __init__(self, env_name='SpaceInvaders-v4', window_width=800, window_height=600):
        self.env = gym.make(env_name)
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Space Invaders AI')
        self.clock = pygame.time.Clock()
        self.fps = 60

    def render(self):
        frame = self.env.ale.getScreenRGB()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (self.window_width, self.window_height))
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))
        pygame.display.update()
        self.clock.tick(self.fps)

    def reset(self):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the first element if state is a tuple
        return state

    def step(self, action):
        action_with_fire = action
        if action == 2 or action == 3:
            action_with_fire = 0  # Set it to fire only
        elif action == 1:
            action_with_fire = 1  # Move right and fire
        elif action == 0:
            action_with_fire = 0  # Move left and fire

        result = self.env.step(action_with_fire)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result

        # Add a small penalty for wrong shots
        if action == 2 or action == 3:
            reward -= 0.1

        # Add a penalty for losing
        if done:
            reward -= 10

        if isinstance(next_state, tuple):
            next_state = next_state[0]  # Extract the first element if next_state is a tuple
        return next_state, reward, done, info

    def close(self):
        self.env.close()
        pygame.quit()

def main():
    pygame.init()
    env_name = 'SpaceInvaders-v4'
    env = SpaceInvadersEnv(env_name)
    state_size = (84, 84, 4)
    action_size = env.env.action_space.n
    agent = DoubleDQNAgent(state_size, action_size)
    batch_size = 2
    episodes = 50

    for e in range(episodes):
        state = env.reset()
        state = preprocess_frame(state)
        stacked_frames = np.stack([state] * 4, axis=2)
        state = np.expand_dims(stacked_frames, axis=0)
        total_reward = 0
        for time in range(5000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            stacked_frames = stack_frames(stacked_frames, next_state, False)
            next_state = np.expand_dims(stacked_frames, axis=0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            print(f"Episode: {e+1}, Time step: {time}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
            if done:
                agent.update_target_model()
                print(f"Episode: {e+1} finished with score: {total_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    env.close()

if __name__ == "__main__":
    main()
