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

        # Adding more convolutional layers
        model.add(keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
        model.add(keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))  # More feature extraction
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))  # Even more feature extraction
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Pooling layer to reduce dimensionality

        # Flatten layer
        model.add(keras.layers.Flatten())

        # Adding more fully connected layers
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))  # Adding another dense layer for better feature learning
        model.add(keras.layers.Dense(256, activation='relu'))  # Another dense layer with fewer units

        # Output layer with action size
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
        self.last_shot_time = 0  # Track when the last shot was fired
        self.shot_penalty_time = 20  # The max time before penalty for not firing
        self.last_lives = 3  # Initial lives count to track loss of life
        self.last_score = 0  # Initial score to track successful shots

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
        self.last_lives = 3  # Reset lives on a new episode
        self.last_score = 0  # Reset score on a new episode
        return state

    def step(self, action):
        action_with_fire = action
        fired_shot = False
        reward = 0

        # Modify action mapping based on your custom action scheme
        if action == 2 or action == 3:  # Firing
            action_with_fire = 0
            fired_shot = True
        elif action == 1:  # Move right and fire
            action_with_fire = 1
        elif action == 0:  # Move left and fire
            action_with_fire = 0

        # Apply action in environment
        result = self.env.step(action_with_fire)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result

        # Track time since last shot and apply penalty if too long
        if fired_shot:
            self.last_shot_time = 0  # Reset shot time if a shot was fired
        else:
            self.last_shot_time += 1
            if self.last_shot_time > self.shot_penalty_time:  # Too long without shooting
                print(f"Penalty: Took too long without firing. Last shot time: {self.last_shot_time}")
                reward -= 5  # Increased penalty for not firing

        # Reward for successful shots (destroyed enemies or enemies hit)
        if 'score' in info and info['score'] > self.last_score:
            print(f"Reward: Successful shot! Score increased from {self.last_score} to {info['score']}")
            reward += 3  # Reward for hitting a target (enemy killed)

        # Reward for dodging bullets (you can define this based on the agent's behavior or environment state)
        if 'lives' in info and info['lives'] > self.last_lives:
            print(f"Reward: Dodged a bullet! Lives increased to {info['lives']}")
            reward += 3  # Reward for dodging bullets (staying alive)
        self.last_lives = info.get('lives', self.last_lives)

        # Penalty for missing shots
        if fired_shot and 'score' in info and info['score'] == self.last_score:
            print(f"Penalty: Missed shot. No score change. Current score: {self.last_score}")
            reward -= 5  # Increased penalty for missing a shot

        # Penalty for losing life or dying
        if done:
            print(f"Penalty: Game Over! Losing life or dying.")
            reward -= 20  # Strong penalty for losing the game

        # Penalty for not dodging bullets (getting hit and losing life)
        if 'lives' in info and info['lives'] < self.last_lives:
            print(f"Penalty: Hit by bullet! Lives decreased from {self.last_lives} to {info['lives']}")
            reward -= 5  # Increased penalty for getting hit (losing a life)

        # Update last score to track changes
        self.last_score = info.get('score', self.last_score)

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
