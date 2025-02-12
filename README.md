# ML-Space-Invaders

How cool would it be for Machine Learning to try Space Invaders? This project sets out to do just that—using reinforcement learning and a Double Deep Q-Network (Double DQN) to teach an agent to play (and hopefully beat) the classic arcade game. Fork this repository, experiment, and share your improvements!

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Agent Architecture](#agent-architecture)
- [Environment Customization](#environment-customization)
- [Code Structure](#code-structure)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

ML-Space-Invaders is an experimental project that leverages a Double DQN agent to learn how to play Space Invaders. By processing raw game frames with a convolutional neural network (CNN) and employing a custom reward system, the agent gradually improves its gameplay through trial and error. The project demonstrates practical techniques in deep reinforcement learning and serves as a starting point for further research and community contributions.

---

## Key Features

- **Double DQN Agent:** Utilizes two networks—an online network for real-time action selection and a target network for stable Q-value estimation.
- **Convolutional Neural Network (CNN):** Processes 84×84 pixel game frames to extract relevant features.
- **Frame Preprocessing:** Converts game frames to grayscale, resizes them, normalizes pixel values, and stacks multiple frames to capture temporal information.
- **Custom Environment Wrapper:** Enhances the OpenAI Gym `SpaceInvaders-v4` environment with:
  - Custom action mapping (e.g., mapping firing, moving left/right, no-op).
  - Detailed reward shaping to encourage actions like accurate shooting and dodging while penalizing missed shots and life loss.
  - Pygame-based rendering for visual feedback.
- **Experience Replay:** Uses a replay memory to store past experiences and train on mini-batches, improving training stability.

---

## Installation

### Prerequisites

Ensure you have Python 3x installed along with the following libraries:

- TensorFlow (with Keras)
- OpenAI Gym
- NumPy
- OpenCV (cv2)
- Pygame

## Installation 

pip install tensorflow gym numpy opencv-python pygame
Note:
For the Atari environment (SpaceInvaders-v4), make sure that you have an Atari emulator (like Stella) installed and properly configured.

Usage
To start training the agent and see it in action, run the main script:

bash
Copy
Edit
python main.py
The script will:

Initialize the custom Space Invaders environment.
Create and train the Double DQN agent over several episodes.
Render the game using Pygame so you can observe the agent’s progress.
Log actions, rewards, and overall performance per episode.
At the end of the training loop, you'll see a printed message:

Input: Receives a state represented as a stack of 4 preprocessed 84×84 frames.
Convolutional Layers: Extract features using:
32 filters (8×8 kernel, stride 4)
64 filters (4×4 kernel, stride 2)
Two consecutive layers with 128 filters each (3×3 kernel)
Max-Pooling Layer: Reduces spatial dimensions.
Dense Layers: Further processes features with fully connected layers (512, 512, and 256 neurons).
Output Layer: Provides Q-values for each available action.
Key training components include:

Epsilon-Greedy Policy: Balances exploration and exploitation.
Experience Replay: Uses a deque to store and sample past experiences.
Target Network Updates: Periodically copies weights from the online network to stabilize training.
Environment Customization
The SpaceInvadersEnv class wraps the Gym environment with enhancements such as:

Custom Action Mapping:

0: Move left
1: Move right
2: Fire (with bonus rewards)
3: No-op (do nothing)
Custom Reward System:

Shooting: Gains a bonus for firing, with additional rewards for successful shots and penalties for misses.
Dodging: Rewards movement if it avoids losing lives.
Life Loss: Imposes a penalty for losing a life.
Winning: Adds bonus rewards if a winning score threshold is reached.
Frame Preprocessing & Stacking:
Uses OpenCV to convert frames to grayscale, resize, and normalize. The stack_frames function stacks sequential frames to provide motion context to the agent.

Rendering:
Utilizes Pygame to display the game window with configurable resolution and FPS.

Code Structure
DoubleDQNAgent: Implements the agent’s neural network, memory replay, and learning algorithm.
SpaceInvadersEnv: Provides a custom environment with modified reward logic and action mapping.
Helper Functions:
preprocess_frame: Preprocesses raw game frames.
stack_frames: Maintains a stack of frames for state representation.
Main Function:
Orchestrates the training loop, resets the environment, manages rendering, and triggers training via experience replay.
Future Enhancements
Pretrained Model Integration:
The repository will soon include pretrained models to allow users to bypass lengthy initial training phases.

 community feedback is greatly appreciated.
