# 🦖 AI Dino Game (NEAT Python)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pygame](https://img.shields.io/badge/Library-Pygame-yellow)
![AI](https://img.shields.io/badge/AI-NEAT-red)

A **"God Level"** implementation of the classic Chrome Dino Game, reinforced with **NeuroEvolution of Augmenting Topologies (NEAT)** to create an AI that learns to play the game by itself.

## 🚀 Features
- **Self-Learning AI:** Uses genetic algorithms to evolve neural networks that master the game.
- **Real-time Evolution:** Watch generation after generation improve from clumsy jumps to god-like reflexes.
- **Robust Architecture:** Built with `pygame` for rendering and `neat-python` for the AI brain.
- **Visual Statistics:** Live counter of alive dinos and current score.

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/CodeWithHarry123/AI_dino_game_N.git
   cd AI_dino_game_N
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Run

Simply execute the main script:

```bash
python main.py
```

The AI will start training immediately. You'll see multiple dinos (population) attempting to jump over obstacles. As they die, the best performing ones reproduce and mutate to create the next generation.

## 🧠 How it Works (NEAT)

The AI uses a **Feed-Forward Neural Network** where:
- **Inputs:**
  1. Dino's Y position (Height)
  2. Distance to the next obstacle
- **Output:**
  - Jump (Activation > 0.5) or Run

The fitness function rewards dinos for surviving longer and penalizes them for hitting obstacles.

## 📂 Project Structure
- `main.py`: The core game loop and NEAT integration.
- `config-feedforward.txt`: Configuration parameters for the NEAT algorithm (population size, mutation rates, etc.).
- `assets/`: Directory for game sprites (Dino, Cactus, Track). *Auto-fallback to shapes if images are missing.*
- `requirements.txt`: Python dependencies.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests to improve the fitness function or add more obstacles (Birds, Pterodactyls)!

---
*Created for the love of AI and Retro Games.*