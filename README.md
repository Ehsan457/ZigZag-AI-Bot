# Zig-Zag Game Automation

This project automates the "Zig-Zag" game using Python. The bot uses computer vision techniques to detect the ball's position and dynamically adjust its movement. It leverages the Playwright library for interacting with the browser and simulating key presses.

## Features

- **Automated Gameplay**: Automatically starts and plays the Zig-Zag game.
- **Ball Detection**: Uses computer vision to detect the ball's position.
- **Dynamic Direction Change**: Adjusts the ball's direction based on detected obstacles.
- **Edge Detection**: Utilizes Canny edge detection to identify game boundaries and obstacles.
- **Playwright Integration**: Interacts with the game through a web browser using Playwright.

## Installation

### Prerequisites

- Python 3.7+
- [Playwright](https://playwright.dev/python/docs/intro)
- [OpenCV](https://opencv.org/)

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/zig-zag-game-automation.git
    cd zig-zag-game-automation
    ```

2. **Install dependencies**:

    ```bash
    pip install -r Requirements.txt
    ```

3. **Install Playwright browsers**:

    ```bash
    playwright install
    ```

## Usage

1. **Run the automation script**:

    ```bash
    python main.py
    ```

2. The script will open the Zig-Zag game in a browser, start the game, and attempt to play it by automatically pressing the space bar to change direction.

## Project Structure

- `main.py`: The main script to run the automation.
- `assets/`: Directory containing the `ScoreBoard.png` used for template matching.
- `images/`: Directory where processed images are saved during each iteration.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.
