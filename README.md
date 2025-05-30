# ML Algorithms

Welcome to **ML Algorithms** repository! This project provides a collection of machine learning algorithms implemented from scratch. It is designed to help both beginners and experienced practitioners gain a deeper understanding of machine learning fundamentals and advanced techniques through hands-on experimentation.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Notebooks & Variations](#notebooks--variations)
6. [Contributing](#contributing)
7. [License](#license)

## Overview

This repository contains implementations of popular machine learning algorithms, including linear regression, logistic regression, and perceptron. Each algorithm is built from scratch to help you understand the underlying mathematics and computational techniques.

## Features

- **Linear Regression**: Both basic and regularized versions.
- **Logistic Regression**: From-first-principles implementations.
- **Perceptron**: A simple neural network model.
- **Educational Notebooks**: Detailed Jupyter notebooks for exploring each algorithm.
- **Extensible Codebase**: Easily add new algorithms and enhancements.

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```shell
   git clone https://github.com/msparsh/ml-algorithms.git
   cd ml-algorithms
   ```

2. **Create a virtual environment:**
   ```shell
   python3 -m venv env
   source ./env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install dependencies:**
   ```shell
   pip install -r requirements.txt
   ```

## Usage

Explore the Jupyter notebooks included in the root and [`variations`](variations/Readme.md) directory to see the algorithms in action. Examples include:

- `Using linear regression.ipynb`
- `Using logistic regression.ipynb`
- Additional versions in the [`variations`](variations/Readme.md) folder (e.g., regularized, polynomial, basic variants)

## Notebooks & Variations

The [`variations`](variations/Readme.md) folder contains several alternative implementations and enhancements:
- **Basic Implementations:** Simplified variants for easier understanding.
- **Batch and Polynomial Variations:** Additional processing approaches.
- **Regularized Regression:** Notebooks demonstrating techniques to reduce overfitting.

## Contributing

Contributions are welcome! To contribute, follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**
   ```shell
   git checkout -b your-feature-name
   ```
   or 
   ```shell
   git branch your-feature-name
   git checkout your-feature-name
   ```
3. **Make your changes.**
4. **Add and commit your changes:**
   ```shell
   git commit -m "Add your commit message here"
   ```
5. **Push to your branch:**
   ```shell
   git push origin your-feature-name
   ```
6. **Open a pull request** with a detailed description of your changes.

### Code Style and Guidelines

Please follow the existing code style. Ensure that any changes enhance the clarity and functionality of the project.

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for more details.

