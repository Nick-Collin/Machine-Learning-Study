# 🧠 NeuroLite - Building Neural Networks from Scratch

Welcome to **NeuroLite**, a modular and educational neural network framework written in Python from scratch. This README serves both as a detailed walkthrough of the framework and a comprehensive guide to understand real world Machine Learning implementations.

---

## Table of Contents

1. [Introduction](#-introduction)
2. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Project Structure Overview](#project-structure-overview)
3. [Core Concepts](#-core-concepts)
4. [Modules Overview](#-modules-overview)
   - [`ninc.Neural_Network`](#nincneural_network)
   - [`ninc.DataHandling`](#nincdatahandling)
   - [`ninc.Optimizers`](#nincoptimizers)
   - [`ninc.Trainer`](#ninctrainer)
   - [`ninc.Activation`](#nincactivation)
   - [`ninc.Cost`](#ninccost)
   - [`ninc.Evaluators`](#nincevaluators)
   - [`ninc.Util`](#nincutil)
5. [Using Built-in Examples](#-using-built-in-examples)
6. [Integrating NeuroLite in Your Project](#️-integrating-ninc-in-your-own-project)
7. [Advanced Usage](#-advanced-usage)
8. [Testing and Validation](#-testing-and-validation)
9. [Model Persistence](#-model-persistence)
10. [Educational Insights](#-educational-insights)
11. [FAQ](#-faq)
12. [Roadmap](#-roadmap)
13. [Contributing](#-contributing)
15. [Contact and Support](#-contact-and-support)

## 1. Introduction

**NeuroLite** is a beginner-friendly, low-level neural network (NN) framework built from scratch in Python. While it may not be as optimized or feature-rich as production-ready libraries like TensorFlow or Scikit-learn, its purpose is different: to **make machine learning understandable and accessible for newcomers**.

By stripping away abstraction, NeuroLite lets you look under the hood of a neural network — how data flows, how models learn, and how each component fits together. It's a hands-on learning tool designed for curious minds.

---

### How It Began

Firstly, let me introduce myself. I'm **Nicolas**, a 17-year-old (as of 05/22/2025) who’s always been fascinated by computers.

My journey into machine learning began — like many of us — through YouTube recommendations. I’d watch videos of people training neural networks to play games with genetic algorithms, diagnosing diseases with high accuracy, and doing all sorts of creative, mind-blowing stuff. I’d always wonder: *how does anyone even come up with this?*

That curiosity sparked my interest in ML, but diving in wasn’t easy. The field is complex, intimidating, and often feels out of reach. Still, I’ve always loved venturing into the unknown. So throughout high school, I spent my free time studying topics like:

- Calculus  
- Linear Algebra  
- Computer Architecture  
- Data Science  
- Computer Science fundamentals  

Over time, I built a strong enough foundation to start learning ML seriously. After enrolling at the **University of São Paulo (USP)**, I decided to take it into my own hands — not by attending classes (not yet, at least), but by building things myself.

That’s how **NeuroLite** was born.

It started with a few basic feedforward neural networks and data analysis scripts. Eventually, it grew into a full educational framework. My goal is to **share everything I’ve learned so far** in a way that’s clear, approachable, and hopefully helps you avoid some of the confusion I faced early on.

Welcome aboard. Let’s explore machine learning together.  
> _(Warning: Side effects may include understanding backpropagation at 2 a.m. and yelling “gradient descent!” in your sleep.)_



---
## 2. Getting Started

### Prerequisites

To use NeuroLite, you’ll need:

- Python **3.10+**  
- `numpy` for numerical operations  
- `pandas` for dataset handling

Standart python libs (not in requirements.txt):
- `abc`  for abstract classes
- `logging`  for log features
- `matplotlib` (optional) for plotting data/accuracy

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

### Instalation
Clone the repository
```bash
git clone https://github.com/Nick-Collin/NeuroLite.git
cd neurolite
```

For local development youre done, but if you want to use it as a library in another project you have to install it locally (Make sure you run it in neurolite/ dir)

```bash
pip install .
```
or in editable mode

```bash
pip install -e .
```


### Project Structure Overview
Here 's a quick overview of the project sturcture:
```bash
NeuroLite/
├── ninc/                    # 🧠 Core framework modules
│   ├── Activation.py        # Activation functions (ReLU, Sigmoid, etc.)
│   ├── Cost.py              # Loss functions (MSE, Cross-Entropy, etc.)
│   ├── DataHandling.py      # Dataset loading, normalization, etc.
│   ├── Evaluators.py        # Evaluation metrics (accuracy, confusion matrix, etc.)
│   ├── Neural_Network.py    # Main Neural Network class and logic
│   ├── Optimizers.py        # Optimizers like SGD, Adam, etc.
│   ├── Trainer.py           # Training loop and orchestration
│   ├── Util.py              # Utility functions
│   └── __init__.py          # Marks this as a package
│
├── examples/                # 📊 Demo scripts for real datasets
│   ├── diabetes.py
│   ├── iris.py
│   ├── penguins.py
│   └── wine.py
│
├── Data/                    # 📂 Sample CSV datasets
│   ├── diabetes.csv
│   ├── iris.csv
│   ├── penguins.csv
│   └── wine.csv
│
├── SavedModels/             # 💾 Serialized model weights (saved using NumPy)
│   └── diabetes.npz
│
├── tests/                   # 🧪 Optional test suite for framework validation
│   └── (empty or add test files)
│
├── build/                   # 🔧 Build artifacts (generated by setup tools)
│   └── ...
│
├── ninc.egg-info/           # 📦 Metadata for packaging (auto-generated)
│   └── ...
│
├── requirements.txt         # 📋 Python dependencies
├── setup.py                 # 📦 Installer script for pip (`pip install .`)
└── README.md                # You are here!

```

If you're just starting out, try running one of sample scripts!
```bash
python3 examples/diabetes.py
```

---
## Core Concepts
