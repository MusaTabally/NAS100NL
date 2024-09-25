Backtesting and Pairs Trading Framework
Project Overview
This repository contains a C++ and Python-based framework for backtesting and executing trading strategies, particularly focused on pairs trading. It includes tools for analyzing financial spreads, calculating profit and loss (PNL), and implementing custom algorithms for identifying pairs and tracking their performance over time. Additionally, the project features a neural network developed in N1.cpp to optimize trading strategies for NAS100 futures.

Key Components
C++ Source Files:

backtest.cpp: Implements the core logic for backtesting trading strategies with a focus on pairs trading.
spread.cpp: Provides functionality to analyze the spread between two financial instruments.
N1.cpp: Implements a neural network that uses machine learning techniques to predict price movements and optimize trading strategies for NAS100 futures.
pairs.cpp: Implements the logic for identifying and tracking pairs of assets for trading.
Python Scripts:

pairs.py: A Python script designed to analyze pairs and perform real-time or historical data analysis.
Jupyter Notebook:

PNL.ipynb: A notebook that calculates profit and loss (PNL) for trades over time, visualizing performance and making it easy to analyze strategy effectiveness.
CMake Build Support:

cmake_minimum_required(VERSION 3.x): CMake configuration for compiling and building the C++ components of the project.
Key Features
Backtest various pairs trading strategies with historical data.
Calculate and analyze the spread between asset pairs to identify trading opportunities.
Develop and deploy a neural network in N1.cpp to trade NAS100 futures, leveraging machine learning to predict market trends.
Python-based analysis of PNL and strategy performance.
Easily configurable and extensible with new strategies or assets.
