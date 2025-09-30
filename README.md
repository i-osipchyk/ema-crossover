# Trading Strategy Research & Alerts

This project is about making an analysis and potentially building an application, that focuses on a simple trading strategy. As of September 16, 2025 it uses 8 and 20 EMA to identify trend in stocks, hence the repository name.

- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [How It Works](#how-it-works)  
- [Usage](#usage)  
- [Updates](#updates)

## Overview

The idea behind the project is that it is possible to use simple indicators and stock selection to identify setups, and get good Expected Value with the help of risk management and position sizing. The project focuses not only on the analysis, that will show if the idea is applicable or not, but potentially on building an application, that will automatically detect trading setups, monitor trading activity etc. 

First results show that this approach works better or more expensive stocks, where the presence of institutions is higher. Therefore, it may be a good idea to rely not only on statistics, but also on sentiment using LLMs.

## Features

#### ✅ Current
- Downlaods data in a specified format
- Adds indicators that I found helpful so far
- Run grid searches over trading strategy parameters
- Aggregate trades and calculate expectancy, win rate, and R-multiples
- Apply technical filters (EMA alignment, MACD, crossover volume, etc.)

#### 🔮 Planned
- Pipeline for ease of adding new indicators and altering strategy
- Daily stock scan & setup alerts
- LLM-powered sentiment screening
- Configurable alert system (email, messenger, or push notifications)
- Live strategy monitoring

## Tech Stack

- **Python**: core backtesting & analysis
- **Pandas / Numpy**: data processing
- **Yfinance**: stock data
- **Matplotlib / Seaborn**: visualization
- **(Planned)** LLM APIs for sentiment analysis
- **(Planned)** notification and monitoring system

## How It Works

1. Load historical stock data
2. Add indicators
3. Run entry parameters grid search and flatten results
4. Run position sizing grid search
5. Filter trades by technical parameters
6. Summarize performance metrics

## Usage

TODO

## Updates

- **September 16, 2025:** executed first entry parameters grid search on full historical data. **Next:** stock selection
- **September 17, 2025:** performed stock selection and executed position sizing grid search. Achieved results of **EV≈0.478** and **Accuracy≈60.6%**. **Next:** set up daily evaluation pipeline
- **September 21, 2025:** added daily setup selection pipeline
- **September 24, 2025:** added automatic evaluation pipeline
- **September 27, 2025:** added small grid search to automatic evaluation pipeline
- **September 29, 2025:** not commited yet, but in the second stage of analysis in found out that the error made in the first one is bigger than I thought. There is incorrect entry condition: if there is a gap between crossover day high and entry day open, entry price will be crossover day high even though it wasn't possible to make that entry in regular trading hours. That draws down EV quite a lot, though it remains positive and there are hopes to increase it. Evaluation pipeline was configured correctly from the beginning. With the number of trades it generates daily it can be soon used for the evaluation. Meanwhile, I continue with the second aprt of analysis to improve the strategy. This situation opened a few possibilities in front of me, for example using gap as a filter.
