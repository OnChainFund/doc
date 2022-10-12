# AI Portfolio Management
###### `tags: OnChainFund`
## Introduction
Deep learning algorithms is used for portfolio optimization.
### Data
We consider a panel data of returns $R_{n,t}$, where time $t = 1, 2, ..., T$  is on hourly basis and $n = 1,2,...,N$ denotes the $n^{th}$ stock in our portfolio.<br>
A sliding window of size $L$ (lookback) is used to generate out dataset.
### Network
Our network architecture is mainly consisted of 2 parts:
- CNN<br>
The network takes as input a window $x^{(i)} = x \in \mathbb{R}^L$ of $L$(= 360) consecutive hourly log returns or log prices, and outputs the feature $\tilde{x} \in M_{L\times F}(\mathbb{R})$ given by computing following quantities for $l = 1,...,L ;\,\,d = 1,...,D$. Where D is the number of convolutional features.
- Transformer<br>
Transformer model is first introduced in 2017 by a team at Google Brain and are increasingly the model of choice for NLP problems, replacing RNN based models such as long short-term memory (LSTM). The additional training parallelization allows training on larger datasets. While tansformer model has become the first choice for sequence modeling, we transformer model to capture dependencies between the time series features that CNN model extracts.

## Mean Variance Objective
Our deep learning model is estimated with a Sharpe ratio objective. For estimated long-short portfolio, the sum of absolute stock weights is normalized to 1, which imposes a  leverage constraint. For portfolios with short selling constraints, the sum of stock weights is normalized to 1.
\begin{equation}
\begin{aligned}
&\max_{w^{\epsilon}\in W,\theta \in \Theta} \quad  \frac{\mathbb{E}[w_{t-1}^{R}R_t]}{\sqrt{Var(w_{t-1}^{R}R_t)}} \quad \quad \text{or}\quad \max_{w^{\epsilon}\in W,\theta \in \Theta} \mathbb{E}[w_{t-1}^{R}R_t] - \gamma Var(w_{t-1}^{R}R_t)\quad\\
&s.t\quad\quad ||w_{t-1}^{R}||_{1} = 1
\end{aligned}
\end{equation}

## Rebalancing

We rebalance our portfolio every 5 days. With data of past 15 days, our model gives the portfolio weight for the following 5 days.

