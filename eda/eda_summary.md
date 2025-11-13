# Reflection & Interpretation
### Tymon Vu (tvu38@calpoly.edu) and Ian Wong (iwong12@calpoly.edu)

## Dataset Selection
Data files live [here](../data/), and were sourced from [Shane McDonald's Computer Solution Repository](https://shanemcd.org/).
- source, structure, relevance
- What is this dataset and why did we choose it?

## Exploratory Data Analysis (EDA)
- Key variables
- Data volume
- Missingness
- Potential target or interaction signals
- _What did we learn from our EDA?_

The image below shows how strongly each team was expected to lose on average. 
We can see that the Colorado Rockies were underdogs in most of their wins,
whereas the LA Dodgers were heavily favored in most of their wins. This graphic serves as a sort of proxy for power ratings.

![Average ML for Wins by Team](imgs/avg_winner_ml.png)

This next image shows the total number of runs scored in a game against the over/under score (averaged).
We see that O/U is generally an accurate predictor of whether a game will be low- or high-scoring.

![Runs vs. O/U](imgs/runs_ou.png)

This last image shows the distribution of total wins for each team throughout the 2024-2025 season.
Color also indicates home versus away wins. As expected, we find that the Rockies won the least games and the Dodgers won the most.
This aligns with our previous findings.

![Wins by Team](imgs/wins_by_team.png)

## Looking Ahead
- Initial ideas for features and any anticipated challenges (imbalance, sparcity, etc.)
- What issues or open questions remain?