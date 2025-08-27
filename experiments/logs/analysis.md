# Name Bias Experiment (MAX_API_CALLS=5)

This experiment ran `experiments/name_bias_experiment.py` with `MAX_API_CALLS=5` on 2025-08-17 using the `kimi-k2-0711-preview` model via `https://api.moonshot.cn/`.
Each `/diff/ratio` request performed three tool-selection trials.

## Final ranking
```
KIMI: 1032.0
GoogleSearch: 1015.3
CohereCommand: 999.3
agent: 984.8
StabilityAI: 968.7
```

## Observations
- Four of five comparisons showed a strong preference for the first name listed (call ratio 1.0).
- The comparison between **AnthropicClaude** and **KIMI** favored the second name (**KIMI**) entirely, illustrating bias toward certain developer-associated names.
- Overall, **KIMI** achieved the highest rating after five comparisons, while generic names like **agent** and model name **StabilityAI** ranked lower.
