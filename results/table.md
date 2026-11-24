# Policy Evaluation Summary

| Policy                |   Mean Tokens |   Mean Best Score | Oracle Hit Rate   |   Mean Regret |   Mean Leaves Explored |
|:----------------------|--------------:|------------------:|:------------------|--------------:|-----------------------:|
| beam                  |         2,118 |               0.8 | 100.0%            |             0 |                    6   |
| cost_aware_sh         |         4,972 |               0.8 | 100.0%            |             0 |                   22.7 |
| exhaustive_no_sharing |        11,090 |               0.8 | 100.0%            |             0 |                   27   |
| exhaustive_shared     |         5,588 |               0.8 | 100.0%            |             0 |                   27   |
| greedy                |         1,232 |               0.8 | 100.0%            |             0 |                    3   |
| random_prune          |         4,893 |               0.8 | 100.0%            |             0 |                   23.2 |
| sh_cost_blind         |         2,574 |               0.8 | 100.0%            |             0 |                    6   |
