---
license: apache-2.0
pretty_name: A* solutions to Boxoban levels
task_categories:
- reinforcement-learning
tags:
- sokoban
- planning
size_categories:
- 1M<n<10M
---

# A* solutions to Boxoban levels

For some levels we were not able to find solutions within the allotted A* budget. These have solution
`SEARCH_STATE_FAILED` or `NOT_FOUND`. These are the ones labeled "Unsolved levels" below.

The search budget was 5 million nodes to expand for medium-difficulty levels, vs. 1 million nodes for
unfiltered-difficulty levels. The heuristic was the sum of Manhattan distances of each box to its closest target.

## Summary table: 

| Level file                | Total size | Unsolved levels | Search budget |
|:--------------------------|-----------:|----------------:|--------------:|
| `unfiltered_train.csv.gz` |    900,000 |             495 |      1M nodes |
| `unfiltered_valid.csv.gz` |    100,000 |             623 |      1M nodes |
| `unfiltered_test.csv.gz`  |      1,000 |              11 |      1M nodes |
| `medium_valid.csv.gz`     |     50,000 |               1 |      5M nodes |


## Solution format
The solution is a sequence of actions to take, where 

| Number | Action |
|--------|--------|
| 0      | Up     |
| 1      | Right  |
| 2      | Down   |
| 3      | Left   |

## Loading the dataset

`dtype=str` is needed so the string of moves that form the solution isn't converted to a number.

```python
import huggingface_hub
import pandas as pd

ds_path = huggingface_hub.snapshot_download("AlignmentResearch/boxoban-astar-solutions", repo_type="dataset")
df = pd.read_csv(ds_path + "/unfiltered_train.csv.gz", dtype=str, index_col=("File", "Level"))
```

## Citation

If you use this dataset, please cite our work:

```bibtex
@inproceedings{garriga-alonso2024planning,
    title={Planning behavior in a recurrent neural network that plays Sokoban},
    author={Adri{\`a} Garriga-Alonso and Mohammad Taufeeque and Adam Gleave},
    booktitle={ICML 2024 Workshop on Mechanistic Interpretability},
    year={2024},
    url={https://openreview.net/forum?id=T9sB3S2hok}
}
```
