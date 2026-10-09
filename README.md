# IMMC 2025 — International Football Tournament Modelling

Python and Jupyter notebook experiments for comparing international football tournament formats, grouping countries geographically, exploring match schedules, and choosing match times for audience engagement.

The project studies the trade-offs between ranking accuracy, the number of matches, travel time, and how evenly the workload and travel are distributed between teams. Country names, Elo ratings, coordinates, group assignments, and model weights are defined directly in the scripts and notebooks.

## Where to start

- **Tournament comparison:** [`manstermodel.ipynb`](manstermodel.ipynb) brings together seven tournament formats using 24 countries and five evaluation metrics.
- **Geographic grouping:** [`singlerobin-grouped.ipynb`](singlerobin-grouped.ipynb) compares single round-robin tournaments with two through eight geographic groups.
- **Audience engagement:** [`timetable1.ipynb`](timetable1.ipynb) combines hourly viewing patterns with country fan indices to identify a three-hour viewing window.
- **Script version:** [`Mastermodel.py`](Mastermodel.py) contains a 20-country comparison with four evaluation metrics and `# %%` cell markers for interactive editors.

These are exploratory research files. Each notebook contains its own setup and model definitions; there is no shared package or single application entry point. Earlier experiments use different country lists and assumptions.

## Models and evaluation

### Tournament simulation

Match outcomes are sampled using the Elo win probability:

```text
P(A beats B) = 1 / (1 + 10 ** ((Elo_B - Elo_A) / 400))
```

The active match simulation chooses a win or loss, awarding three points to the winner and zero to the loser. Although some functions include a draw branch, their random outcome selection does not generate draws. Elo ratings remain fixed during a simulation.

The consolidated models compare:

1. Full single round robin.
2. Full double round robin.
3. Single round robin with unbalanced geographic groups.
4. Double round robin with unbalanced geographic groups.
5. Single round robin with balanced groups.
6. Double round robin with balanced groups.
7. Random elimination.

Grouped formats combine group matches, a qualifying round robin, an eight-team knockout, and placement matches. Exact group sizes and advancement rules depend on the file: the 20-country script uses five balanced groups of four, while the 24-country main notebook uses six groups of four despite some older section titles.

### Comparison metrics

| Metric | What it measures |
| --- | --- |
| `games` | Average total number of matches per tournament. |
| `time` | Average estimated total flight time. |
| `deviation` | Standard deviation across teams of their average number of matches. |
| `loss` | Weighted pairwise inversions between the simulated ranking and the ranking obtained by sorting Elo ratings. |
| `airtimedev` | Mean within-tournament standard deviation of team flight times, divided by average total flight time; included in the main notebook and related experiments. |

Travel estimates use Haversine distances between country coordinates and an assumed flight speed of 900 km/h. The main notebook counts return travel and randomly assigns each match's travel burden to one of its teams. This is a simplified travel proxy; the scheduling experiments separately explore routes and home/away assignments.

The consolidated analysis applies min–max normalization to each metric and calculates:

```text
overall_index = 1 - sum(normalized_metric * metric_weight)
```

Higher overall indices indicate a better score under the selected weights. The notebooks and script differ in loss definitions, travel accounting, and included metrics, so results should be interpreted within the file that produced them.

## File guide

### Tournament formats and analysis

| File | Purpose |
| --- | --- |
| [`Mastermodel.py`](Mastermodel.py) | Consolidated 20-country simulation and four-metric comparison; defaults to one trial. |
| [`manstermodel.ipynb`](manstermodel.ipynb) | Consolidated 24-country simulation, travel-distribution analysis, and five-metric comparison; defaults to 1,000 trials. |
| [`manstermodel-sensitivity_analysis.ipynb`](manstermodel-sensitivity_analysis.ipynb) | Related 20-country comparison with five metrics and fixed weights. |
| [`manstermodel-sensitivity analysis1.ipynb`](manstermodel-sensitivity%20analysis1.ipynb) | 20-country comparison that also increases and decreases each metric weight by 10% individually. |
| [`Fullsingleroundrobin.ipynb`](Fullsingleroundrobin.ipynb) | Standalone full single round-robin experiment. |
| [`Doublefullrobin.ipynb`](Doublefullrobin.ipynb) | Standalone full double round-robin experiment. |
| [`singlerobin-5groupsof4.ipynb`](singlerobin-5groupsof4.ipynb) | Single round robin with five groups of four, qualification, knockout, and placement stages. |
| [`doublerobin-5groupsof4.ipynb`](doublerobin-5groupsof4.ipynb) | Double round-robin version of the five-group format. |
| [`singlerobin-unbalancedgroups.ipynb`](singlerobin-unbalancedgroups.ipynb) | Single round robin with group sizes of 4, 4, 4, and 8. |
| [`doublerobin-unbalancedgroups.ipynb`](doublerobin-unbalancedgroups.ipynb) | Double round-robin version of the unbalanced-group format. |
| [`singlerobin-grouped.ipynb`](singlerobin-grouped.ipynb) | Geographic clustering and tournament evaluation across two to eight groups for 24 countries. |
| [`matchmaking2.ipynb`](matchmaking2.ipynb) | Earlier four-pool experiment with a knockout for the top eight and a consolation round robin for the remaining twelve. |
| [`matchmaking3.ipynb`](matchmaking3.ipynb) | Earlier group/qualifier/knockout experiment with a placement round robin for the bottom twelve. |
| [`knockout.ipynb`](knockout.ipynb) | Early elimination-bracket experiment plotting champion frequencies. |
| [`evaluation.ipynb`](evaluation.ipynb) | Elo probability and weighted ranking-loss exploration. |

### Clustering, scheduling, and viewing times

| File | Purpose |
| --- | --- |
| [`k-means.ipynb`](k-means.ipynb) | Custom capacity-constrained geographic clustering using Haversine distances; maps five groups for 20 countries. |
| [`k-meansvisualisation.ipynb`](k-meansvisualisation.ipynb) | Maps countries and centroids during clustering into six groups for 24 countries. |
| [`Schedule1.ipynb`](Schedule1.ipynb) | Randomized scheduling experiment with home/away streak, Elo-gap, travel, and jet-lag considerations. |
| [`Schedule2.ipynb`](Schedule2.ipynb) | Greedy scheduling experiment that accepts matches subject to home/away and Elo-gap checks. |
| [`test.ipynb`](test.ipynb) | OR-Tools CP-SAT scheduling prototype for 20 teams over 180 days, with home/away counts, rest constraints, and a combined travel/Elo objective. |
| [`plot_schedule.py`](plot_schedule.py) | Reads a schedule matrix from `solution.txt` and writes Folium maps of countries and a selected team's route. |
| [`timetable1.ipynb`](timetable1.ipynb) | Viewing-time model with fan indices, longitude-based time offsets, and a search for the best three consecutive hourly slots. |
| [`timedata.txt`](timedata.txt) | 24 hourly intervals with three numeric columns; the viewing-time notebook uses the last numeric column. |
| [`output.png`](output.png) | Saved engagement and distribution chart for South Korea and New Zealand. |
| [`Japan and NZ.png`](Japan%20and%20NZ.png) | Saved Japan–New Zealand engagement chart with a highlighted viewing window. |
| [`Uruguay and Italy.png`](Uruguay%20and%20Italy.png) | Saved Uruguay–Italy engagement chart with a highlighted viewing window. |

## Running the notebooks

Notebook metadata records Python 3.11.5. The repository does not contain a dependency lockfile or pinned package versions. To prepare a Python environment, install the packages imported by the project:

```bash
python -m pip install jupyterlab ipython numpy pandas matplotlib folium pytz ortools
```

From the repository folder, launch JupyterLab:

```bash
python -m jupyterlab
```

Open a notebook and run its cells in order from a fresh kernel. Most tournament and clustering notebooks contain their data inline. `timetable1.ipynb` expects `timedata.txt` in the current working directory. OR-Tools is used by `test.ipynb`; `pytz` is imported by the scheduling and viewing-time experiments.

The consolidated script can also be run directly:

```bash
python Mastermodel.py
```

It executes all seven format comparisons, displays Matplotlib charts, and prints metrics and overall indices. Its Folium display is best viewed in a notebook or an interactive editor that supports rich output.

## Practical notes

- Simulation results vary between runs because most experiments do not fix random seeds. Trial counts are configured inside each file and can make runs lengthy.
- Saved notebook outputs are snapshots of earlier runs; they are not a guarantee that every current cell runs successfully from a fresh kernel. `test.ipynb` includes a saved missing-OR-Tools error, and `Schedule1.ipynb` includes an interrupted execution.
- `Schedule1.ipynb` currently initializes ten countries but defines jet-lag time zones for only four different countries and later selects New Zealand for mapping. It therefore needs its input definitions reconciled before it can run end to end. Its randomized schedule search also has no fixed retry limit.
- `plot_schedule.py` requires `solution.txt`, which is not included in this repository. The expected input is whitespace-separated integer rows representing venue indices for each team in each round. Running it writes `map.html` and `map_with_lines.html`.
- The viewing-time notebook applies one hourly viewing profile to all countries, adjusts hours using longitude rather than actual time-zone rules, and combines it with hard-coded fan indices. Its first chart uses explicitly labelled sample data.
- The notebooks preserve several versions of the research. Some markdown headings and comments describe earlier configurations; use each file's active country lists, groups, and formulas when interpreting its results.

## Example visualization

The Japan–New Zealand chart combines the two countries' weighted viewing distributions and highlights the selected viewing window:

![Japan and New Zealand audience engagement by hour](Japan%20and%20NZ.png)
