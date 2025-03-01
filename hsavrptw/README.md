# Harmony Search Algorithm for Vehicle Routing Problem with Time Windows

A slight modification to tweinyan's implementation of hsa.

Original repository: https://github.com/tweinyan/hsavrptw/tree/master

This is implementation of
Harmony Search Algorithm for Vehicle Routing Problem with Time Windows
based on the work with the same title by Esam Taha Yassen et al.
You can read the paper here:
http://scialert.net/qredirect.php?doi=jas.2013.633.638&linkid=pdf

Python is chosen as an implementation language
and pyHarmonySearch as Harmony Search Implementation.
Check out https://github.com/gfairchild/pyHarmonySearch/

This repository contains:
- VRPTWObjectiveFunction.py - main script
- problems/ - directory with problems
- pyharmonysearch/ - directory with pyHarmonySearch library
- run.sh - script for running the main script with the full dataset
- run_test.sh - script for running the main script with the mini dataset

## How to Run This Project

### Prerequisites

- Python 3.x

### Running the Main Script

To run this project, there are two scripts available: one for testing with the mini dataset and the other for testing with the full dataset.

Note that to run the scripts, you need to have WSL installed.

To run the script for testing with the mini dataset, execute the following command:

```bash
./run_test.sh
```

Or you can paste the following command in the terminal if you don't feel like installing at the moment
```sh
python vrptw_objective_function.py problems/minimal.txt --hms=20 --hmcr=0.7 --parmax=0.9 --parmin=0.3 --ni=1000
```

To run the script for testing with the full dataset, execute the following command:

```bash
./run.sh
```
Or you can paste the following command in the terminal if you don't feel like installing at the moment
```sh
python vrptw_objective_function.py problems/C101.txt --hms=20 --hmcr=0.7 --parmax=0.9 --parmin=0.3 --ni=1000
```