# Solving Sudoku #

## Description ##
This code is a part of my final project for my Intro to AI class. It the experimental component of an analysis for 5 solving algorithms for Sudoku puzzles

Three of these could be considered backtracking-based: backtracking (DFS), backtracking with arc consistency (AC-3), and dancing links (DLX)

Two are soft computing-based or stochastic algorithms: genetic algorithm (GA) and simulated annealing (SA)

## Execution ##
To execute the experiment, just run main.py! Additional analysis points can be gathered from running calculate_stats.py.
This will populate the solution_results.csv file and create various plots in the "plots" directory.
The number of iterations at each difficulty and which algorithms will be run can be customized from within main.py

## Sources ##
Code in this repository is borrowed and modified from:\
    [Puzzle generator by RutledgePaulV](https://github.com/RutledgePaulV/sudoku-generator)\
    [DFS with AC-3 by aurbano](https://github.com/aurbano/sudoku_py)\
    [DLX by sraaphorst](https://github.com/sraaphorst/dlx-python)\
    [GA by ctjacobs](https://github.com/ctjacobs/sudoku-genetic-algorithm)\
    [SA by erichowens](https://github.com/erichowens/SudokuSolver)
    
