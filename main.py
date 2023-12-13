""""
MASON MCFARLAND
CSCI 4511W
Final project

'Solving Sudoku'

Implementations used:
    [1] https://github.com/RutledgePaulV/sudoku-generator by RutledgePaulV
    [2] https://github.com/aurbano/sudoku_py by aurbano
    [3] https://github.com/sraaphorst/dlx-python by sraaphorst
    [4] https://github.com/ctjacobs/sudoku-genetic-algorithm by ctjacobs
    [5] https://github.com/erichowens/SudokuSolver by erichowens
"""

# import generator & solvers
from generator.Sudoku.Generator import Generator
from solvers.dfs_ac3 import SudokuSolver
from solvers.sudoku import DLXsudoku, checkSudoku
from solvers.genetic_algorithm import Sudoku
from solvers.simulated_annealing import sudoku_solver
from solvers.backtracking import BacktrackingSolver

# other libraries
import numpy as np
import time
import csv
import tracemalloc
import statistics
import matplotlib.pyplot as plt
import logging
from timeout_decorator import timeout, TimeoutError

# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# adapted from [1] sudoku_generator.py
# base_file - .txt file
# difficulty - string == 'easy', 'medium', 'hard', or 'extreme'
def generate_board(difficulty, base_file="base.txt"):
    # setting difficulties and their cutoffs for each solve method
    difficulties = {
        'easy': (35, 0),
        'medium': (81, 5),
        'hard': (81, 10),
        'extreme': (81, 15)
    }

    # get difficulty specs
    difficulty = difficulties[difficulty]

    # constructing generator object from puzzle file (space delimited columns, line delimited rows)
    gen = Generator(base_file)

    # applying 100 random transformations to puzzle
    gen.randomize(100)

    # applying logical reduction with corresponding difficulty cutoff
    gen.reduce_via_logical(difficulty[0])

    # catching zero case
    if difficulty[1] != 0:
        # applying random reduction with corresponding difficulty cutoff
        gen.reduce_via_random(difficulty[1])

    # getting copy after reductions are completed
    final = gen.board.copy()

    # # printing out complete board (solution)
    # print("The initial board before removals was: \r\n\r\n{0}".format(initial))
    #
    # # printing out board after reduction
    # print("The generated board after removals was: \r\n\r\n{0}".format(final))

    return final


# regular backtracking - written by me
def ss_backtracking(puzzle):
    # print("-- Running regular DFS --")

    # convert to a numpy array to work with this algorithm
    n_p = convert_puzzle_to_numpy(puzzle)
    sudoku_solver = BacktrackingSolver(n_p)
    sudoku_solver.solve()

    # return solution -> 2d array
    if sudoku_solver.get_solution().any() == -1:
        return False
    else:
        return True


# from [2]
def ss_backtracking_ac3(puzzle):
    # print("-- Running DFS w/ AC3 --")

    # convert to a numpy array to work with this algorithm
    n_p = convert_puzzle_to_numpy(puzzle)
    sudoku_solver = SudokuSolver(n_p)
    sudoku_solver.solve()

    # return solution -> 2d array
    if sudoku_solver.get_solution().any() == -1:
        return False
    else:
        return True


# from [3]
def ss_dlx(puzzle):
    # print("-- Running DLX --")

    # convert to string to work with DLX implementation
    puzzle_string = convert_puzzle_to_string(puzzle)
    solver = DLXsudoku(puzzle_string)

    for sol in solver.solve():
        if checkSudoku(solver.createSolutionGrid(sol)):
            return True
        else:
            return False

        # return solver.createSolutionGrid(sol)


# from [4]
def ss_genetic(puzzle):
    # print("-- Running Genetic Algorithm --")

    # convert puzzle to numpy array
    n_p = convert_puzzle_to_numpy(puzzle)

    # initialize sudoku object with the puzzle
    s = Sudoku()
    s.load_array(n_p)

    # solve - returns
    solution = s.solve()

    if solution is not None:
        # return solution.values.reshape(9, 9)
        return True
    else:
        return False


# from [5]
def ss_simulated_annealing(puzzle):
    # print("-- Running Simulated Annealing --")

    # convert puzzle -> String -> numpy array to match input format
    puzzle_string = convert_puzzle_to_string(puzzle)
    puzzle_array = np.array([int(val) for val in puzzle_string])

    # solve - receives a numpy array
    numpy_solution = sudoku_solver(input_data=puzzle_array)

    # default simulated annealing solver for testing
    # solution_string = sudoku_solver()

    if numpy_solution is not None:
        return True
    else:
        return False


# INPUT: puzzle from generate_board
# required to work with DFS w/ AC3
def convert_puzzle_to_numpy(puzzle):
    numpy_puzzle = np.zeros((9, 9))
    i = 0
    j = 0

    for cell in puzzle.cells:
        numpy_puzzle[i, j] = cell.value
        i += 1
        if i == 9:
            j += 1
            i = 0

    return numpy_puzzle


# required for DLX
def convert_puzzle_to_string(puzzle):
    puzzle_string = ""
    for cell in puzzle.cells:
        puzzle_string += str(cell.value)

    return puzzle_string


# gets solution, time (sec), and peak memory usage (KB) for each algorithm
@timeout(480, use_signals=True)
def measure_algorithm(puzzle, solving_function):
    try:
        start_time = time.time()
        tracemalloc.start()
        solved = solving_function(puzzle)
        peak_memory_usage = tracemalloc.get_traced_memory()[1] / 1024  # take peak mem and convert to KB
        end_time = time.time()
        solution_time = end_time - start_time
        tracemalloc.stop()
        return solved, solution_time, peak_memory_usage
    except TimeoutError:
        logging.info(f"{solving_function} timed out after 480 seconds")
        return False, -1, -1


def solve_puzzles():
    logging.info("Solving puzzles...")
    # results structure ex. = {'dfs': {'easy': {'time': [x, y], 'memory': [a, b], avg: 0}, 'medium'...}, 'dfs_ac3'...}
    algorithm_metrics = {alg[0]: {diff: {'time': [], 'memory': []}
                                  for diff in list(set(puzzle_difficulties))} for alg in algorithms}

    def store_results(solver, difficulty, time_usage, memory_usage):
        algorithm_metrics[solver][difficulty]['time'].append(round(time_usage, 5))
        algorithm_metrics[solver][difficulty]['memory'].append(memory_usage)

    with open(filename, 'w') as f:

        file_write = csv.writer(f, delimiter='\t')
        file_write.writerow(csv_headers)

        for i, difficulty in enumerate(puzzle_difficulties):
            print(f"Iteration {i} -- Diff = {difficulty}")

            # create puzzle
            puzzle = generate_board(difficulty=difficulty)

            # get solutions and write to file
            for algorithm_name, solving_function in algorithms:
                # get solution, solution time, memory usage
                # if timeout, will be False, 0.0, 0.0
                solved, st, mu = measure_algorithm(puzzle, solving_function)

                # save results to csv
                csv_row = [algorithm_name, str(i), difficulty, solved, str("{:.10f}".format(st)),
                           str(mu)]
                file_write.writerow(csv_row)
                # save results to dictionary
                store_results(algorithm_name, difficulty, st, mu)

            print(f"Iteration {i} complete.")

        f.close()
        logging.info(f"Done solving puzzles. Check {filename} for results!")

    return algorithm_metrics


def create_plots():

    logging.info("Creating plots...")

    # create struct for storing averages
    avg_performance = {alg[0]: {'time': 0, 'memory': 0} for alg in algorithms}

    # compute averages for each algorithm/difficulty
    for algorithm, difficulties in algorithm_results.items():
        all_metrics = {'time': [], 'memory': []}
        plt.figure()
        for difficulty, results in difficulties.items():
            # Only plot if a solution was found
            if results['time'] != -1 and results['memory'] != -1:
                all_metrics['time'] += results['time']
                all_metrics['memory'] += results['memory']
                plt.scatter(results['time'], results['memory'], label=difficulty)

        avg_performance[algorithm]['time'] = statistics.mean(all_metrics['time'])
        avg_performance[algorithm]['memory'] = statistics.mean(all_metrics['memory'])

        plt.xlabel("Time (sec)")
        plt.ylabel("Peak Memory Usage (KB)")
        plt.title(f"Time vs Memory - {algorithm}")
        plt.legend()

        plt.savefig(f'plots/{algorithm}_plot.png')
        plt.close()

    plt.figure()
    for algorithm, results in avg_performance.items():
        plt.scatter(results['time'], results['memory'], label=algorithm)

    plt.xlabel("Time (sec)")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.title(f"Time vs Memory - Average Performance")
    plt.legend()

    plt.savefig(f'plots/averages_plot.png')
    plt.close()

    logging.info("Done creating plots. Check 'plots/' directory!")


if __name__ == '__main__':
    logging.info("Starting solver script")
    start_time = time.time()

    # name of csv log file for all solution results
    filename = 'solution_results.csv'

    # set num puzzles for each difficulty
    num_easy    = 20
    num_medium  = 20
    num_hard    = 20
    num_extreme = 20

    # initialize array of puzzle difficulties
    puzzle_difficulties = []
    puzzle_difficulties += ['easy' for _ in range(num_easy)]
    puzzle_difficulties += ['medium' for _ in range(num_medium)]
    puzzle_difficulties += ['hard' for _ in range(num_hard)]
    puzzle_difficulties += ['extreme' for _ in range(num_extreme)]

    csv_headers = ['alg', 'i', 'diff', 'solved', 'time', 'mem']

    algorithms = [
        ('Depth-first search', ss_backtracking),
        ('Depth-first with AC3', ss_backtracking_ac3),
        ('Dancing Links', ss_dlx),
        ('Genetic Algorithm', ss_genetic),
        ('Simulated Annealing', ss_simulated_annealing)
    ]

    # solve puzzles, save to csv for reference, and get results for time/memory per alg/diff
    algorithm_results = solve_puzzles()

    # create plots and save to 'plots' folder
    # makes plots per algorithm across difficulties, and one plot of avg performance
    create_plots()

    end_time = time.time()
    logging.info(f"Total runtime: {end_time - start_time}")
    logging.info("Script has finished execution")
