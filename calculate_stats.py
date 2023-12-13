""""
MASON MCFARLAND
CSCI 4511W
Final project

'Solving Sudoku'

This file is used to calculate some additional statistics and create additional plots
that I thought would be helpful for my paper
"""


import pandas as pd
import matplotlib.pyplot as plt


# find how many puzzles were left unsolved by each algorithm
def count_unsolved_puzzles(file_path):
    # read CSV file
    df = pd.read_csv(file_path, delimiter='\t')

    # group by algorithm & count unsolved puzzles
    unsolved_counts = df.groupby('alg')['solved'].apply(lambda x: (x == False).sum()).reset_index()

    return unsolved_counts


def read_and_plot(filename):
    # read CSV file
    df = pd.read_csv(filename, delimiter='\t')

    # create a scatter plot by alg
    algorithms = df['alg'].unique()

    for algorithm in algorithms:
        algorithm_data = df[df['alg'] == algorithm]

        plt.scatter(
            algorithm_data['time'],
            algorithm_data['mem'],
            label=algorithm,
            alpha=0.7
        )

    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory (MB)')
    plt.title('Time vs. Memory - All Algorithms')

    plt.legend()

    plt.savefig(f'plots/all_algs_plot.png')
    plt.close()


def analyze_results(file_path):
    # read csv
    df = pd.read_csv(file_path, delimiter='\t')

    # calculate overall solved rate, avg time, avg memory for each alg
    results_summary_alg = df.groupby('alg').agg(
        overall_solved_rate=('solved', 'mean'),
        overall_avg_time=('time', 'mean'),
        overall_avg_memory=('mem', 'mean')
    ).reset_index()

    # calculate solved rate, avg time, and avg memory per alg/diff
    results_summary_diff = df.groupby(['alg', 'diff']).agg(
        solved_rate=('solved', 'mean'),
        avg_time=('time', 'mean'),
        avg_memory=('mem', 'mean')
    ).reset_index()

    return results_summary_alg, results_summary_diff


def extract_max_memory(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter='\t')

    # Extract maximum memory consumed for each algorithm
    max_memory_per_algorithm = df.groupby('alg')['mem'].max().reset_index()

    return max_memory_per_algorithm


if __name__ == '__main__':

    file = "solution_results.csv"

    # create an additional plot
    read_and_plot(file)

    results_summary_alg, results_summary_diff = analyze_results(file)

    print("Average results per algorithm:")
    print(results_summary_alg)

    print("Average results per algorithm per difficulty:")
    print(results_summary_diff)

    print("Max memory for each algorithm:")
    print(extract_max_memory(file))