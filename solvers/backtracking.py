import numpy as np

# Modified from DFS w/ AC3 by https://github.com/aurbano/sudoku_py by aurbano


class BacktrackingSolver:
    def __init__(self, sudoku):
        # Initialise the related cells cache
        self.related_cells = dict()
        self.calculate_relations()

        self.analyser = BacktrackingSudokuAnalyser(sudoku, self.related_cells)

    def solve(self):
        if not self.analyser.sudoku.is_valid():
            return False

        return self.analyser.solve()

    def get_solution(self):
        return self.analyser.get_solution()

    def calculate_relations(self):
        for r in range(0, 9):
            for c in range(0, 9):
                coords = (c, r)
                self.related_cells[coords] = self.get_related_cells(coords)

    @staticmethod
    def get_related_cells(coords):
        # return a list of all cells that are constrained by this one
        # same row, col and square

        related = list()

        for i in range(0, 9):
            related.append((i, coords[1]))
            related.append((coords[0], i))

        square_x = int((coords[0]) / 3) * 3
        square_y = int((coords[1]) / 3) * 3

        for x in range(0, 3):
            for y in range(0, 3):
                related.append((square_x + x, square_y + y))

        related_set = set(related)
        related_set.remove(coords)

        return related_set


class BacktrackingSudokuAnalyser:
    def __init__(self, sudoku, related_cells, possible_values=None):
        self.related_cells = related_cells

        self.sudoku = Sudoku(sudoku, related_cells, possible_values)
        self.is_valid = True

        if possible_values is None:
            self.is_valid = self.sudoku.is_valid()

    def apply_value(self, coord, value):
        new_possible_values = dict()
        for k, v in self.sudoku.possible_values.items():
            new_possible_values[k] = v.copy()

        new_solver = BacktrackingSudokuAnalyser(
            self.sudoku.grid,
            self.related_cells,
            new_possible_values
        )

        new_solver.is_valid = new_solver.sudoku.set_value(coord, value)

        return new_solver

    def search(self):
        queue = [(k, v) for k, v in self.sudoku.possible_values.items()]
        queue.sort(key=lambda item: len(item[1]))

        while len(queue) > 0:
            coords, poss = queue.pop(0)

            for value in poss:
                # Apply this value and analyse down this branch
                new_solver = self.apply_value(coords, value)

                if new_solver.solve():
                    self.sudoku = new_solver.sudoku
                    return True

            # Completed this branch - no valid values found
            break

        # No valid values at this depth, backtrack out
        return False

    def solve(self):
        if not self.is_valid:
            return False

        if self.sudoku.is_finished() and self.is_valid:
            # Solved!
            return True

        if not self.is_valid:
            return False

        solved = self.search()

        return solved and self.sudoku.is_finished() and self.is_valid

    def get_solution(self):
        if not self.sudoku.is_finished() or not self.is_valid:
            return np.full((9, 9), -1)

        return self.sudoku.grid


class Sudoku:
    def __init__(self, grid, related_cells, possible_values=None):
        self.grid = np.copy(grid)
        self.related_cells = related_cells

        if possible_values is None:
            self.possible_values = self.initial_possible_values()
        else:
            self.possible_values = possible_values

    # initial possible values without any pruning
    def initial_possible_values(self):
        possibilities = dict()

        for r in range(0, 9):
            for c in range(0, 9):
                coords = (c, r)
                cell = self.grid[coords[1]][coords[0]]

                if cell == 0:
                    possibilities[coords] = set(range(1, 10))

        return possibilities

    def is_finished(self):
        return len(self.possible_values) < 1

    def is_valid(self):
        # possible values must be at least one
        for _, values in self.possible_values.items():
            if len(values) == 0:
                return False

        # validate rows/cols
        for i in range(0, 9):
            row = self.grid[i, :]
            column = self.grid[:, i]

            if (not self.valid_line(row) or not self.valid_line(column)):
                return False

        # validate squares
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                square = self.grid[0+r:3+r, 0+c:3+c]
                if not self.valid_line(square.flatten()):
                    return False

        return True

    # set a value in a coordinate, cascading all updates for forward checking
    def set_value(self, coords, value):
        if coords not in self.possible_values or value not in self.possible_values[coords]:
            return False

        self.grid[coords[1]][coords[0]] = value
        del self.possible_values[coords]

        # update all related cells
        for related in self.related_cells[coords]:
            if related not in self.possible_values:
                if self.grid[related[1]][related[0]] == value:
                    return False

                continue

            related_values = self.possible_values[related]

            related_values.discard(value)

            if len(related_values) == 0:
                return False

            if len(related_values) == 1:
                (last_value,) = related_values
                if not self.set_value(related, last_value):
                    return False

        return True

    def get_unfinished_possible_values(self):
        return set(k for k, v in self.possible_values.items())

    def get_unfinished_cells(self, coord_set):
        return [coord for coord in coord_set if coord in self.possible_values]

    @staticmethod
    def valid_line(line):
        return len(line) == 9 and sum(line) == sum(set(line))
