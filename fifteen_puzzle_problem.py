# flake8: noqa

import copy

import numpy as np

from genetic_algorithm import GeneticAlgorithm


class FifteenPuzzleExample(GeneticAlgorithm):
    def __init__(self, board=None, solution_size=80, heuristic="a", display_documentation=True):
        self.board = board
        self.board_size = 4
        self.solution_size = solution_size
        self.possible_moves = ["u", "d", "l", "r"]
        self.example_board = [[14, 0, 11, 10], [13, 3, 7, 5], [12, 15, 2, 9], [4, 1, 8, 6]]
        self.target_solution = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

        if heuristic == "a":
            self.target_score = 0
        if heuristic == "b":
            self.target_score = 15

        if board == None:
            self.board = self.example_board

        super(FifteenPuzzleExample, self).__init__(
            self.possible_moves, self.solution_size, self.target_score, display_documentation=display_documentation
        )

    def move_piece(self, move=None, board=None, modify=False, display_board=False):
        if move == None:
            move = np.random.choice(self.possible_moves, 1, replace=False)[0]
        assert move in self.possible_moves, "Invalid move!"

        if board == None:
            board = self.board

        board = copy.deepcopy(board)

        r = -1
        c = -1

        for i in range(len(self.target_solution)):
            for j in range(len(self.target_solution)):
                if board[i][j] == 0:
                    r, c = i, j

        if move == "u":
            new_r = r
            new_c = c
            if r - 1 >= 0:
                new_r = r - 1

        elif move == "d":
            new_r = r
            new_c = c
            if r + 1 < len(self.target_solution):
                new_r = r + 1

        elif move == "l":
            new_r = r
            new_c = c
            if c - 1 >= 0:
                new_c = c - 1

        else:
            new_r = r
            new_c = c
            if c + 1 < len(self.target_solution):
                new_c = c + 1

        board[new_r][new_c], board[r][c] = board[r][c], board[new_r][new_c]

        if display_board:
            self.display(board=board)

        if modify:
            self.board = board

        return board

    def manhattan_distance_heuristic(self, board=None):
        if board == None:
            board = self.board

        initial_config = list(np.array(board).flatten())
        man_distance = 0

        for i, item in enumerate(initial_config):
            prev_row, prev_col = int(i / self.board_size), i % self.board_size
            goal_row, goal_col = int(item / self.board_size), item % self.board_size
            man_distance += abs(prev_row - goal_row) + abs(prev_col - goal_col)
        return -man_distance

    def correct_position_heuristic(self, board=None):
        if board == None:
            board = self.board

        fitness = 0
        for r in range(len(self.target_solution)):
            for c in range(len(self.target_solution)):
                if board[r][c] == self.target_solution[r][c]:
                    fitness += 1
        return fitness

    def compute_fitness(self, solution=None, board=None, heuristic="a", modify=False, display_board=False):
        if board == None:
            board = self.board

        board = copy.deepcopy(board)
        if solution != None:
            for move in solution:
                board = self.move_piece(move, board)

        if heuristic == "a":
            fitness = self.manhattan_distance_heuristic(board)

        if heuristic == "b":
            fitness = self.correct_position_heuristic(board)

        if display_board:
            self.display(board=board)

        if modify:
            self.board = board

        return fitness

    def display(self, solution=None, board=None):
        if board == None:
            board = self.board

        if solution != None:
            for move in solution:
                board = self.move_piece(move, board)

        for r in range(len(self.target_solution)):
            for c in range(len(self.target_solution)):
                if board[r][c] < 10:
                    print(board[r][c], end="    ")
                else:
                    print(board[r][c], end="   ")
            print("\n")
        print("Fitness: {}\n\n".format(self.compute_fitness(board=board)))
