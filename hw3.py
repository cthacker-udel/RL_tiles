from __future__ import annotations
from enum import Enum
from typing import Optional
import random
random.seed(1)

"""
[ ] - Implement basic board functionality (placing tiles where they need to be)
[ ] - Write test case suite for automated testing every run through
[ ] - Implement beginning of Q-Value algorithm (early structure)
[ ] - Test final product
"""

# region Test Cases

test_case_1 = ["15 12 8 6 p", "1\tup\n2\tright\n3\tup\n4\tleft\n5\tup\n6\twall-square\n7\tup\n8\tforbid\n9\tup\n10\tup\n11\tup\n12\tgoal\n13\tright\n14\tright\n15\tgoal\n16\tup"]
test_case_2 = ["15 12 8 6 q 11",
               "up\t100.0\nright\t100.0\ndown\t0.89\nleft\t0.89"]
test_case_3 = ["10 8 9 6 p", "1\tright\n2\tright\n3\tup\n4\tup\n5\tdown\n6\twall-square\n7\tright\n8\tgoal\n9\tforbid\n10\tgoal\n11\tleft\n12\tdown\n13\tright\n14\tdown\n15\tdown\n16\tdown\n"]
test_case_4 = ["10 8 9 6 q 2",
               "up\t-0.01\nright\t0.89\ndown\t-0.01\nleft\t-0.1"]
test_case_5 = ["12 7 5 6 p", "1\tright\n2\tright\n3\tup\n4\tup\n5\tforbid\n6\twall-square\n7\tgoal\n8\tup\n9\tup\n10\tup\n11\tup\n12\tgoal\n13\tup\n14\tup\n15\tup\n16\tup\n"]
test_case_6 = ["12 7 5 6 q 3",
               "up\t100.0\nright\t0.89\ndown\t9.9\nleft\t0.89\n"]
test_case_7 = ["13 11 16 5 p", "1\tright\n2\tup\n3\tup\n4\tup\n5\twall-square\n6\tup\n7\tup\n8\tup\n9\tup\n10\tright\n11\tgoal\n12\tleft\n13\tgoal\n14\tleft\n15\tdown\n16\tforbid\n"]
test_case_8 = ["13 11 7 15 p", "1\tup\n2\tup\n3\tright\n4\tup\n5\tup\n6\tup\n7\tforbid\n8\tup\n9\tup\n10\tright\n11\tgoal\n12\tleft\n13\tgoal\n14\tleft\n15\twall-square\n16\tdown\n"]
test_cases = [test_case_1, test_case_2, test_case_3, test_case_4,
              test_case_5, test_case_6, test_case_7, test_case_8]

# endregion

# region Constants

START_IND = 2

# endregion

# region Enums


class OutputFormat(Enum):
    PRINT = "p",
    OPTIMAL_Q = "q"


class TileType(Enum):
    NORMAL = 0,
    GOAL = 1,
    FORBIDDEN = 2,
    WALL = 3,
    START = 4


class AgentAction(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3

# endregion

# region Helper Classes


class BoardTile:
    def __init__(self: BoardTile, x: int, y: int, index: int = -1, tile_type=TileType.NORMAL) -> None:
        self.x = x
        self.y = y
        self.index: int = index
        self.tile_type: TileType = tile_type
        self.q_north = 0  # up
        self.q_east = 0  # right
        self.q_south = 0  # down
        self.q_west = 0  # left
        self.reward = 0  # reward for tile


class ParsedInput:
    def __init__(self: ParsedInput, start_ind: int, goal_1_ind: int, goal_2_ind: int, forbid_ind: int, wall_ind: int, output_format: str, q_ind: Optional[int] = None):
        self.goal_1_ind = goal_1_ind
        self.goal_2_ind = goal_2_ind
        self.forbid_ind = forbid_ind
        self.wall_ind = wall_ind
        self.output_format: OutputFormat = OutputFormat.PRINT if output_format == "p" else OutputFormat.OPTIMAL_Q
        self.q_ind = q_ind
        self.start_ind = start_ind

    def classify_tile_by_ind(self: ParsedInput, ind: int) -> TileType:
        if ind == self.goal_1_ind or ind == self.goal_2_ind:
            return TileType.GOAL
        elif ind == self.forbid_ind:
            return TileType.FORBIDDEN
        elif ind == self.wall_ind:
            return TileType.WALL
        elif ind == self.start_ind:
            return TileType.START
        else:
            return TileType.NORMAL


class Agent:
    def __init__(self: Agent, x=0, y=0, ind=0):
        self.x = x
        self.y = y
        self.ind = ind
        self.moves = [AgentAction.UP, AgentAction.DOWN,
                      AgentAction.LEFT, AgentAction.RIGHT]

    def choose_move(self: Agent, policy: AgentAction, epsilon=1):
        rand_value = random.random()
        curr_policy = 1 - epsilon
        if rand_value <= curr_policy:
            # choose to move randomly
            return policy

        return random.choice(self.moves)

    def clone(self: Agent):
        return Agent(self.x, self.y, self.ind)

# endregion

# region Helpers


def create_basic_board(rows, cols) -> list[list[BoardTile]]:
    board: list[list[BoardTile]] = []
    for i in range(rows):
        sub_row = []
        for j in range(cols):
            sub_row.append(BoardTile(j, i, (i + 1) + j))
        board.append(sub_row)
    return board


def apply_input_to_board(board: list[list[BoardTile]], parsed_input: ParsedInput) -> list[list[BoardTile]]:
    for each_row in board:
        for each_tile in each_row:
            tile_classification = parsed_input.classify_tile_by_ind(
                each_tile.index)
            each_tile.tile_type = tile_classification
            if tile_classification == TileType.GOAL:
                each_tile.reward = 100
            elif tile_classification == TileType.FORBIDDEN:
                each_tile.reward = -100
    return board


def parse_input(inp: str) -> ParsedInput:
    split_inp = inp.split(' ')
    if len(split_inp) == 5:
        [goal_ind1, goal_ind2, forbidden_ind, wall_ind, output_type] = split_inp
        return ParsedInput(START_IND, int(goal_ind1), int(goal_ind2), int(forbidden_ind), int(wall_ind), output_type)
    else:
        [goal_ind1, goal_ind2, forbidden_ind, wall_ind,
            output_type, q_value_ind] = split_inp
        return ParsedInput(START_IND, int(goal_ind1), int(goal_ind2), int(forbidden_ind), int(wall_ind), output_type, int(q_value_ind))


def find_tile_by_ind(board: list[list[BoardTile]], ind: int) -> BoardTile:
    for each_row in board:
        for each_col in each_row:
            if each_col.index == ind:
                return each_col
    raise Exception("Tile index does not exist")


# endregion

# region Main Classes


class BoardSolver:
    def __init__(self: BoardSolver, rows=4, cols=4, living_reward=-0.1, discount=0.1, learning_rate=0.3, max_iter=100_000, board: Optional[list[list[BoardTile]]] = None, agent: Optional[Agent] = None):
        self.board: list[list[BoardTile]] = board if board is not None else create_basic_board(
            rows, cols)
        self.living_reward = living_reward  # r
        self.discount_rate = discount  # gamma
        self.learning_rate = learning_rate  # alpha
        self.iterations = 0
        self.max_iter = max_iter
        if self.agent is None:
            found_start_tile = find_tile_by_ind(self.board, START_IND)
            self.agent = Agent(found_start_tile.x, found_start_tile.y)
        else:
            self.agent = agent

    def learn(self: BoardSolver) -> None:
        pass


# endregion


def main(inp: bool = False):
    if inp:
        parsed_input = parse_input(input())
    else:
        for each_test_case in test_cases:
            parsed_input = parse_input(each_test_case[0])
            board_solver = BoardSolver()
            apply_input_to_board(board_solver.board, parsed_input)


if __name__ == '__main__':
    main()
