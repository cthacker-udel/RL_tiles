from __future__ import annotations
from enum import Enum
from typing import Optional, Self
import time
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
        self.reward_north = -0.1
        self.reward_east = -0.1
        self.reward_south = -0.1
        self.reward_west = -0.1

    def get_reward(self: BoardTile, action: AgentAction):
        if action == AgentAction.DOWN:
            return self.reward_south
        elif action == AgentAction.UP:
            return self.reward_north
        elif action == AgentAction.RIGHT:
            return self.reward_east

        # left
        return self.reward_west

    def get_q(self: BoardTile, action: AgentAction):
        if action == AgentAction.DOWN:
            return self.q_south
        elif action == AgentAction.UP:
            return self.q_north
        elif action == AgentAction.RIGHT:
            return self.q_east

        # moving left
        return self.q_west

    def clone(self: BoardTile) -> BoardTile:
        cloned_tile = BoardTile(self.x, self.y, self.index, self.tile_type)
        cloned_tile.q_north = self.q_north
        cloned_tile.q_south = self.q_south
        cloned_tile.q_east = self.q_east
        cloned_tile.q_west = self.q_west
        cloned_tile.reward_north = self.reward_north
        cloned_tile.reward_south = self.reward_south
        cloned_tile.reward_east = self.reward_east
        cloned_tile.reward_west = self.reward_west

        return cloned_tile


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

    def simulate_input(self: Agent, action: AgentAction) -> Agent:
        # returns the row
        if action == AgentAction.UP:
            return Agent(self.x, self.y + 1)
        elif action == AgentAction.DOWN:
            return Agent(self.x, self.y - 1)
        elif action == AgentAction.LEFT:
            return Agent(self.x - 1, self.y)
        # RIGHT
        return Agent(self.x + 1, self.y)

# endregion

# region Helper Functions


def create_basic_board(rows: int, cols: int) -> list[list[BoardTile]]:
    board: list[list[BoardTile]] = []
    ind = 1
    for i in range(rows):
        sub_row = []
        for j in range(cols):
            sub_row.append(BoardTile(j, i, ind))
            ind += 1
        board.append(sub_row)
    return board


def stringify_agent_action(action: AgentAction) -> str:
    return 'EAST' if action == AgentAction.RIGHT else "WEST" if action == AgentAction.LEFT else "NORTH" if action == AgentAction.UP else "SOUTH"


def apply_input_to_board(board: list[list[BoardTile]], parsed_input: ParsedInput) -> list[list[BoardTile]]:
    for each_row in board:
        for each_tile in each_row:
            tile_classification = parsed_input.classify_tile_by_ind(
                each_tile.index)
            each_tile.tile_type = tile_classification
            if tile_classification == TileType.GOAL:
                each_tile.reward_west, each_tile.reward_east, each_tile.reward_north, each_tile.reward_south = 100, 100, 100, 100
            elif tile_classification == TileType.FORBIDDEN:
                each_tile.reward_west, each_tile.reward_east, each_tile.reward_north, each_tile.reward_south = - \
                    100, -100, -100, -100
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
    raise ValueError("Tile index does not exist")


def clone_board(board: list[list[BoardTile]]) -> list[list[BoardTile]]:
    cloned_board: list[list[BoardTile]] = []
    for each_row in board:
        sub_row = []
        for each_cell in each_row:
            sub_row.append(each_cell.clone())
        cloned_board.append(sub_row)
    return cloned_board


def print_board(board: list[list[BoardTile]], agent_x: int, agent_y: int) -> None:
    q_tops = []
    q_middles = []
    q_bottoms = []
    for each_row in board:
        row_tops = [f' {x.q_north} ' for x in each_row]
        q_tops.append(row_tops)
        row_middles = [f'{x.q_west} {x.q_east}' for x in each_row]
        q_middles.append(row_middles)
        row_bottoms = [f' {x.q_south} ' for x in each_row]
        q_bottoms.append(row_bottoms)
    q_prints = []
    for i in range(len(q_tops)):
        q_prints.append(f'{"\t".join(q_tops[i])}\n{
                        "\t".join(q_middles[i])}\n{"\t".join(q_bottoms[i])}')
    q_output = '\n\n'.join(q_prints)
    print(q_output)

    r_tops = []
    r_middles = []
    r_bottoms = []
    for each_row in board:
        row_tops = [f'\t  {x.reward_north}\t' for x in each_row]
        r_tops.append(row_tops)
        row_middles = [f'\t{x.reward_west} {
            x.reward_east}\t' for x in each_row]
        r_middles.append(row_middles)
        row_bottoms = [f'\t  {x.reward_south}\t' for x in each_row]
        r_bottoms.append(row_bottoms)
    r_prints = []
    for i in range(len(r_tops)):
        r_prints.append(f'{"\t".join(r_tops[i])}\n{
                        "".join(r_middles[i])}\n{"\t".join(r_bottoms[i])}')
    r_output = '\n\n'.join(r_prints)
    print("####################################")
    print("################################### REWARDS ###################################")
    print(r_output)
    print("###############################################################################")

    print("################## BOARD ###################")
    board_output = []
    y = 0
    for each_row in board:
        sub_row = []
        for i in range(len(each_row)):
            if y == agent_y and agent_x == i:
                sub_row.append('[A]\t')
            else:
                sub_row.append('[ ]\t')
        board_output.append(sub_row)
        y += 1
    board_print = []
    for each_row in board_output:
        board_print.append('\t'.join(each_row))
    b_output = '\n'.join(board_print)
    print(b_output)
    print("######################################################")


# endregion

# region Main Classes


class State:
    def __init__(self: State, rows=4, cols=4, living_reward=-0.1, discount=0.1, learning_rate=0.3, max_iter=100_000, board: Optional[list[list[BoardTile]]] = None, agent: Optional[Agent] = None, iterations=0):
        self.board: list[list[BoardTile]] = board if board is not None else create_basic_board(
            rows, cols)
        self.living_reward = living_reward  # r
        self.discount_rate = discount  # gamma
        self.learning_rate = learning_rate  # alpha
        self.iterations = iterations
        self.max_iter = max_iter
        if agent is None:
            found_start_tile = find_tile_by_ind(self.board, START_IND)
            self.agent = Agent(found_start_tile.x, found_start_tile.y)
        else:
            self.agent = agent
        self.rows = rows
        self.cols = cols
        self.debug = True

    def q_value(self: State, action: AgentAction) -> float:
        # Q(s,a) is 0 initially

        if self.debug:
            print('-----------------------------------------------------')
            print('############### Q_VALUES ###############')
            print_board(self.board, self.agent.x, self.agent.y)
            print('-----------------------------------------------------')
            time.sleep(2)

        if self.agent.x < 0 or self.agent.x >= self.cols:
            return 0.0
        elif self.agent.y < 0 or self.agent.y >= self.rows:
            return 0.0

        found_tile = find_tile_by_ind(
            self.board, self.board[self.agent.y][self.agent.x].index)

        if found_tile.tile_type == TileType.FORBIDDEN:
            return found_tile.reward_east
        elif found_tile.tile_type == TileType.GOAL:
            return found_tile.reward_east

        left = (1 - self.learning_rate) * found_tile.get_q(action)

        sprime = self.clone(self.iterations + 1).move_agent(action)
        right = self.learning_rate * \
            (found_tile.get_reward(action) +
             self.discount_rate * max(sprime.q_value(x) for x in [AgentAction.UP, AgentAction.DOWN, AgentAction.LEFT, AgentAction.RIGHT]))
        return left + right

    def clone(self: State, new_iter=0) -> State:
        return State(self.rows, self.cols, self.living_reward, self.discount_rate, self.learning_rate, self.max_iter, clone_board(self.board), self.agent.clone(), new_iter)

    def move_agent(self: State, action: AgentAction) -> State:
        self.agent = self.agent.simulate_input(action)
        return self

    def learn(self: State) -> None:
        self.q_value(AgentAction.UP)
        self.q_value(AgentAction.DOWN)
        self.q_value(AgentAction.RIGHT)
        self.q_value(AgentAction.LEFT)


# endregion


def main(inp: bool = False):
    if inp:
        parsed_input = parse_input(input())
    else:
        for each_test_case in test_cases:
            parsed_input = parse_input(each_test_case[0])
            board_solver = State()
            apply_input_to_board(board_solver.board, parsed_input)
            board_solver.learn()


if __name__ == '__main__':
    main()
