from __future__ import annotations
from enum import Enum
from typing import Optional, Callable
import time
import random
random.seed(1)

"""
[ ] - Implement basic board functionality (placing tiles where they need to be)
[X] - Write test case suite for automated testing every run through
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

# region Constants


START_IND = 2
ITERATION_COUNT = 0
ACTIONS = [AgentAction.UP, AgentAction.DOWN, AgentAction.RIGHT, AgentAction.LEFT]
CLOCKWISE_POLICY_ORDER = [AgentAction.UP, AgentAction.DOWN, AgentAction.RIGHT, AgentAction.LEFT]

# endregion

# region Helper Classes


class BoardTile:
    def __init__(self: BoardTile, x: int, y: int, index: int = -1, tile_type=TileType.NORMAL) -> None:
        self.x = x
        self.y = y
        self.index: int = index
        self.tile_type: TileType = tile_type
        self.q_north, self.q_east, self.q_south, self.q_west = 0, 0, 0, 0
        self.reward_north, self.reward_east, self.reward_south, self.reward_west = -0.1, -0.1, -0.1, -0.1
        self.is_terminal = False

    def get_reward(self: BoardTile, action: AgentAction):
        if action == AgentAction.DOWN:
            return self.reward_south
        if action == AgentAction.UP:
            return self.reward_north
        if action == AgentAction.RIGHT:
            return self.reward_east

        # AgentAction.LEFT
        return self.reward_west

    def get_q(self: BoardTile, action: AgentAction):
        if action == AgentAction.DOWN:  # in terms of the board being flipped, the directions are flipped
            return self.q_south
        elif action == AgentAction.UP:  # in terms of the board being flipped, the directions are flipped
            return self.q_north
        elif action == AgentAction.RIGHT:
            return self.q_east

        # AgentAction.LEFT
        return self.q_west

    def set_q(self: BoardTile, action: AgentAction, value: float) -> None:
        if action == AgentAction.DOWN:  # in terms of the board being flipped, the directions are flipped
            self.q_south = value
        elif action == AgentAction.UP:  # in terms of the board being flipped, the directions are flipped
            self.q_north = value
        elif action == AgentAction.LEFT:
            self.q_west = value
        elif action == AgentAction.RIGHT:
            self.q_east = value

    def get_all_q(self: BoardTile) -> list[float]:
        return [self.q_east, self.q_west, self.q_north, self.q_south]

    def set_all_q(self: BoardTile, value: float) -> None:
        self.q_north, self.q_south, self.q_east, self.q_west = value, value, value, value


class ParsedInput:
    def __init__(self: ParsedInput, start_ind: int, goal_1_ind: int, goal_2_ind: int, forbid_ind: int, wall_ind: int, output_format: str, q_ind: Optional[int] = None):
        self.goal_1_ind = goal_1_ind
        self.goal_2_ind = goal_2_ind
        self.forbid_ind = forbid_ind
        self.wall_ind = wall_ind
        self.output_format: OutputFormat = OutputFormat.PRINT if output_format == "p" else OutputFormat.OPTIMAL_Q
        self.q_ind = q_ind
        self.start_ind = start_ind
        self.direction_to_string: Callable[[
            AgentAction], str] = lambda x: 'up' if x == AgentAction.UP else 'down' if x == AgentAction.DOWN else 'right' if x == AgentAction.RIGHT else 'left'
        self.tile_type_to_string: Callable[[
            TileType], str] = lambda x: 'wall-square' if x == TileType.WALL else 'forbid' if x == TileType.FORBIDDEN else 'goal'

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

    def print_policies(self: ParsedInput, board: list[list[BoardTile]]) -> None:
        board_tiles: dict[int, str] = {}
        for each_row in board:
            for each_tile in each_row:
                if each_tile.tile_type in (TileType.FORBIDDEN, TileType.GOAL, TileType.WALL):
                    board_tiles[each_tile.index] = self.tile_type_to_string(each_tile.tile_type)
                else:
                    max_policy = [round(each_tile.get_q(AgentAction.UP), 2), AgentAction.UP]
                    for each_direction in CLOCKWISE_POLICY_ORDER:
                        curr_q = round(each_tile.get_q(each_direction), 2)
                        if curr_q > max_policy[0]:
                            max_policy[0] = curr_q
                            max_policy[1] = each_direction
                    board_tiles[each_tile.index] = self.direction_to_string(max_policy[1])
        joined_strings = []
        for each_key in board_tiles:
            joined_strings.append(f'{each_key}\t{board_tiles[each_key]}')
        print('\n'.join(joined_strings))

    def print_q_values(self: ParsedInput, tile: BoardTile) -> None:
        directions = []
        for each_direction in CLOCKWISE_POLICY_ORDER:
            directions.append(f'{self.direction_to_string(each_direction)}\t{round(tile.get_q(each_direction), 2)}')
        print('\n'.join(directions))


class Agent:
    def __init__(self: Agent, x=0, y=0, ind=0):
        self.x = x
        self.y = y
        self.ind = ind

# endregion

# region Helper Functions


def create_basic_board(rows: int, cols: int) -> list[list[BoardTile]]:
    board: list[list[BoardTile]] = []
    starting_ranges = [13, 9, 5, 1]
    for i in range(rows):
        ind = starting_ranges[i]
        sub_row = []
        for j in range(cols):
            sub_row.append(BoardTile(j, i, ind))
            ind += 1
        board.append(sub_row)
    return board


def apply_input_to_board(board: list[list[BoardTile]], parsed_input: ParsedInput) -> list[list[BoardTile]]:
    for each_row in board:
        for each_tile in each_row:
            tile_classification = parsed_input.classify_tile_by_ind(
                each_tile.index)
            each_tile.tile_type = tile_classification
            if tile_classification == TileType.GOAL:
                each_tile.reward_west, each_tile.reward_east, each_tile.reward_north, each_tile.reward_south = 100, 100, 100, 100
                each_tile.is_terminal = True
            elif tile_classification == TileType.FORBIDDEN:
                each_tile.reward_west, each_tile.reward_east, each_tile.reward_north, each_tile.reward_south = - \
                    100, -100, -100, -100
                each_tile.is_terminal = True
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


def simulate_move_on_board(board: list[list[BoardTile]], action: AgentAction, curr_x: int, curr_y: int) -> BoardTile:
    if action == AgentAction.DOWN:
        new_y = curr_y + 1

        future_piece = board[new_y][curr_x]

        if future_piece.tile_type == TileType.WALL:
            raise Exception("Hit wall")

        return future_piece

    if action == AgentAction.UP:
        new_y = curr_y - 1

        future_piece = board[curr_y - 1][curr_x]  # in terms of the board being flipped, the directions are flipped

        # IS WALL OR OUT OF BOUNDS
        if future_piece.tile_type == TileType.WALL or new_y == -1:
            raise Exception("Hit wall or Out of Bounds")

        return future_piece

    if action == AgentAction.LEFT:
        new_x = curr_x - 1

        future_piece = board[curr_y][curr_x - 1]

        # IS WALL OR OUT OF BOUNDS
        if future_piece.tile_type == TileType.WALL or new_x == -1:
            raise Exception("Hit wall or Out of Bounds")

        return future_piece

    new_x = curr_x + 1

    future_piece = board[curr_y][new_x]

    # IS WALL
    if future_piece.tile_type == TileType.WALL:
        raise Exception("Hit wall")

    # AgentAction.RIGHT
    return future_piece


def chooses_random(epsilon: float | int = 0.5) -> bool:
    rand_value = random.uniform(0, 1)
    curr_policy = 1 - epsilon
    if rand_value <= curr_policy:
        return False

    return True


f_q: Callable[[float], float] = lambda x: round(x, 2)


def print_q_values(board: list[list[BoardTile]]) -> None:
    q_tops = []
    q_middles = []
    q_bottoms = []
    for each_row in board:
        row_tops = [f' {round(x.q_north, 2)} '.rjust(20) for x in each_row]
        q_tops.append(row_tops)
        row_middles = [f'{round(x.q_west, 2)} {round(x.q_east, 2)}'.rjust(20) for x in each_row]
        q_middles.append(row_middles)
        row_bottoms = [f' {round(x.q_south, 2)} '.rjust(20) for x in each_row]
        q_bottoms.append(row_bottoms)
    q_prints = []
    for i in range(len(q_tops)):
        q_prints.append(f'{"".join(q_tops[i]).ljust(20)}\n{"".join(
            q_middles[i]).ljust(20)}\n{"".join(q_bottoms[i]).ljust(20)}')
    q_output = '\n\n'.join(q_prints)
    print(q_output)


def print_board(board: list[list[BoardTile]], agent_x: int, agent_y: int) -> None:
    print_q_values(board)

    print(f"################## BOARD [Agent({agent_x, agent_y})] ###################")
    board_output = []
    y = 0
    for each_row in board:
        sub_row = []
        for ind, each_cell in enumerate(each_row):
            if y == agent_y and agent_x == ind:
                sub_row.append('[A]\t')
            elif each_cell.tile_type == TileType.FORBIDDEN:
                sub_row.append('[X]\t')
            elif each_cell.tile_type == TileType.WALL:
                sub_row.append('[W]\t')
            elif each_cell.tile_type == TileType.GOAL:
                sub_row.append('[G]\t')
            elif each_cell.tile_type == TileType.START:
                sub_row.append('[S]\t')
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
    def __init__(self: State, rows=4, cols=4, living_reward=-0.1, discount=0.1, learning_rate=0.3, max_iter=100_000, board: Optional[list[list[BoardTile]]] = None, agent: Optional[Agent] = None, iterations=0, epsilon=0.5):
        self.board: list[list[BoardTile]] = board if board is not None else create_basic_board(
            rows, cols)
        self.living_reward = living_reward  # r
        self.discount_rate = discount  # gamma
        self.learning_rate = learning_rate  # alpha
        self.iterations = iterations
        self.max_iter = max_iter
        if agent is None:
            found_start_tile = find_tile_by_ind(self.board, START_IND)
            self.agent = Agent(found_start_tile.x, found_start_tile.y, found_start_tile.index)
        else:
            self.agent = agent
        self.rows = rows
        self.cols = cols
        self.epsilon = epsilon
        self.debug = True

    def q_value(self: State) -> None:
        # Q(s,a) is 0 initially

        iteration_count = 0
        while self.epsilon != 0:
            current_tile = find_tile_by_ind(self.board, START_IND)
            self.agent.x = current_tile.x
            self.agent.y = current_tile.y
            self.agent.ind = current_tile.index
            while not current_tile.is_terminal:
                if iteration_count > self.max_iter:
                    self.epsilon = 0
                    self.debug = True

                if self.debug:
                    print(f'----------------- ITERATION {self.iterations} --------------------------')
                    print('############### Q_VALUES ###############')
                    print_board(self.board, self.agent.x, self.agent.y)
                    print('-----------------------------------------------------')

                q_and_action = [current_tile.get_q(AgentAction.UP), AgentAction.UP]
                for each_action in [AgentAction.RIGHT, AgentAction.DOWN, AgentAction.LEFT]:
                    old_count = q_and_action[0]
                    cur_q = current_tile.get_q(each_action)
                    if cur_q > old_count:
                        q_and_action[1] = each_action
                        q_and_action[0] = cur_q

                random_choice = chooses_random(self.epsilon)

                if random_choice:
                    random_action = random.choice(ACTIONS)
                    q_and_action = [current_tile.get_q(random_action), random_action]

                try:
                    new_q_value = current_tile.get_q(q_and_action[1])
                    future_tile = simulate_move_on_board(self.board, q_and_action[1], self.agent.x, self.agent.y)

                    # discount_value = self.discount_rate * max(future_tile.get_all_q())
                    # right_value = self.learning_rate * (current_tile.get_reward(q_and_action[1]) + discount_value)
                    # left_value = (1 - self.learning_rate) * new_q_value
                    # new_q_value = left_value + right_value

                    discount_value = current_tile.get_reward(q_and_action[1]) + ((self.discount_rate *
                                                                                 max(future_tile.get_all_q())) - current_tile.get_q(q_and_action[1]))
                    new_q_value += (self.learning_rate * discount_value)

                    current_tile.set_q(q_and_action[1], new_q_value)

                    current_tile = future_tile
                    self.agent.x = future_tile.x
                    self.agent.y = future_tile.y
                    self.agent.ind = future_tile.index
                except Exception as ex:
                    # left_value = (1 - self.learning_rate) * current_tile.get_q(q_and_action[1])
                    # right_value = self.learning_rate * self.living_reward
                    # new_q_value = left_value + right_value

                    new_q_value = current_tile.get_q(q_and_action[1])

                    discount_value = (current_tile.get_reward(q_and_action[1]) - current_tile.get_q(q_and_action[1]))
                    new_q_value += (self.learning_rate * discount_value)

                    current_tile.set_q(q_and_action[1], new_q_value)

                if current_tile.is_terminal:
                    # left_part = (1 - self.learning_rate) * current_tile.get_q(AgentAction.UP)
                    # right_part = self.learning_rate * current_tile.get_reward(AgentAction.UP)
                    # new_q_value = left_part + right_part

                    new_q_value = current_tile.get_q(AgentAction.UP)

                    discount_value = (current_tile.get_reward(AgentAction.UP) - current_tile.get_q(AgentAction.UP))
                    new_q_value += (self.learning_rate * discount_value)

                    current_tile.set_all_q(new_q_value)
                iteration_count += 1
                print(iteration_count)

    def learn(self: State) -> None:
        self.q_value()


# endregion


def main(inp: bool = False):
    if inp:
        global ITERATION_COUNT
        ITERATION_COUNT = 0
        parsed_input = parse_input(input())
    else:
        for each_test_case in test_cases:
            parsed_input = parse_input(each_test_case[0])
            board_solver = State()
            apply_input_to_board(board_solver.board, parsed_input)
            board_solver.learn()
            if parsed_input.output_format == OutputFormat.PRINT:
                parsed_input.print_policies(board_solver.board)
            elif parsed_input.q_ind is not None and parsed_input.output_format == OutputFormat.OPTIMAL_Q:
                parsed_input.print_q_values(find_tile_by_ind(board_solver.board, parsed_input.q_ind))


if __name__ == '__main__':
    main()
