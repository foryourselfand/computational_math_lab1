import copy
import decimal
from math import isclose
from random import randint
from random import uniform
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type


EPSILON = 1e-08
decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # round(0.5) -> 1.0


def __is_close(first: float, second: float) -> bool:
    return isclose(first, second, rel_tol=EPSILON, abs_tol=EPSILON)


def __round(number: float, precision: Optional[int] = None) -> float:
    if precision is not None:
        number_decimal: decimal.Decimal = decimal.Decimal(str(number))
        number_rounded = round(number_decimal, precision)
        return float(number_rounded)
    return number


def __check_zero_division(matrix: List[List[float]], index: int) -> None:
    if __is_close(matrix[index][index], 0):
        if all(__is_close(matrix[index][j], 0) for j in range(len(matrix[index]))):
            raise Exception('Infinite many solutions')
        else:
            raise Exception('No solutions')


def get_solutions(matrix: List[List[float]], precision: Optional[int]) -> List[float]:
    a: List[List[float]] = copy.deepcopy(matrix)
    n: int = len(a)
    print_matrix_and_precision_and_iteration(a, precision)
    for k in range(1, n):
        for j in range(k, n):
            __check_zero_division(a, k - 1)
            m: float = __round(a[j][k - 1] / a[k - 1][k - 1], precision)
            for i in range(n + 1):
                a[j][i] = __round(a[j][i] - m * a[k - 1][i], precision)
        print_matrix_and_precision_and_iteration(a, precision, iteration=k)
    
    x: List[float] = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        __check_zero_division(a, i)
        x[i] = __round(a[i][-1] / a[i][i], precision)
        for c in range(n - 1, i, -1):
            x[i] = __round(x[i] - (a[i][c] * x[c] / a[i][i]), precision)
    return x


def get_residuals(matrix: List[List[float]], solution: List[float], precision: Optional[int]) -> List[float]:
    matrix_right: List[float] = [row[-1] for row in matrix]
    temp: List[float] = [0 for _ in range(len(matrix))]
    
    residuals: List[float] = [0 for _ in range(len(matrix))]
    for i in range(len(matrix[0]) - 1):
        temp[i] = 0
        for j in range(len(matrix[0]) - 1):
            temp[i] = __round(temp[i] + solution[j] * matrix[i][j], precision)
        residuals[i] = __round(temp[i] - matrix_right[i], precision)
    
    return residuals


def determinant_recursive(matrix: List[List[float]], determinant: float = 0) -> float:
    indices: List[int] = list(range(len(matrix)))
    
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    
    for focus_column in indices:
        submatrix: List[List[float]] = copy.deepcopy(matrix)
        submatrix = submatrix[1:]
        
        for i in range(len(submatrix)):
            submatrix[i] = submatrix[i][0:focus_column] + submatrix[i][focus_column + 1:]
        
        sign: int = (-1) ** (focus_column % 2)
        
        subdeterminant: float = determinant_recursive(submatrix)
        determinant += sign * matrix[0][focus_column] * subdeterminant
    
    return determinant


def solve(matrix_and_precision_getter: Callable[[], Tuple[List[List[float]], Optional[int]]]) -> None:
    matrix, precision = matrix_and_precision_getter()
    
    matrix_short: List[List[float]] = [matrix[i][:len(matrix[i]) - 1] for i in range(len(matrix))]
    
    determinant: float = determinant_recursive(matrix_short)
    determinant_formatted: str = __get_number_formatted(determinant, precision)
    
    if __is_close(determinant, 0):
        print('determinant is zero, so either there are no solutions, or there are infinitely many solutions')
    
    solutions: List[float] = get_solutions(matrix, precision)
    print(f'determinant: {determinant_formatted}\n')
    __print_resulted_values(values=solutions, text_long='solutions', text_short='x', precision=precision)
    
    residuals: List[float] = get_residuals(matrix, solutions, precision)
    __print_resulted_values(values=residuals, text_long='residuals', text_short='r', precision=precision)


def __print_resulted_values(values: List[float], text_long: str, text_short: str, precision: Optional[int] = None) -> None:
    print(f'{text_long}: ')
    for index, value in enumerate(values, 1):
        value_formatted: str = __get_number_formatted(value, precision)
        print(f'{text_short}[{index}] = {value_formatted}')
    print()


def solve_from_random() -> None:
    solve(matrix_and_precision_getter=get_matrix_and_precision_from_random)


def solve_from_file() -> None:
    solve(matrix_and_precision_getter=get_matrix_and_precision_from_file)


def solve_from_console() -> None:
    solve(matrix_and_precision_getter=get_matrix_and_precision)


def get_matrix_and_precision_from_random() -> Tuple[List[List[float]], Optional[int]]:
    matrix_size_input: str = input('Enter matrix size (default = 5): ').strip()
    matrix_size: int = int(matrix_size_input) if matrix_size_input else 5
    
    random_generator: Callable = randint
    number_converter: Callable = int
    random_generator_input = input('Enter values type (int/float) (default = int): ').strip()
    if random_generator_input == 'float':
        random_generator = uniform
        number_converter = float
    
    solutions_range: Tuple[float, float] = __get_range(name='solutions', bound_left=-10, bound_right=10, number_converter=number_converter)
    coefficients_range: Tuple[float, float] = __get_range(name='coefficient', bound_left=-100, bound_right=100, number_converter=number_converter)
    
    solutions: List[float] = [random_generator(solutions_range[0], solutions_range[1]) for _ in range(matrix_size)]
    
    matrix: List[List[float]] = []
    for _ in range(matrix_size):
        coefficients: List[float] = [random_generator(coefficients_range[0], coefficients_range[1]) for _ in range(matrix_size)]
        multipliers: List[float] = [solution * coefficient for solution, coefficient in zip(solutions, coefficients)]
        after_equal_sign: float = sum(multipliers)
        rows: List[float] = coefficients + [after_equal_sign]
        matrix.append(rows)
    
    return matrix, None


def __get_range(name: str, bound_left: float, bound_right: float, number_converter: Callable) -> Tuple[float, float]:
    range_input: str = input(f'Enter {name} range (default = {bound_left} {bound_right}): ')
    range_default: Tuple[float, float] = bound_left, bound_right
    if range_input:
        range_splitted: List[str] = range_input.split()
        try:
            range_default = number_converter(range_splitted[0]), number_converter(range_splitted[1])
        except ValueError:
            raise Exception('Expected int')
    return range_default


def get_matrix_and_precision_from_file() -> Tuple[List[List[float]], Optional[int]]:
    file_name: str = input('Enter file name: ')
    with open(file_name, 'r') as file_input:
        lines: List[str] = file_input.read().splitlines()
        return get_matrix_and_precision(lines)


def get_matrix_and_precision(lines: Optional[List[str]] = None) -> Tuple[List[List[float]], Optional[int]]:
    print('Enter optional floating point precision and then enter matrix: ')
    
    matrix: List[List[float]] = []
    
    line_first: str = __get_line_from_lines_or_input(lines, 0)
    precision: Optional[int] = None
    if __try_convert(line_first, int):
        precision = int(line_first)
        if precision <= 0:
            raise Exception('precision must be non-negative')
        line_first = __get_line_from_lines_or_input(lines, 1)
    
    line_first_converted: List[float] = __convert_line_and_add_to_matrix(line_first, matrix, precision)
    line_first_len: int = len(line_first_converted)

    if precision is None and (line_first_len <= 2 or (lines and len(lines) <= 1)):
        raise Exception('size of matrix must greater than one')
    
    matrix_size: int = len(matrix[0]) - 1
    index_additional: int = 0 if precision is None else 1
    
    if lines:
        lines_len: int = len(lines) - index_additional
        if lines_len < matrix_size:
            raise Exception('number of equations is less than number of unknowns')
        elif lines_len > matrix_size:
            raise Exception('number of equations is greater than number of unknowns')
    
    for index in range(1 + index_additional, matrix_size + index_additional):
        line_current: str = __get_line_from_lines_or_input(lines, index)
        
        line_current_converted: List[float] = __convert_line_and_add_to_matrix(line_current, matrix, precision)
        line_current_len: int = len(line_current_converted)
        if line_current_len != line_first_len:
            raise Exception('matrix must be extended square')
    
    return matrix, precision


def print_matrix_and_precision_and_iteration(matrix: List[List[float]], precision: Optional[int] = None, iteration: Optional[int] = None):
    if iteration is not None:
        print(f'iteration: {iteration}')
    else:
        print()
    
    if precision is not None:
        print(f'floating point precision: {precision}')
    
    lengths_max = __get_lengths_max(matrix, precision)
    
    print('matrix: ')
    for row in matrix:
        for number, length_max in zip(row, lengths_max):
            number_formatted: str = __get_number_formatted(number, precision)
            print(f'{number_formatted:>{length_max}}', end=' ')
        print()
    print()


def __get_number_formatted(number: float, precision: Optional[int] = None) -> str:
    number_formatted: str = f'{number:f}'
    if precision is not None:
        number_formatted = f'{number:.{precision}f}'
    return number_formatted.rstrip('0').rstrip('.')


def __try_convert(value: Any, type_to_convert: Type) -> bool:
    try:
        type_to_convert(value)
        return True
    except ValueError:
        return False


def __get_line_from_lines_or_input(lines: Optional[List[str]], index: int) -> str:
    if lines is not None:
        print(lines[index])
        return lines[index]
    else:
        return input().strip()


def __convert_line_and_add_to_matrix(line: str, matrix: List[List[float]], precision: Optional[int] = None) -> List[float]:
    line_converted: List[float] = []
    for element_splitted in line.split():
        element_float: float = float(element_splitted)
        element_rounded: float = __round(element_float, precision)
        
        line_converted.append(element_rounded)
    matrix.append(line_converted)
    return line_converted


def __get_lengths_max(matrix: List[List[float]], precision: Optional[int] = None) -> List[int]:
    lengths_max: List[int] = [len(__get_number_formatted(element, precision)) for element in matrix[0]]
    for row in matrix:
        for length_index, number in enumerate(row):
            number_formatted: str = __get_number_formatted(number, precision)
            length_current: int = len(number_formatted)
            if length_current > lengths_max[length_index]:
                lengths_max[length_index] = length_current
    return lengths_max


def handler_exit() -> None:
    exit()


def handler_unknown() -> None:
    print('Unknown action')


def main() -> None:
    action_to_handler: Dict[str, Callable] = {
        '0': handler_exit,
        '1': solve_from_random,
        '2': solve_from_file,
        '3': solve_from_console,
    }
    print('Gauss method')
    while True:
        action: str = input("""
Enter action:
0 - exit
1 - enter from random
2 - enter from file
3 - enter from console
""")
        action_handler: Callable = action_to_handler.get(action, handler_unknown)
        try:
            action_handler()
        except ValueError:
            print('Only numbers are allowed')
        except FileNotFoundError:
            print('File not found')
        except Exception as exception:
            print(exception)


if __name__ == '__main__':
    main()
