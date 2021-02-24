import decimal
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union


# round(0.5) -> 1.0
decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def solve(matrix: List[List[float]], precision: Optional[int]):
    print_matrix_and_precision_and_iteration(matrix, precision)
    for i in range(len(matrix) - 1):
        for j in range(i + 1, len(matrix)):
            check_zero_division(matrix, i)
            div: float = __round(matrix[j][i] / matrix[i][i], precision)
            matrix[j][-1] = __round(matrix[j][-1] - div * matrix[i][-1], precision)
            for k in range(i, len(matrix)):
                matrix[j][k] = __round(matrix[j][k] - div * matrix[i][k], precision)
        print_matrix_and_precision_and_iteration(matrix, precision, iteration=i + 1)
    
    solution: List[float] = [0 for _ in range(len(matrix))]
    for i in range(len(matrix) - 1, -1, -1):
        check_zero_division(matrix, i)
        numerator: float = sum(matrix[i][j] * solution[j] for j in range(i + 1, len(matrix)))
        solution[i] = __round((matrix[i][-1] - numerator) / matrix[i][i], precision)
    return solution


def check_zero_division(matrix: List[List[float]], i: int) -> None:
    if matrix[i][i] == 0:
        if all(matrix[i][j] == 0 for j in range(len(matrix[i]))):
            raise Exception('Infinite number of answer')
        else:
            raise Exception('No solutions')


def solve_from_file():
    matrix, precision = get_matrix_and_precision_from_file()
    solution = solve(matrix, precision)
    print('solution')
    print(solution)


def solve_from_console():
    matrix, precision = get_matrix_and_precision()
    solution = solve(matrix, precision)
    print('solution')
    print(solution)


def get_matrix_and_precision_from_file() -> Tuple[List[List[float]], Optional[int]]:
    file_name: str = input('Enter file name: ')
    with open(file_name, 'r') as file_input:
        lines: List[str] = file_input.read().splitlines()
        return get_matrix_and_precision(lines)


def get_matrix_and_precision(lines: Optional[List[str]] = None) -> Tuple[List[List[float]], Optional[int]]:
    print('Enter optional floating point precision and then enter matrix: ')
    
    matrix: List[List[float]] = []
    
    line_first, precision = __get_line_first_and_precision(lines)
    
    line_first_converted: List[float] = __convert_line_and_add_to_matrix(line_first, matrix, precision)
    line_first_len: int = len(line_first_converted)
    if line_first_len <= 2 or (lines is not None and len(lines) <= 1):
        raise Exception('size of matrix must greater than one')
    
    matrix_size: int = len(matrix[0]) - 1
    index_additional: int = 0 if precision is None else 1
    
    for index in range(1 + index_additional, matrix_size + index_additional):
        line_current: str = __get_line_from_lines_or_input(lines, index)
        
        line_current_converted: List[float] = __convert_line_and_add_to_matrix(line_current, matrix, precision)
        line_current_len: int = len(line_current_converted)
        if line_current_len != line_first_len:
            raise Exception('matrix must be extended square')
    
    return matrix, precision


def __get_line_first_and_precision(lines: Optional[List[str]]) -> Tuple[str, Optional[int]]:
    line_first: str = __get_line_from_lines_or_input(lines, 0)
    precision: Optional[int] = None
    if __try_convert(line_first, int):
        precision = int(line_first)
        if precision < 0:
            raise Exception('precision must be non-negative')
        if lines is not None and len(lines) > 1:
            line_first = __get_line_from_lines_or_input(lines, 1)
    return line_first, precision


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
            number_formatted: str = __get_float_number_formatted(number, precision)
            number_representable: str = __without_zeros(number_formatted)
            print(f'{number_representable:>{length_max}}', end=' ')
        print()
    print()


def __get_float_number_formatted(number: float, precision: Optional[int] = None) -> str:
    number_formatted: str = f'{number:f}'
    if precision is not None:
        number_formatted = f'{number:.{precision}f}'
    return number_formatted


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


def __round(number: float, precision: Optional[int] = None) -> float:
    if precision is not None:
        number_decimal: decimal.Decimal = decimal.Decimal(str(number))
        number_rounded = round(number_decimal, precision)
        return float(number_rounded)
    return number


def __without_zeros(element: Union[str, int, float]) -> str:
    return str(element).rstrip('0').rstrip('.')


def __get_lengths_max(matrix: List[List[float]], precision: Optional[int] = None) -> List[int]:
    lengths_max: List[int] = [len(__without_zeros(element)) for element in matrix[0]]
    for row in matrix:
        for length_index, number in enumerate(row):
            number_formatted: str = __get_float_number_formatted(number, precision)
            length_current: int = len(__without_zeros(number_formatted))
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
        '1': solve_from_file,
        '2': solve_from_console,
    }
    print('Gauss method')
    while True:
        action: str = input("""
Enter action:
0 - exit
1 - enter from file
2 - enter from console
""")
        action_handler: Callable = action_to_handler.get(action, handler_unknown)
        try:
            action_handler()
        except ValueError:
            print('Only numbers are allowed')
        except Exception as exception:
            print(exception)


if __name__ == '__main__':
    main()
