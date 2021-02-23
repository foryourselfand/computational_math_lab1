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
            raise Exception('matrix must be square')
    
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


def print_matrix_and_precision(matrix: List[List[float]], precision: Optional[int] = None):
    print()
    if precision is not None:
        print(f'floating point precision: {precision}')
    
    lengths_max = __get_lengths_max(matrix)
    
    print('matrix: ')
    for row in matrix:
        for number, length_max in zip(row, lengths_max):
            number_representable: str = __without_zeros(f'{number}')
            print(f'{number_representable:>{length_max}}', end=' ')
        print()


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


def __get_lengths_max(matrix: List[List[float]]) -> List[int]:
    lengths_max: List[int] = [len(__without_zeros(element)) for element in matrix[0]]
    for row in matrix:
        for length_index, number in enumerate(row):
            length_current: int = len(__without_zeros(number))
            if length_current > lengths_max[length_index]:
                lengths_max[length_index] = length_current
    return lengths_max


def handler_exit() -> None:
    exit()


def main() -> None:
    matrix: List[List[float]]
    precision: Optional[int]
    
    action_to_handler: Dict[str, Callable] = {
        '0': handler_exit,
        '1': get_matrix_and_precision_from_file,
        '2': get_matrix_and_precision,
    }
    print('Gauss method')
    while True:
        action: str = input("""
Enter action:
0 - exit
1 - enter from file
2 - enter from console
""")
        action_handler: Callable = action_to_handler.get(action, None)
        if action_handler is None:
            print('Unknown action')
            continue
        
        try:
            matrix, precision = action_handler()
            print_matrix_and_precision(matrix, precision)
        except ValueError:
            print('Only numbers are allowed')
        except Exception as exception:
            print(exception)


if __name__ == '__main__':
    main()
