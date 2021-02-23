import decimal
from typing import List
from typing import Optional
from typing import Tuple
# round(0.5) -> 1.0
from typing import Union


decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def get_matrix_and_precision_from_file() -> Tuple[List[List[float]], Optional[int]]:
    file_name: str = input('Enter file name: ')
    with open(file_name, 'r') as file_input:
        lines: List[str] = file_input.read().splitlines()
        return get_matrix_and_precision(lines)


def get_matrix_and_precision(lines: Optional[List[str]] = None) -> Tuple[List[List[float]], Optional[int]]:
    print('Enter optional floating point precision and then enter matrix: ')
    matrix: List[List[float]] = []
    precision: Optional[int] = None
    
    line_first: str = __get_line_from_lines_or_input(lines, 0)
    if line_first.replace('.', '', 1).isnumeric():
        precision = int(line_first)
        line_first = __get_line_from_lines_or_input(lines, 1)
    
    __get_convert_line_and_add_to_matrix(line_first, matrix, precision)
    
    matrix_size: int = len(matrix[0]) - 1
    index_additional: int = 0 if precision is None else 1
    
    for index in range(1 + index_additional, matrix_size + index_additional):
        line_input: str = __get_line_from_lines_or_input(lines, index)
        __get_convert_line_and_add_to_matrix(line_input, matrix, precision)
    
    return matrix, precision


def __get_line_from_lines_or_input(lines: Optional[List[str]], index: int) -> str:
    if lines is not None:
        print(lines[index])
        return lines[index]
    else:
        return input().strip()


def __get_convert_line_and_add_to_matrix(line: str, matrix: List[List[float]], precision: Optional[int] = None) -> None:
    line_converted: List[float] = []
    for element_splitted in line.split():
        element_float: float = float(element_splitted)
        element_rounded: float = __round(element_float, precision)
        
        line_converted.append(element_rounded)
    matrix.append(line_converted)


def __round(number: float, precision: Optional[int] = None) -> float:
    if precision is not None:
        number_decimal: decimal.Decimal = decimal.Decimal(str(number))
        number_rounded = round(number_decimal, precision)
        return float(number_rounded)
    return number


def __without_zeros(element: Union[str, int, float]) -> str:
    return str(element).rstrip('0').rstrip('.')


def print_matrix_and_precision(matrix: List[List[float]], precision: Optional[int] = None):
    if precision is not None:
        print(f'floating point precision: {precision}')
    
    lengths_max = __get_lengths_max(matrix)
    
    print('matrix: ')
    for row in matrix:
        for number, length_max in zip(row, lengths_max):
            number_representable: str = __without_zeros(f'{number}')
            print(f'{number_representable:>{length_max}}', end=' ')
        print()


def __get_lengths_max(matrix: List[List[float]]) -> List[int]:
    lengths_max: List[int] = [len(__without_zeros(element)) for element in matrix[0]]
    for row in matrix:
        for length_index, number in enumerate(row):
            length_current: int = len(__without_zeros(number))
            if length_current > lengths_max[length_index]:
                lengths_max[length_index] = length_current
    return lengths_max


def main() -> None:
    matrix: List[List[float]]
    precision: Optional[int]
    matrix, precision = get_matrix_and_precision_from_file()
    print_matrix_and_precision(matrix, precision)


if __name__ == '__main__':
    main()
