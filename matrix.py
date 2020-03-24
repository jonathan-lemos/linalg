from typing import Iterable, List, Tuple, Union
from fractions import Fraction
from copy import deepcopy
from functools import reduce

Number = Union[float, int, Fraction]


def num_to_string(n: Number):
    return str(Fraction(n).limit_denominator())


class Row:
    def __init__(self, x: Iterable[Number]):
        self.__nums = list(x)

    def __len__(self):
        return len(self.__nums)

    def __getitem__(self, item: Union[int, slice]) -> Number:
        return self.__nums[item]

    def __iter__(self) -> Iterable[Number]:
        return iter(self.__nums)

    def __neg__(self):
        return Row(-x for x in self.__nums)

    def __add__(self, other: "Row") -> "Row":
        if len(other) != len(self):
            raise ValueError(f"Cannot add rows of lengths {len(self)} and {len(other)}.")
        return Row(x + y for x, y in zip(self.__nums, other.__nums))

    def __sub__(self, other: "Row"):
        return self + -other

    def __mul__(self, other: Number):
        return Row(x * other for x in self.__nums)

    def __truediv__(self, other: Number):
        return self * (1 / other)

    def __str__(self):
        return f"[{', '.join(num_to_string(x) for x in self.__nums)}]"

    def __copy__(self):
        return Row(self.__nums)

    def __deepcopy__(self, memodict):
        return Row(self.__nums)

    def __setitem__(self, key: int, value: Number):
        self.__nums[key] = value

    def pivot(self):
        for i in range(len(self)):
            if self[i] != 0:
                return i
        return -1

    def is_zero(self):
        return all(x == 0 for x in self.__nums)


class Matrix:
    def __init__(self, x: Iterable[Iterable[Number]]):
        self.__rows = list(Row(z) for z in x)
        if len(self.__rows) == 0:
            raise ValueError("A matrix must have at least one row.")
        row_lengths = set(len(x) for x in self.__rows)
        if len(row_lengths) > 1:
            raise ValueError(
                f"A matrix can only have one column size, but the rows have {list(sorted(row_lengths))} entries.")
        self.__col_len = row_lengths.pop()

    def __len__(self):
        return len(self.__rows)

    def dim(self) -> Tuple[int, int]:
        return len(self), len(self.__rows[0])

    def __getitem__(self, item: Union[int, slice]) -> Union[Row, Iterable[Row]]:
        return self.__rows[item]

    def __setitem__(self, key: int, value: Row):
        self.__rows[key] = value

    def __iter__(self) -> Iterable[Row]:
        return iter(self.__rows)

    def __str__(self):
        max_len = max(len(num_to_string(n)) for row in self.__rows for n in row)
        rows = [f"| {' '.join(num_to_string(n).ljust(max_len) for n in row)} |" for row in self.__rows]
        header = '-' * max(len(s) for s in rows)
        return header + "\n" + "\n".join(rows) + "\n" + header

    def __copy__(self):
        return Matrix(self.__rows)

    def __deepcopy__(self, memodict):
        return Matrix(self.__rows)

    def swap_rows(self, i: int, j: int):
        self.__rows[i], self.__rows[j] = self.__rows[j], self.__rows[i]

    def col(self, i: int) -> List[int]:
        return [self.__rows[j][i] for j in range(len(self))]

    def det(self, verbose: int = 0):
        new = deepcopy(self)
        neg = 1

        if new.dim()[1] != new.dim()[1]:
            raise ValueError(f"Cannot get the determinant of a non-square matrix (was {new.dim()}).")

        if verbose > 0:
            print("Initial: ")
            print(new)

        for i in range(len(new)):
            for k in range(i, len(new)):
                if new[k][i] != 0:
                    break
            else:
                continue

            if k != i:
                new.swap_rows(i, k)
                neg *= -1

            if verbose > 1 and k != i:
                print(f"r{i} <-> r{k}")
                print(new)

            pivot = new[i][i]

            for j in range(i + 1, len(new)):
                op_is_not_useless = new[j][i] != 0 and not new[i].is_zero()

                if verbose > 1 and op_is_not_useless:
                    print(f"r{j} += r{i} * {num_to_string(-new[j][i] / pivot)} ")

                new.__rows[j] += -new[i] * new[j][i] / pivot

                if verbose > 1 and op_is_not_useless:
                    print(new)

            if verbose == 1:
                print(new)

        return neg * reduce(lambda a, c: a * new[c][c], range(len(new)), 1)

    def ref(self, verbose: int = 0) -> "Matrix":
        new = deepcopy(self)

        if verbose > 0:
            print("Initial: ")
            print(new)

        for i in range(len(new)):
            pivots = [x for x in sorted(
                ((row.pivot(), ind) for row, ind in zip(new[i:len(new.__rows)], range(i, len(new.__rows)))),
                key=lambda x: (x[0], x[1])) if x[0] != -1]
            if len(pivots) == 0:
                continue
            pivot_ind, pivot_row = pivots[0]
            if pivot_ind == -1:
                continue
            pivot = new[pivot_row][pivot_ind]
            new.swap_rows(i, pivot_row)

            if verbose > 1 and pivot_row != i:
                print(f"r{i} <-> r{pivot_row}")
                print(new)

            new.__rows[i] /= pivot

            if verbose > 1 and pivot != 1:
                print(f"r{i} *= {num_to_string(1 / pivot)}")
                print(new)

            for j in range(i + 1, len(new)):
                op_is_not_useless = new[j][pivot_ind] != 0 and not new[i].is_zero()

                if verbose > 1 and op_is_not_useless:
                    print(f"r{j} += r{i} * {num_to_string(-new[j][pivot_ind])} ")

                new.__rows[j] += -new[i] * new[j][pivot_ind]

                if verbose > 1 and op_is_not_useless:
                    print(new)

            if verbose == 1:
                print(new)

        return new

    def rref(self, verbose: int = 0) -> "Matrix":
        new = self.ref(verbose)

        for i in range(1, len(new)):
            pivot_pos = new[i].pivot()
            if pivot_pos == -1:
                continue
            pivot = new[i][pivot_pos]
            for j in range(0, i):
                op_is_not_useless = new[j][pivot_pos] != 0 and not new[i].is_zero()

                if verbose > 1 and op_is_not_useless:
                    print(f"r{j} += r{i} * {num_to_string(-new[j][pivot_pos] / pivot)}")

                new.__rows[j] += -new[i] * new[j][pivot_pos] / pivot

                if verbose > 1 and op_is_not_useless:
                    print(new)

            if verbose == 1:
                print(new)

        return new
