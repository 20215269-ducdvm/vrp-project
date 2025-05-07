from helpers.helpers import remove_duplicates


def test_remove_duplicates():
    v = [6, 5, 1, 3, 3, 4, 1, 6]
    vu = remove_duplicates(v, upper_bound=6, max_zeros_allowed=2)
    print(vu)

