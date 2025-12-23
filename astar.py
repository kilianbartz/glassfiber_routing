class Cable:
    def __init__(
        self, start: tuple[int, int], end: tuple[int, int], color: str, name: str = ""
    ):
        self.start = start
        self.end = end
        self.name = name
        self.color = color

    def __repr__(self):
        return f"Cable({self.start}, {self.end})"

    def __str__(self):
        return f"Cable from {self.start} to {self.end}"


class Instance:
    def __init__(self, cables: list[Cable], grid_size: int):
        self.cables = cables
        self.grid_size = grid_size

    def __repr__(self):
        return f"Instance({self.cables}, {self.grid_size})"

    def __str__(self):
        return f"Instance with {len(self.cables)} cables and grid size {self.grid_size}"


def solve(instance: Instance) -> list[list[tuple[int, int]]]:
    """
    Solve the given instance using A* algorithm.
    """
    # Placeholder for A* algorithm implementation
    # This should return a list of cables that represent the solution
    # For now, we will just return the cables in the instance
    return [[cable.start, cable.end] for cable in instance.cables]


def main():
    # Example usage
    cables = [
        Cable((0, 0), (8, 0), "yellow"),
        Cable((1, 0), (9, 0), "brown"),
        Cable((2, 0), (9, 2), "green"),
        Cable((3, 0), (9, 6), "red"),
        Cable((6, 0), (5, 9), "blue"),
    ]
    instance = Instance(cables, grid_size=10)
    print(instance)
    solution = solve(instance)
    print("Solution:", solution)


if __name__ == "__main__":
    main()
