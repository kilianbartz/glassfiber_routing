import random
import sys
import typer
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from typing_extensions import Annotated

sys.setrecursionlimit(10**5)

app = typer.Typer()

@contextmanager
def get_output_stream(file_path: Optional[Path], number: int):
    # If a path is provided, open the file; otherwise, use stdout
    if file_path:
        # Ensure the parent directory exists
        file_path.mkdir(parents=True, exist_ok=True)
        with open(file_path / f"{number}.txt", "w", encoding="utf-8") as f:
            yield f
    else:
        yield sys.stdout

@dataclass
class Instance:
    nets: set[tuple[int, int]]

    def __hash__(self):
        return hash(frozenset(self.nets))

    def __eq__(self, other):
        if len(self.nets) != len(other.nets):
            return False
        return self.nets == other.nets


def get_vertex_number(x: int, y: int, grid_size: int) -> int:
    return x + grid_size * y + 1

def generate_point(side: str, used_points: set, grid_size: int) -> tuple[int, int]:
    if side == "top":
        point = (random.randint(1, grid_size - 2), 0)
    elif side == "bottom":
        point = (random.randint(1, grid_size - 2), grid_size - 1)
    elif side == "left":
        point = (0, random.randint(1, grid_size - 2))
    else:
        point = (grid_size - 1, random.randint(1, grid_size - 2))
    if point in used_points:
        return generate_point(side, used_points, grid_size)
    used_points.add(point)
    return point

SIDES = ["top", "bottom", "left", "right"]

@app.command()
def generate_instances(
    grid_size: int,
    number_of_nets: int,
    number_of_instances: Annotated[int, typer.Option("--instances", "-i", help="Number of instances to generate")] = 1,
    output: Optional[Path] = typer.Option("--output", "-o", help="Path to output directory.")
):
    """
    Generates random, legal cabling problem instances with the specified grid size and number of cables.
    When generating multiple instances, providing an output directory is prefered.
    """
    generated_instances = set()
    while len(generated_instances) < number_of_instances:
        ins = generate_single_instance(grid_size, number_of_nets)
        generated_instances.add(ins)

    
    for id, ins in enumerate(generated_instances):
        with get_output_stream(output, id) as stream:
            stream.write(f"{grid_size}\n")
            for net in ins.nets:
                start, target = net
                stream.write(f"{start} {target}\n")

def generate_single_instance(grid_size: int, num_cables: int) -> Instance:
    used_points = set()
    nets = set()
    for i in range(num_cables):
        source_side = random.choice(SIDES)
        sink_side = random.choice(SIDES)
        while source_side == sink_side:
            sink_side = random.choice(SIDES)
    
        source = generate_point(source_side, used_points, grid_size)
        sink = generate_point(sink_side, used_points, grid_size)
        nets.add((get_vertex_number(*source, grid_size), get_vertex_number(*sink, grid_size)))
    
    return Instance(nets=nets)

if __name__ == "__main__":
    app()
