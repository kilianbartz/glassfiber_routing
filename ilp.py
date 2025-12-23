import gurobipy as gp
import json
from gurobipy import GRB
from plot_solution import plot
from time import time
import argparse

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Literal

SolutionMapping = Dict[
    Tuple[int, int],
    Dict[
        str,
        List[
            Tuple[
                Tuple[int, int],
                Tuple[int, int]
            ]
        ]
    ]
]

@dataclass
class Result:
    """
    Dataclass for storing the results of an Integer Linear Programming (ILP)
    optimization run.

    Fields:
        type (Literal["ilp"]): Indicates the result is from an ILP process.
        solved (bool): True if the optimization process found an optimal solution.
        solution_mapping (SolutionMapping): The complex nested dictionary representing
            the structure of the solution.
            - Keys are tuples of (int, int).
            - Values are lists of tuples.
            - Each inner tuple contains two (int, int) tuples, often used to
              represent paths or connections between nodes.
    """
    solved: bool
    time: float
    grid_size: int
    missing: list[str]
    type: Literal["ilp"] = "ilp"

    # The complex dictionary type specified in the request
    paths: SolutionMapping = field(default_factory=dict)
    

parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="This program tries to solve an instance of a grid-based glass fiber switch box routing problem. It expects a text instance (TI) as stdin input (see project report)",
)
parser.add_argument("-p", "--plot", action="store_true", help="Whether the result should be plotted")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

GRID_SIZE = int(input("Enter the grid size (e.g., 10 for a 10x10 grid): " if args.verbose else ""))
times = []


def get_vertex_number(x, y):
    return x + GRID_SIZE * y + 1


def get_vertex_coordinates(vertex):
    x = (vertex - 1) % GRID_SIZE
    y = (vertex - 1) // GRID_SIZE
    return (x, y)


# Create a new model
model = gp.Model("MultiCommodityFlow")

# Define the network


nodes = list(range(1, GRID_SIZE * GRID_SIZE + 1))
edges = []
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        vertex = get_vertex_number(x, y)
        right_neighbor = (
            get_vertex_number(x + 1, y)
            if x < GRID_SIZE - 1 and not (y == 0 or y == GRID_SIZE - 1)
            else None
        )
        down_neighbor = (
            get_vertex_number(x, y + 1)
            if y < GRID_SIZE - 1 and not (x == 0 or x == GRID_SIZE - 1)
            else None
        )
        left_neighbor = (
            get_vertex_number(x - 1, y)
            if x > 0 and not (y == 0 or y == GRID_SIZE - 1)
            else None
        )
        up_neighbor = (
            get_vertex_number(x, y - 1)
            if y > 0 and not (x == 0 or x == GRID_SIZE - 1)
            else None
        )
        if right_neighbor:
            edges.append((vertex, right_neighbor))
        if down_neighbor:
            edges.append((vertex, down_neighbor))
        if left_neighbor:
            edges.append((vertex, left_neighbor))
        if up_neighbor:
            edges.append((vertex, up_neighbor))

edge_capacity = 1

# Define commodities (source, sink)
if args.verbose:
    print(
        "Enter the commodities (source, sink) pairs, one per line (e.g., 1 2). Use vertex numbers from 1 to",
        GRID_SIZE * GRID_SIZE,
    )
commodities = []
try:
    while True:
        line = input().strip()
        if line == "":
            break

        source, sink = map(int, line.split())
        if source in nodes and sink in nodes and source != sink:
            commodities.append((source, sink))
        else:
            print("Invalid input. Please enter valid node numbers.")
except EOFError:
    # End of file reached, just continue with program
    pass

# Create flow variables for each commodity on each edge
flow = {}
for k, commodity in enumerate(commodities):
    for i, j in edges:
        flow[k, i, j] = model.addVar(name=f"flow_{k}_{i}_{j}", lb=0, vtype=GRB.BINARY)
        # For convenience, create variables for reverse direction with 0 flow
        if (j, i) not in edges:
            flow[k, j, i] = model.addVar(
                name=f"flow_{k}_{j}_{i}", lb=0, ub=0, vtype=GRB.BINARY
            )

if args.verbose:
    print("Flow variables created successfully.")
# Update model to integrate new variables
model.update()

start = time()
# Add constraints
# 1. Flow conservation constraints
demand = 1  # Demand for each commodity
for k, (source, sink) in enumerate(commodities):
    for node in nodes:
        outflow = sum(
            flow[k, node, j] for j in nodes if (node, j) in edges or (j, node) in edges
        )
        inflow = sum(
            flow[k, j, node] for j in nodes if (j, node) in edges or (node, j) in edges
        )

        if node == source:
            # Net outflow at source must equal the demand
            model.addConstr(outflow - inflow == demand, name=f"source_{k}_{node}")
        elif node == sink:
            # Net inflow at sink must equal the demand
            model.addConstr(inflow - outflow == demand, name=f"sink_{k}_{node}")
        else:
            # Flow conservation at transit nodes
            model.addConstr(outflow - inflow == 0, name=f"transit_{k}_{node}")
        #model.addConstr(outflow <= 1, name=f"transit_{k}_{node}_outflow")
        #model.addConstr(inflow <= 1, name=f"transit_{k}_{node}_inflow")

dur = time() - start
times.append(dur)
if args.verbose:
    print(
        f"Flow conservation constraints added successfully. Time taken: {dur:.2f} seconds"
    )
start = time()
# 2. Capacity constraints for each edge
for i, j in edges:
    # Sum of all commodity flows on an edge (undirected)must not exceed the edge capacity
    model.addConstr(
        sum(flow[k, i, j] for k in range(len(commodities)))
        + sum(flow[k, j, i] for k in range(len(commodities)))
        <= 1,
        name=f"capacity_{i}_{j}",
    )

dur = time() - start
times.append(dur)
if args.verbose:
    print(f"Capacity constraints added successfully. Time taken: {dur:.2f} seconds")
start = time()

# Set objective: minimize total flow cost
total_flow = sum(flow[k, i, j] for k in range(len(commodities)) for i, j in edges)
model.setObjective(total_flow, GRB.MINIMIZE)

dur = time() - start
times.append(dur)
if args.verbose:
    print(f"Model built successfully. Time taken: {dur:.2f}. Starting optimization...")
start = time()

# Solve the model
model.optimize()
dur = time() - start
times.append(dur)

grid = {}

# Print the results
if model.status == GRB.OPTIMAL:
    paths = {}
    for k, (source, sink) in enumerate(commodities):
        grid[k] = []
        # print(f"\nCommodity {k+1} (from {source} to {sink}, demand={demand}):")
        path = []
        for i, j in edges:
            if flow[k, i, j].x > 1e-6:  # Only print non-zero flows
                path.append((get_vertex_coordinates(i), get_vertex_coordinates(j)))
        paths[str((source, sink))] = path
    result = Result(solved=True, paths=paths, time=sum(times), grid_size=GRID_SIZE, missing=[])
    print(json.dumps(asdict(result)))
    if args.plot:
        paths_flat = [p for p in paths.values()]
        plot(GRID_SIZE, paths_flat)

else:
    result = Result(solved=False, time=sum(times), grid_size=GRID_SIZE, missing=commodities)
    print(json.dumps(asdict(result)))
if args.verbose:
    print(f"Optimization completed in {dur:.2f} seconds")

