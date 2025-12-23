import matplotlib.pyplot as plt
import json
import colorsys

def extract_vertex(text):
    temp = text[1:-1].split(":")
    res = []
    for v in temp:
        x, y = map(int, v[1:-1].split(","))
        res.append((x, y))
    return res

def generate_distinct_colors(k):
    colors = []
    for i in range(k):
        # Evenly spaced hues
        h = i / k
        # Keep saturation and value high for better distinction
        s = 0.8
        v = 0.9
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b, 1.0))  # Add alpha=1.0
    return colors

def plot_on_axis(ax, edges, grid_size, title):
    # draw paths
    colors = generate_distinct_colors(len(edges))

    # sort edges by their min vertex to ensure same color assignment to cables across ilp and astar
    edges.sort(key=lambda L: float('inf') if not L else min(min(u, v) for (u, v) in L))

    for idx, commodity in enumerate(edges):
        for edge in commodity:
            x1, y1 = edge[0]
            x2, y2 = edge[1]
            ax.plot([x1, x2], [-y1, -y2], color=colors[idx], linewidth=2)

    # Configure grid and axes
    ax.set_xticks(range(grid_size), minor=True)
    ax.set_yticks(range(-grid_size + 1, 1,1), minor=True)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.set_title(title, wrap=True)
    ax.set_aspect('equal', adjustable='box')

def plot(grid_size=None, edges=None):
    solver = "ilp"
    if grid_size is None:
        input_json = input()
        input_json = input_json[input_json.find("{"):]
        print(input_json)
        obj = json.loads(input_json)
        grid_size = obj["grid_size"]
        # leads to [[(t0, x), (...)], [(t1, y), (...)]]
        edges = [e for e in obj["paths"].values()]
        solver = obj["type"]
        
    GRID_SIZE = (
        grid_size
        if grid_size is not None
        else int(input("Enter the grid size (e.g., 10 for a 10x10 grid): "))
    )

    def get_vertex_number(x, y):
        return x + GRID_SIZE * y + 1

    def get_vertex_coordinates(vertex):
        x = (vertex - 1) % GRID_SIZE
        y = (vertex - 1) // GRID_SIZE
        return (x, y)

    if edges is None:
        lines = []
        partial_sols = []
        partial_sols_missing = []
        complete_sol = -1
        stops = 0 # no. of "stoplines" = every line which gets ignored. generates offset when looking for solutions later
        counter = 0
        try:
            while True:
                counter += 1
                line = input().strip()
                if "starting" in line or "nflic" in line:
                    stops += 1
                    continue
                if "partialsol" in line:
                    stops += 1
                    partial_sols.append(len(lines))
                    partial_sols_missing.append(line.split(".")[-1])
                    continue
                if line == "sol":
                    stops += 1
                    complete_sol = counter
                    continue
                if not line or "time" in line:
                    break
                lines.append(line)
        except EOFError:
            # End of file reached, just continue with program
            pass

        if complete_sol != -1:
            # Single plot if there are no partial solutions
            if len(partial_sols) == 0:
                edges = [[extract_vertex(v) for v in line[:-1].split(";")] for line in lines]
            else:
                edges = [[extract_vertex(v) for v in line[:-1].split(";")] for line in lines[complete_sol - stops :]]
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_on_axis(ax, edges, GRID_SIZE, f"Complete Solution (fails: {len(partial_sols)})")
        else:
            partial_sols.append(len(lines))
            # Create a figure with subplots for each partial solution
            fig, axes = plt.subplots(1, len(partial_sols)-1, figsize=(6*len(partial_sols), 6))

            if len(partial_sols) == 2:  # Handle case with only one subplot
                axes = [axes]

            for i, ax in enumerate(axes):
                edges = [[extract_vertex(v) for v in line[:-1].split(";")] for line in lines[partial_sols[i]:partial_sols[i+1]]]
                title = f"Missing {partial_sols_missing[i]}"
                plot_on_axis(ax, edges, GRID_SIZE, title)

            fig.tight_layout(pad=3.0)

        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_on_axis(ax, edges, GRID_SIZE, f"Complete Solution ({solver})")
        plt.show()
if __name__ == "__main__":
    plot()
