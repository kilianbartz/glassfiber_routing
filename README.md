This repository includes the **project report** and source code for my research seminar and research project on the topic of **grid-based routing of glass fiber switch boxes**.

The source code includes 
- an implementation of an ILP-based solver (`ilp.py`),
- an implementation of an A*-based solved (`astar_rust` directory),
- utility programs/scripts (`plot_solution.py`, `algo_tester`, `gen_instance.py`), described in the project report,
- utility shell scripts (`densitytests.sh`, `gen_densitytests.sh`), which I used to generate the results for the experiments in my project report.

The structure and working principle of all components is described in the **project report**.

# Configuration
First, clone this repository.
The Python part of this repository is managed by the package manager [uv](https://docs.astral.sh/uv/). After [installing uv](https://docs.astral.sh/uv/getting-started/installation/), a virtual environment can be created by running `uv sync` in the base directory of this repository.
The python scripts can then be called using uv: `uv run <script.py>`.

The Rust project `astar_rust` and the Algorithm Tester (`algo_tester`) are managed by Cargo which comes with a standard installation of Rust. After [installing Rust](https://rustup.rs), go to the respective subdirectory and run `cargo build --release` to build an executable or `cargo run --release` to directly run the program.

# Usage
The ILP-based solver, the A*-based solver, the Algorithm Tester and the random instances generator all provide CLIs with a help page. It can be accessed by `uv run <script.py> --help` or `cargo run --release -- --help` (or `./bin --help` if you already built a binary) respectively and explains the available commands and arguments, which can be passed to the programs. Additional information can be found in **project report**.
