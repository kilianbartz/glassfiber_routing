use clap::{Parser, Subcommand, command};
use glob::glob;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::prelude::*;

type Terminal = u32;
type Net = (Terminal, Terminal);
type Matching = Vec<Net>;
type Edge = ((i32, i32), (i32, i32));

#[derive(Subcommand)]
enum Commands {
    /// Generates all legal instances within the specified limits and runs them through the command. The output is written to `results_<p>_<s>-<n_min>-<n_max>.jsonl` at the end.
    Generated {
        grid_size: usize,
        /// Minimum number of cables per instance
        start_cables: u32,
        /// Maximum number of cables per instance
        end_cables: u32,
        command: String,
    },
    /// Runs all instances found in folder/**/*.txt through command. The output is written to `folder_<folder>.jsonl`
    Folder { folder: PathBuf, command: String },
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Serialize, Eq, Ord, PartialEq, PartialOrd)]
struct Instance {
    grid_size: usize,
    matching: Matching,
    id: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct AlgoOutput {
    r#type: String,
    solved: bool,
    paths: HashMap<String, Vec<Edge>>,
    time: f64,
}

#[derive(Debug, Clone, Serialize)]
struct RunResult {
    instance: Instance,
    success: bool,
    total_time: f64,
    missing: i32,
    output: AlgoOutput,
}
impl RunResult {
    fn from_algo_output(algo_output: AlgoOutput, instance: Instance) -> Self {
        Self {
            success: algo_output.solved,
            total_time: algo_output.time,
            missing: (instance.matching.len() - algo_output.paths.keys().len()) as i32,
            instance,
            output: algo_output,
        }
    }
}

impl Instance {
    fn new(grid_size: usize, matching: Matching, id: usize) -> Self {
        Self {
            grid_size,
            matching,
            id,
        }
    }

    fn from_text_file(path: PathBuf, id: usize) -> Result<Self, std::io::Error> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let lines: Vec<String> = contents.lines().map(|s| s.to_owned()).collect();
        let grid_size: usize = lines[0].parse().expect("Not a valid grid size in line 0!");
        let mut nets = vec![];
        for line in &lines[1..] {
            let parts: Vec<String> = line.split_whitespace().map(|s| s.to_owned()).collect();
            assert_eq!(parts.len(), 2);
            let start_term: Terminal = parts[0].parse().expect("Not a valid terminal number!");
            let end_term: Terminal = parts[1].parse().expect("Not a valid terminal number!");
            nets.push((start_term, end_term) as Net);
        }
        Ok(Instance {
            grid_size,
            matching: nets as Matching,
            id,
        })
    }
}
impl fmt::Display for Instance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", self.grid_size)?;
        for (v1, v2) in &self.matching {
            writeln!(f, "{} {}", v1, v2)?;
        }
        Ok(())
    }
}

struct MatchingGenerator {
    partitions: BTreeMap<String, Vec<Terminal>>,
}

impl MatchingGenerator {
    fn new(partitions: BTreeMap<String, Vec<Terminal>>) -> Self {
        Self { partitions }
    }

    fn generate_matchings(&self, matching_size: usize) -> impl Iterator<Item = Matching> + '_ {
        // Create a reverse mapping from vertex to its partition for easy lookup
        let mut vertex_to_partition: BTreeMap<Terminal, String> = BTreeMap::new();
        let mut all_vertices: Vec<Terminal> = Vec::new();
        for (p_name, v_set) in &self.partitions {
            for &vertex in v_set {
                vertex_to_partition.insert(vertex, p_name.clone());
                all_vertices.push(vertex);
            }
        }
        all_vertices.sort();

        // Generate all possible edges in the complete multipartite graph
        let mut all_possible_edges: Vec<Net> = Vec::new();

        // Iterate over every unique pair of vertices
        for combo in all_vertices.iter().combinations(2) {
            let &v1 = combo[0];
            let &v2 = combo[1];

            // An edge is valid if its vertices are in different partitions
            if vertex_to_partition[&v1] != vertex_to_partition[&v2] {
                // Store the edge in a canonical format (sorted)
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                all_possible_edges.push(edge);
            }
        }

        // Generate all combinations of edges of the desired size
        all_possible_edges
            .into_iter()
            .combinations(matching_size)
            .filter(move |edge_combo| {
                // Validate if the combination of edges is a true matching, i.e., every used vertex is unique
                // It might be more idiomatic to use a HashSet for this, but because of its more expensive
                // allocation and overhead, this solution is faster.
                // Another alternative would be to just iterate over the edges and fail early if a vertex
                // appears a second time using a bit array. However, heap allocation (for the array)
                // would require a hard-coded maximum number of terminals.
                let mut vertices_in_combo: Vec<Terminal> =
                    edge_combo.iter().flat_map(|(v1, v2)| [*v1, *v2]).collect();

                // dedup only removes consecutive repeated elements
                vertices_in_combo.sort();
                vertices_in_combo.dedup();

                vertices_in_combo.len() == matching_size * 2
            })
    }
}

fn parse(instance: Instance, input: &str) -> RunResult {
    let Some(index) = input.find('{') else {
        panic!("Invalid input: {}", input);
    };
    let input = &input[index..];
    let output = serde_json::from_str::<AlgoOutput>(input).unwrap();
    RunResult::from_algo_output(output, instance)
}

fn get_vertex_number(x: usize, y: usize, grid_size: usize) -> Terminal {
    (x + grid_size * y + 1) as Terminal
}

fn create_partitions(grid_size: usize) -> BTreeMap<String, Vec<Terminal>> {
    let top_terminals: Vec<Terminal> = (1..grid_size - 1)
        .map(|x| get_vertex_number(x, 0, grid_size))
        .collect();

    let bottom_terminals: Vec<Terminal> = (1..grid_size - 1)
        .map(|x| get_vertex_number(x, grid_size - 1, grid_size))
        .collect();

    let left_terminals: Vec<Terminal> = (1..grid_size - 1)
        .map(|y| get_vertex_number(0, y, grid_size))
        .collect();

    let right_terminals: Vec<Terminal> = (1..grid_size - 1)
        .map(|y| get_vertex_number(grid_size - 1, y, grid_size))
        .collect();

    let mut partitions = BTreeMap::new();
    partitions.insert("top".to_string(), top_terminals);
    partitions.insert("bottom".to_string(), bottom_terminals);
    partitions.insert("left".to_string(), left_terminals);
    partitions.insert("right".to_string(), right_terminals);

    partitions
}

fn run_command_with_input(
    command: &str,
    args: &[&str],
    input_data: &str,
) -> Result<String, String> {
    let mut child = Command::new(command)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute command '{}': {}", command, e))?;

    // Write input data to stdin
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(input_data.as_bytes())
            .map_err(|e| format!("Failed to write to stdin: {}", e))?;
    }

    // Wait for the command to finish and collect output
    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to read output: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn test_instance_with_command(
    instance: &Instance,
    _instance_id: usize,
    command1: (&str, Vec<&str>),
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let instance_data = instance.to_string();
    //println!("{:?}", instance_data);

    let cmd_result = run_command_with_input(command1.0, &command1.1, &instance_data);
    let cmd_output = match &cmd_result {
        Ok(output) => output.clone(),
        Err(error) => error.clone(),
    };

    // Create result JSON
    let result = parse(instance.clone(), &cmd_output);
    Ok(result)
}

fn sanitize_filename(name: &str) -> String {
    // List of characters forbidden on most filesystems (Windows/Unix)
    let forbidden_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|'];

    let mut safe_name = String::with_capacity(name.len());

    for c in name.chars() {
        if forbidden_chars.contains(&c) || c.is_control() {
            // Replace forbidden characters and control characters with an underscore
            safe_name.push('_');
        } else {
            safe_name.push(c);
        }
    }

    // Optional: Trim leading/trailing whitespace which can sometimes be problematic
    let trimmed_name = safe_name.trim();

    // Prevent an empty string if the original contained only forbidden characters/whitespace
    if trimmed_name.is_empty() {
        return "untitled_file".to_string();
    }

    trimmed_name.to_string()
}

fn convert_command(command: &str) -> (&str, Vec<&str>) {
    let parts = command.split(" ");
    let cmd = parts.collect::<Vec<&str>>();
    (cmd[0], cmd[1..].to_vec())
}

fn test_generated_instances(
    grid_size: usize,
    start_cables: u32,
    end_cables: u32,
    command: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create partitions
    let partitions = create_partitions(grid_size);
    let generator = MatchingGenerator::new(partitions);

    let file_name = format!(
        "results_{}_{}-{}-{}.jsonl",
        sanitize_filename(command),
        grid_size,
        start_cables,
        end_cables
    );
    let command = convert_command(&command);

    // Generate and test instances
    // Note: An argument could be made for consuming the generate_matchings iterator using par_bridge(),
    // so that the entire vector does not have to be allocated and kept in memory. However,
    // for the small instances relevant here (grid size 5, 6), the convenience of a progress bar outweighs
    // the memory savings imo.
    let instances: Vec<Instance> = (start_cables..end_cables + 1)
        .flat_map(|matching_size| {
            generator
                .generate_matchings(matching_size as usize)
                .enumerate()
                .map(|(id, matching)| Instance::new(grid_size, matching, id))
        })
        .collect();
    generate_and_write_results(instances, &file_name, command)
}

fn generate_and_write_results(
    instances: Vec<Instance>,
    file_name: &str,
    command: (&str, Vec<&str>),
) -> Result<(), Box<dyn std::error::Error>> {
    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&file_name)?;
    let results: Vec<Result<RunResult, Box<dyn std::error::Error + Send + Sync>>> =
        instances
            .par_iter()
            .enumerate()
            .progress_count(instances.len() as u64)
            .with_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {eta}"
                    )
                    .expect("Failed to set progress bar template")
                    .progress_chars("#>-"),
            )
            .map(|(global_id, instance)| {
                // Clone command for this thread
                test_instance_with_command(instance, global_id, command.clone())
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(e.to_string()))
                    })
            })
            .collect();

    // Write results to file sequentially to avoid file access conflicts
    for (global_id, result) in results.into_iter().enumerate() {
        match result {
            Ok(run_result) => {
                let res_string = serde_json::to_string(&run_result)
                    .map_err(|e| format!("Failed to serialize result to JSON: {}", e))?;
                writeln!(output_file, "{}", res_string)?;
            }
            Err(e) => {
                eprintln!("Error testing instance {}: {}", global_id, e);
            }
        }
    }
    output_file.flush()?;

    println!("Results written to {}", file_name);
    Ok(())
}

fn test_folder(folder: PathBuf, command: &str) -> Result<(), Box<dyn std::error::Error>> {
    let instances: Vec<Instance> = glob(&format!("{}/**/*.txt", folder.to_str().unwrap()))?
        .enumerate()
        .map(|(id, p)| Instance::from_text_file(p.unwrap(), id).unwrap())
        .collect();
    let file_name = format!("folder_{}.jsonl", folder.to_str().unwrap());
    let command = convert_command(&command);
    generate_and_write_results(instances, &sanitize_filename(&file_name), command)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    match args.command {
        Commands::Generated {
            grid_size,
            start_cables,
            end_cables,
            command,
        } => test_generated_instances(grid_size, start_cables, end_cables, &command),
        Commands::Folder { folder, command } => test_folder(folder, &command),
    }
}
