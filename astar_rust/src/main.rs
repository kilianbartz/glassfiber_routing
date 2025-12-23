//! This crate implements an A*-based solver for grid-based glass fiber cabling problems. Solutions may be optimized through bend flipping to maximize the number of routed nets.
//! The program offers a CLI interface. More information can be found with ./astar --help.
//!
//! The project is structured as follows:
//! - main.rs defines the cli, parses the stdin input, generates the problem instance and calls the solver
//! - cabling::base contains the basic primitive data structures for Terminals, Nets, Edges
//! - cabling::astar contains data structures to describe the cabling problem, its solution and helper structures. Apart from that,
//!   an implementation of a priority queue (based on MinHeap) and the process_pq function to perform a single A*-step.
//! More information can be found in the accompanying documentation pdf.
mod cabling;

use cabling::astar::{
    CablingProblem, PriorityQueue, Solution, find_conflicting_net, fix_bend, fix_bend_dynamic,
    process_pq,
};
use clap::{Parser, ValueEnum};
use itertools::Itertools;
use std::collections::HashSet;
use std::fmt;
use std::io;
use std::time::Instant;

#[derive(Debug, Clone, ValueEnum)]
enum ShoveAsideStrategy {
    Static,
    Dynamic,
}
impl fmt::Display for ShoveAsideStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the debug format and strip the "ShoveAsideStrategy::" prefix
        write!(f, "{}", self.to_possible_value().unwrap().get_name())
    }
}

///A*-based solver for grid-based glass fiber cabling problems. Solutions may be optimized through bend flipping to maximize the number of routed nets
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    ///Sets the constant c of shove aside iterations. In total, n+c (with n the number of nets) bend flips are tried
    #[arg(short = 'i', long)]
    shove_aside_iterations: usize,

    ///Sets the bend flip strategy to either static or dynamic. In general, static is faster, but dynamic may solve more instances
    #[arg(short, long, default_value_t = ShoveAsideStrategy::Dynamic)]
    shove_aside_strategy: ShoveAsideStrategy,

    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    let stdin = io::stdin();
    let mut first_line = String::new();
    stdin.read_line(&mut first_line).unwrap();
    let grid_size: i32 = first_line.trim().parse().unwrap();
    let nets: Vec<(i32, i32)> = stdin
        .lines()
        .map(|v| {
            let s = v.unwrap();
            let numbers: Vec<i32> = s
                .split_whitespace()
                .map(|num| num.parse().unwrap())
                .collect();
            //[4, 15]
            assert_eq!(numbers.len(), 2);
            (numbers[0], numbers[1])
        })
        .collect();

    let start = Instant::now();
    let mut prob = CablingProblem::new(grid_size, nets);

    let mut sol = Solution::new();
    let starting_points = prob.init_grid_points();
    let mut pq = PriorityQueue::new();
    pq.push_vec(starting_points);
    while process_pq(&mut pq, &mut prob, &mut sol) {}
    let mut filtered_nets = HashSet::new();
    if args.shove_aside_iterations == 0 {
        return;
    }
    for i in 0..(prob.nets.len() + args.shove_aside_iterations) {
        if args.verbose {
            println!("iteration {}", i);
        }
        if !sol.is_finished(&prob) {
            let unfinished_net = prob
                .nets
                .iter()
                .filter(|n| !sol.paths.keys().contains(*n))
                .next()
                .expect("There should be at least one unfinished net");
            if args.verbose {
                println!("unfinished net {}", unfinished_net);
            }
            if let Some((conflicting_net, conflicting_edge)) =
                find_conflicting_net(unfinished_net, &mut sol, &prob, &filtered_nets)
            {
                let old_path = sol.paths.get(&conflicting_net).unwrap();
                if args.verbose {
                    println!(
                        "trying to re-route conflicting net {} with previous path {:?}",
                        conflicting_net, old_path
                    );
                }
                //update path with flipped bend
                if let Some(new_path) = match args.shove_aside_strategy {
                    ShoveAsideStrategy::Dynamic => fix_bend_dynamic(
                        &conflicting_edge,
                        unfinished_net,
                        &prob,
                        old_path.clone(),
                        &mut sol,
                    ),
                    ShoveAsideStrategy::Static => fix_bend(
                        &conflicting_edge,
                        unfinished_net,
                        &prob,
                        old_path.clone(),
                        &mut sol,
                    ),
                } {
                    //update used_edges
                    if args.verbose {
                        println!(
                            "found new path for conflicting net {:?}. Trying to now complete missing net {}",
                            new_path, unfinished_net
                        );
                    }
                    sol.paths.insert(conflicting_net, new_path);
                    let mut new_prob = prob.recreate_for_flips();
                    let mut new_pq = PriorityQueue::new();
                    let gp = new_prob.reinit_grid_point(unfinished_net);
                    new_pq.push(gp);
                    while process_pq(&mut new_pq, &mut new_prob, &mut sol) {}
                    if args.verbose {
                        println!("resulting solution {:?}", sol);
                    }
                } else {
                    filtered_nets.insert(conflicting_net);
                }
            } else {
                break;
            }
        } else {
            if args.verbose {
                println!("sol finished");
            }
            break;
        }
    }
    println!("{}", sol.to_json(&prob, start.elapsed().as_secs_f64()));
}
