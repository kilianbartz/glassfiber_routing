///This module contains data structures to represent a cabling problem and solution.
/// Additionally, all data structures and functions related to the A*-search are included.
/// This encompasses
/// - GridPoint, represents a grid cell in the search algorithm
/// - Edge (start=(i32, i32), (i32, i32))
/// - manhattan_dist((i32, i32), (i32, i32))
/// - Net(terminal, terminal)
use super::base::{Edge, Net, Terminal, manhattan_dist};
use itertools::Itertools;
use min_heap::MinHeap;
use serde::Serialize;
use std::cmp::Ordering;
use std::cmp::max;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::{fmt, hash};

#[derive(Clone, Debug)]
pub struct GridPoint {
    x: i32,
    y: i32,
    g: i32,
    h: i32,
    net: Net,
    net_version: i32,
    prior: Option<Rc<GridPoint>>,
}
impl hash::Hash for GridPoint {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.net.hash(state);
        self.net_version.hash(state);
    }
}
impl Into<(i32, i32)> for GridPoint {
    fn into(self) -> (i32, i32) {
        (self.x, self.y)
    }
}
impl Into<(i32, i32)> for &GridPoint {
    fn into(self) -> (i32, i32) {
        (self.x, self.y)
    }
}

impl Ord for GridPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        let f = self.g + self.h;
        let other_f = &other.g + &other.h;
        f.cmp(&other_f)
            .then_with(|| self.h.cmp(&other.h))
            .then_with(|| self.x.cmp(&other.x))
            .then_with(|| self.y.cmp(&other.y))
            .then_with(|| self.net.cmp(&other.net))
            .then_with(|| self.net_version.cmp(&other.net_version))
    }
}

impl PartialOrd for GridPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for GridPoint {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.net == other.net
            && self.net_version == other.net_version
    }
}

impl Eq for GridPoint {}
impl fmt::Display for GridPoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let prior = match self.prior.as_ref() {
            Some(prior) => format!("({}, {})", prior.x, prior.y),
            None => "None".to_string(),
        };
        write!(
            f,
            "(coords: ({}, {}), f: {}, net: {}, nv: {}, prior: {})",
            self.x,
            self.y,
            self.g + self.h,
            self.net,
            self.net_version,
            prior
        )
    }
}
impl GridPoint {
    ///Computes the distance from the GridPoints coordinates to the target terminal of its net
    pub fn dist_to_target(&self) -> i32 {
        manhattan_dist(self.into(), self.net.1.into())
    }
    ///Checks whether this GridPoint completes a net at the target terminal
    pub fn has_arrived(&self) -> bool {
        self.x == self.net.1.0 && self.y == self.net.1.1
    }
    ///Generates an Edge between the GridPoint and its predecessor (if set)
    pub fn edge(&self) -> Option<Edge> {
        //let Some(ref p) = self.prior else {
        //    return None;
        //};
        let p = self.prior.as_ref()?;
        Some(Edge((p.x, p.y), (self.x, self.y)))
    }
    ///Retraces all edges from this GridPoint to the starting terminal.
    /// The resulting vector starts with the youngest edge (GridPoint to predecessor)
    /// and ends with the oldest edge (starting Terminal to second GridPoint)
    pub fn reconstruct_path(&self) -> Vec<Edge> {
        let mut ret = vec![];
        let mut target = self;
        while let Some(e) = target.edge() {
            ret.push(e);
            if let Some(ref p) = target.prior {
                target = p;
            }
        }
        ret
    }
}
///PriorityQueue implemented using a MinHeap. Internally, a HashSet is used to keep track of
/// the elements inside and allow checks like pq.includes(x) in O(1).
#[derive(Debug)]
pub struct PriorityQueue<T: Clone + hash::Hash + Eq + Ord> {
    pq: MinHeap<T>,
    set: HashSet<T>,
}
impl<T: Clone + hash::Hash + Eq + Ord> PriorityQueue<T> {
    ///Adds a single element to the PriorityQueue.
    pub fn push(&mut self, item: T) {
        if !self.set.contains(&item) {
            self.set.insert(item.clone());
            self.pq.push(item);
        }
    }
    ///Adds a vector of elements to the PriorityQueue.
    pub fn push_vec(&mut self, vec: Vec<T>) {
        for item in vec {
            self.push(item);
        }
    }
    ///Retrieves the element with the smallest key from the PriorityQueue.
    pub fn pop(&mut self) -> Option<T> {
        if let Some(ret) = self.pq.pop() {
            self.set.remove(&ret);
            return Some(ret);
        }
        None
    }
    pub fn is_empty(&self) -> bool {
        self.pq.len() == 0
    }
    pub fn new() -> Self {
        Self {
            pq: MinHeap::new(),
            set: HashSet::new(),
        }
    }
}
/// Represents a grid-based cabling problem. net_has_explored and net_versions are used
/// during the solving process to invalidate GridPoints.
#[derive(Debug, Serialize, Clone)]
pub struct CablingProblem {
    pub grid_size: i32,
    pub nets: Vec<Net>,
    #[serde(skip)]
    net_has_explored: HashMap<Net, HashSet<(i32, i32)>>,
    #[serde(skip)]
    net_versions: HashMap<Net, i32>,
}
impl CablingProblem {
    fn output_net(&self, net: &Net) -> String {
        let start = net.0.to_number(self.grid_size);
        let dest = net.1.to_number(self.grid_size);
        format!("({}, {})", start, dest)
    }
    /// Generates the vector of a GridPoints' neighbors, i.e., the neighboring GridPoints reachable
    /// from the current GridPoint. GridPoints which have been explored for this net already are filtered
    /// to prevent infinite loops.
    fn get_gridpoint_neighbors(&self, gp: GridPoint) -> Vec<GridPoint> {
        let potentials = self.get_potential_gridpoint_neighbors(gp);
        potentials
            .into_iter()
            .filter(|gp| match self.net_has_explored.get(&gp.net) {
                Some(already_explored) => !already_explored.contains(&(gp).into()),
                None => true,
            })
            .collect()
    }
    /// Generates the vector of potential GridPoint neighbors. The generation function only checks
    /// the coordinate bounds, and if the neighboring cell is assigned to a different net's terminal
    /// to filter out illegal cases. The GridPoints' heuristics values include a reward of 10
    /// if no bend occurs from the current GridPoint to the neighbor to heuristically implement
    /// bend minimization.
    fn get_potential_gridpoint_neighbors(&self, gp: GridPoint) -> Vec<GridPoint> {
        //this includes bend minimization
        let mut ret: Vec<GridPoint> = Vec::with_capacity(4);
        let this = Rc::new(gp.clone());
        let net_version = gp.net_version;
        //has left neighbor
        if gp.x > 0 && gp.y > 0 && gp.y < self.grid_size - 1 {
            let mut next = GridPoint {
                x: gp.x - 1,
                y: gp.y,
                g: gp.g + 1,
                h: gp.dist_to_target(),
                net: gp.net,
                prior: Some(Rc::clone(&this)),
                net_version,
            };
            let edge = next.edge().unwrap();

            //if both edges exist and have the same direction, award a reward of 10
            if let Some(prior_edge) = gp.edge() {
                if edge.direction() == prior_edge.direction() {
                    next.h = max(0, next.h - 10);
                }
            }
            //prohibit running to terminals of other nets
            let temp_term = Terminal(next.x, next.y);
            let illegal = self
                .nets
                .iter()
                .map(|n| *n != gp.net && (n.0 == temp_term || n.1 == temp_term))
                .reduce(|a, b| a || b)
                .unwrap();
            if !illegal {
                ret.push(next);
            }
        }
        //has right neighbor
        if gp.x < self.grid_size - 1 && gp.y > 0 && gp.y < self.grid_size - 1 {
            let mut next = GridPoint {
                x: gp.x + 1,
                y: gp.y,
                g: gp.g + 1,
                h: gp.dist_to_target(),
                net: gp.net,
                prior: Some(Rc::clone(&this)),
                net_version,
            };
            let edge = next.edge().unwrap();

            //if both edges exist and have the same direction, award a reward of 10
            if let Some(prior_edge) = gp.edge() {
                if edge.direction() == prior_edge.direction() {
                    next.h = max(0, next.h - 10);
                }
            }
            //prohibit running to terminals of other nets
            let temp_term = Terminal(next.x, next.y);
            let illegal = self
                .nets
                .iter()
                .map(|n| *n != gp.net && (n.0 == temp_term || n.1 == temp_term))
                .reduce(|a, b| a || b)
                .unwrap();
            if !illegal {
                ret.push(next);
            }
        }
        //has down neighbor
        if gp.y > 0 && gp.x > 0 && gp.x < self.grid_size - 1 {
            let mut next = GridPoint {
                x: gp.x,
                y: gp.y - 1,
                g: gp.g + 1,
                h: gp.dist_to_target(),
                net: gp.net,
                prior: Some(Rc::clone(&this)),
                net_version,
            };
            let edge = next.edge().unwrap();

            //if both edges exist and have the same direction, award a reward of 10
            if let Some(prior_edge) = gp.edge() {
                if edge.direction() == prior_edge.direction() {
                    next.h = max(0, next.h - 10);
                }
            }
            //prohibit running to terminals of other nets
            let temp_term = Terminal(next.x, next.y);
            let illegal = self
                .nets
                .iter()
                .map(|n| *n != gp.net && (n.0 == temp_term || n.1 == temp_term))
                .reduce(|a, b| a || b)
                .unwrap();
            if !illegal {
                ret.push(next);
            }
        }
        //has up neighbor
        if gp.y < self.grid_size - 1 && gp.x > 0 && gp.x < self.grid_size - 1 {
            let mut next = GridPoint {
                x: gp.x,
                y: gp.y + 1,
                g: gp.g + 1,
                h: gp.dist_to_target(),
                net: gp.net,
                prior: Some(Rc::clone(&this)),
                net_version,
            };
            let edge = next.edge().unwrap();

            //if both edges exist and have the same direction, award a reward of 10
            if let Some(prior_edge) = gp.edge() {
                if edge.direction() == prior_edge.direction() {
                    next.h = max(0, next.h - 10);
                }
            }
            //prohibit running to terminals of other nets
            let temp_term = Terminal(next.x, next.y);
            let illegal = self
                .nets
                .iter()
                .map(|n| *n != gp.net && (n.0 == temp_term || n.1 == temp_term))
                .reduce(|a, b| a || b)
                .unwrap();
            if !illegal {
                ret.push(next);
            }
        }
        ret
    }
    ///Generates the GridPoint, representing the starting terminal, for all nets of a cabling problem.
    pub fn init_grid_points(&mut self) -> Vec<GridPoint> {
        let mut ret = Vec::with_capacity(self.nets.len());
        for net in self.nets.iter() {
            let st_term = net.0;
            let dest_term = net.1;
            ret.push(GridPoint {
                x: st_term.0,
                y: st_term.1,
                g: 0,
                h: st_term.dist(&dest_term),
                net: *net,
                prior: None,
                net_version: 0,
            });
            self.net_versions.insert(*net, 0);
        }
        ret
    }
    /// This function is used as part of solution optimization and generates the starting GridPoint
    /// for a singular net that has not been solved yet.
    pub fn reinit_grid_point(&mut self, net: &Net) -> GridPoint {
        let st_term = net.0;
        let dest_term = net.1;
        let net_version = self.net_versions.get(&net).unwrap() + 1;
        self.net_versions.insert(*net, net_version);
        GridPoint {
            x: st_term.0,
            y: st_term.1,
            g: 0,
            h: st_term.dist(&dest_term),
            net: *net,
            prior: None,
            net_version,
        }
    }
    /// Generates a Terminal from a terminal number
    pub fn terminal_from_number(grid_size: i32, number: i32) -> Terminal {
        let x = (number - 1) % grid_size;
        let y = (number - 1) / grid_size;
        Terminal(x, y)
    }
    /// Used as part of Terminal validation, as valid terminals have to lie at the border of the grid
    fn is_at_edge(grid_size: i32, coord: i32) -> bool {
        if coord == 0 || coord == grid_size - 1 {
            return true;
        }
        false
    }
    fn is_valid_terminal(grid_size: i32, coords: (i32, i32)) -> bool {
        Self::is_at_edge(grid_size, coords.0) || Self::is_at_edge(grid_size, coords.1)
    }
    pub fn new(grid_size: i32, nets_as_terminal_numbers: Vec<(i32, i32)>) -> Self {
        let mut nets = vec![];
        for (x, y) in &nets_as_terminal_numbers {
            let t1 = Self::terminal_from_number(grid_size, *x);
            let t2 = Self::terminal_from_number(grid_size, *y);
            //test if these are valid terminals (edge of grid)
            assert!(Self::is_valid_terminal(grid_size, t1.into()));
            assert!(Self::is_valid_terminal(grid_size, t2.into()));
            nets.push(Net(t1, t2));
        }
        Self {
            grid_size,
            nets,
            net_has_explored: HashMap::new(),
            net_versions: HashMap::new(),
        }
    }
    /// Used in solution optimization and resets the net_has_explored map to allow for new exploration.
    pub fn recreate_for_flips(&self) -> Self {
        let mut ret = self.clone();
        ret.net_has_explored = HashMap::new();
        ret
    }
}

/// This struct is used for serialization of a solution.
#[derive(Debug, Serialize)]
pub struct SolutionSerialized {
    r#type: String,
    solved: bool,
    paths: HashMap<String, Vec<Edge>>,
    missing: Vec<String>,
    time: f64,
    grid_size: i32,
}

impl SolutionSerialized {
    fn new(sol: &Solution, prob: &CablingProblem, time: f64) -> Self {
        let mut paths: HashMap<String, Vec<Edge>> = HashMap::new();
        for (net, path) in &sol.paths {
            let net_name = prob.output_net(net);
            paths.insert(net_name, path.clone());
        }
        let missing = prob
            .nets
            .iter()
            .filter(|n| !sol.paths.keys().contains(n))
            .map(|n| prob.output_net(n))
            .collect();
        Self {
            r#type: "astar".to_string(),
            solved: sol.is_finished(prob),
            paths,
            time,
            missing,
            grid_size: prob.grid_size,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub finished_nets: HashSet<Net>,
    pub used_edges: HashSet<Edge>,
    pub paths: BTreeMap<Net, Vec<Edge>>,
}

impl Solution {
    pub fn new() -> Self {
        Self {
            finished_nets: HashSet::new(),
            used_edges: HashSet::new(),
            paths: BTreeMap::new(),
        }
    }
    pub fn to_json(&self, prob: &CablingProblem, time: f64) -> String {
        let result = SolutionSerialized::new(self, prob, time);
        serde_json::to_string(&result).unwrap()
    }
    pub fn is_finished(&self, prob: &CablingProblem) -> bool {
        prob.nets
            .iter()
            .map(|n| self.paths.keys().contains(n))
            .reduce(|a, b| a && b)
            .unwrap()
    }
}
/// This function performs a single A*-search step to try solving the problem prob.
/// It takes
/// - the current state of the PriorityQueue (retrieve next GridPoint, append neighbors)
/// - the cabling problem prob
/// - the solution sol.
/// The returned boolean represents whether search should continue after the step (true), or
/// if it can be terminated (false). Termination occurs when the problem has been fully solved
/// or no GridPoints to explore remain.
pub fn process_pq(
    pq: &mut PriorityQueue<GridPoint>,
    prob: &mut CablingProblem,
    sol: &mut Solution,
) -> bool {
    let Some(gp) = pq.pop() else {
        return false;
    };
    //println!("{}", gp);
    //check if gp should be skipped
    //this is the case if either the edge is already taken or the net is already completed or if the net version is invalid (see below)
    if let Some(edge) = gp.edge()
        && sol.used_edges.contains(&edge)
    {
        return true;
    }
    if sol.finished_nets.contains(&gp.net) {
        return true;
    }
    //net version invalid: this gridpoint is from a previous exploration phase where edges could have been taken which are illegal now
    if gp.net_version != *prob.net_versions.get(&gp.net).unwrap() {
        return true;
    }
    prob.net_has_explored
        .entry(gp.net)
        .or_insert_with(HashSet::new)
        .insert((&gp).into());
    //net is finished
    if gp.has_arrived() {
        let path = gp.reconstruct_path();

        //first ensure that the path is still legal, i.e., that no other path which has finished in the meantime occupies one of the edges
        let illegal = path
            .iter()
            .map(|e| sol.used_edges.contains(e))
            .reduce(|a, b| a | b);
        if illegal.unwrap() {
            //start at beginning again
            pq.push(prob.reinit_grid_point(&gp.net));
            prob.net_has_explored.remove(&gp.net);
            return true;
        }

        for edge in &path {
            sol.used_edges.insert(*edge);
            //println!("{}", edge);
        }
        sol.paths.insert(gp.net, path);
        sol.finished_nets.insert(gp.net);
        //check if all nets are finished
        if sol.is_finished(prob) {
            return false;
        }
    } else {
        pq.push_vec(prob.get_gridpoint_neighbors(gp));
    }
    true
}

/// Used as part of solution optimization. Takes in a problematic net p, which is currently missing from the solution,
/// and tries to find a conflicting net c and edge e, where the c has been solved and is not part of the filtered_nets,
/// and e is currently assigned to c, but assigning it to p would advance p's routing.
pub fn find_conflicting_net(
    problematic_net: &Net,
    sol: &mut Solution,
    p: &CablingProblem,
    filtered_nets: &HashSet<Net>,
) -> Option<(Net, Edge)> {
    //println!("find_conflicting_net");
    let mut prob = p.clone();
    let start = prob.reinit_grid_point(problematic_net);
    let mut pq = PriorityQueue::new();
    pq.push(start);
    let mut problematic_edges = HashSet::new();
    let mut explored: HashSet<(i32, i32)> = HashSet::new();
    //iterate over potential neighbors and form edges to them
    //the loop below finds all edges which the problematic_net could use to come further
    while !pq.is_empty() {
        let gp = pq.pop().unwrap();
        //println!("{}", gp);
        if explored.contains(&(&gp).into()) {
            continue;
        }
        explored.insert((&gp).into());
        let potentials = prob.get_potential_gridpoint_neighbors(gp);
        for neighbor in potentials.into_iter() {
            let edge = neighbor.edge().unwrap();
            if sol.used_edges.contains(&edge) {
                problematic_edges.insert(edge);
            } else {
                pq.push(neighbor);
            }
        }
    }
    assert!(!problematic_edges.is_empty());
    //the paths are iterated from back (target terminal) to front which ensures that
    for (net, path) in &sol.paths {
        for e2 in path {
            if filtered_nets.contains(net) {
                continue;
            }
            if problematic_edges.contains(e2) {
                return Some((*net, *e2));
            }
        }
    }
    None
}
/// Used as part of dynamic bend flips. Tries to connect the starting grid cell (modelled as a Terminal)
/// to the end grid cell (modelled as a Terminal) using A*-search.
fn route_new_subnet(
    entire_net: &Net,
    start: Terminal,
    end: Terminal,
    problem: &CablingProblem,
    sol: &mut Solution,
) -> Option<Vec<Edge>> {
    let new_net = Net(start, end);
    let mut prob = problem.clone();
    let mut ret_sol = sol.clone();
    prob.nets = vec![new_net];
    prob.net_versions.insert(new_net, 0);
    //cases we want to prevent:
    //following old path --> achieved by not removing old path from used_edges
    ret_sol.paths.remove(entire_net);
    let gp = prob.reinit_grid_point(&new_net);
    let mut pq = PriorityQueue::new();
    pq.push(gp);
    while process_pq(&mut pq, &mut prob, &mut ret_sol) {}
    if ret_sol.is_finished(&prob) {
        return Some(ret_sol.paths.get(&new_net).unwrap().clone());
    }
    None
}

/// Given the path of a conflicting net and the conflicting edge (obtained through find_conflicting_net),
/// this function computes the start index (every edge before this can be carried over from previous path and does not have to be changed),
/// the end index (every edge after this edge can be carried over from previous path and does not have to be changed),
/// the edge w and the vector of edges E_r.
/// The functioning principle is explained in Figure 5 of the documentation pdf.
fn find_bend(conflicting_edge: &Edge, path: Vec<Edge>) -> Option<(usize, usize, Edge, Vec<Edge>)> {
    let mut bfe = None;
    let mut edges_to_put_first: Vec<Edge> = Vec::new();
    let mut end_index = path.len() - 1;
    let mut start_index = 0;
    for (index, edge) in path.iter().enumerate() {
        if edge == conflicting_edge {
            edges_to_put_first.push(*edge);
            end_index = index;
            continue;
        }
        if edges_to_put_first.len() > 0 {
            if edge.direction() != conflicting_edge.direction() {
                bfe = Some(*edge);
                start_index = index;
                break;
            } else {
                edges_to_put_first.push(*edge);
            }
        }
    }
    //if there is no bend before the conflicting edge, flipping cannot solve the problem
    let Some(bend_first_edge) = bfe else {
        return None;
    };
    Some((start_index, end_index, bend_first_edge, edges_to_put_first))
}
/// This function implements dynamic bend flipping as solution optimization. The functioning principle
/// is explained in Section 5.5 of the documentation pdf.
pub fn fix_bend_dynamic(
    conflicting_edge: &Edge,
    net_to_solve: &Net,
    problem: &CablingProblem,
    path: Vec<Edge>,
    sol: &mut Solution,
) -> Option<Vec<Edge>> {
    if let Some(sol) = fix_bend(conflicting_edge, net_to_solve, problem, path.clone(), sol) {
        return Some(sol);
    }
    //find bend
    //println!("conflicting edge: {}", conflicting_edge);
    let (start_index, end_index, bend_first_edge, _) = find_bend(conflicting_edge, path.clone())?;
    //check that it is legal to mirror bend_first_edge to the end of the path
    let mut new_path = (&path)[0..end_index].to_owned();
    let relocated_bend_first = Edge::relocate_to_point(&bend_first_edge, conflicting_edge.1);
    if is_edge_illegal(&relocated_bend_first, problem, &sol) {
        //println!("relocating bfe is illegal");
        return None;
    }
    // it is also illegal to relocate terminal edge
    let temp_term1 = Terminal::from_pair(bend_first_edge.0);
    let temp_term2 = Terminal::from_pair(bend_first_edge.1);
    let is_terminal_edge = problem
        .nets
        .iter()
        .map(|n| n.0 == temp_term1 || n.0 == temp_term2 || n.1 == temp_term2 || n.1 == temp_term1)
        .reduce(|a, b| a || b)
        .unwrap();
    if is_terminal_edge {
        //println!("relocating bfe is illegal");
        return None;
    }
    //println!("reLocating bfe {} to {}", bend_first_edge, relocated_bend_first);
    new_path.push(*(&relocated_bend_first));
    let new_sub_path = route_new_subnet(
        net_to_solve,
        Terminal::from_pair(bend_first_edge.0),
        Terminal::from_pair(relocated_bend_first.0),
        problem,
        &mut sol.clone(),
    )?;
    for edge in &new_sub_path {
        new_path.push(*edge);
    }
    new_path.extend_from_slice(&path[start_index + 1..]);
    //update used_edges
    for edge in &path {
        sol.used_edges.remove(edge);
    }
    for edge in &new_path {
        sol.used_edges.insert(*edge);
    }
    Some(new_path)
}

/// This function implements static bend flipping as solution optimization. The functioning principle
/// is explained in Section 5.5 of the documentation pdf.
pub fn fix_bend(
    conflicting_edge: &Edge,
    _: &Net,
    problem: &CablingProblem,
    path: Vec<Edge>,
    sol: &mut Solution,
) -> Option<Vec<Edge>> {
    //find bend
    let (start_index, end_index, bend_first_edge, edges_to_put_first) =
        find_bend(conflicting_edge, path.clone())?;
    //the relocated edges have to all be legal. If not, return the old path again
    let mut ret = (&path)[0..end_index].to_owned();
    let relocated_bend_first = Edge::relocate_to_point(&bend_first_edge, conflicting_edge.1);
    if is_edge_illegal(&relocated_bend_first, problem, sol) {
        return None;
    }
    ret.push(*(&relocated_bend_first));
    for edge in &edges_to_put_first {
        let new_edge = Edge::relocate_to_point(edge, relocated_bend_first.0);
        ret.push(new_edge);
        if is_edge_illegal(&new_edge, problem, sol) {
            return None;
        }
    }
    ret.extend_from_slice(&path[start_index + 1..]);
    //update used_edges
    for edge in &path {
        sol.used_edges.remove(edge);
    }
    for edge in &ret {
        sol.used_edges.insert(*edge);
    }
    Some(ret)
}

/// This function is used to check whether an edge may be relocated to a new coordinate
fn is_edge_illegal(edge: &Edge, prob: &CablingProblem, sol: &Solution) -> bool {
    let temp_term1 = Terminal::from_pair(edge.0);
    let temp_term2 = Terminal::from_pair(edge.1);
    let illegal_due_to_terminals = prob
        .nets
        .iter()
        .map(|n| n.0 == temp_term1 || n.0 == temp_term2 || n.1 == temp_term2 || n.1 == temp_term1)
        .reduce(|a, b| a || b)
        .unwrap();
    let illegal_due_to_fixed_edges = sol.used_edges.contains(edge);
    illegal_due_to_terminals || illegal_due_to_fixed_edges
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabling;
    #[test]
    fn test_fix_bend() {
        let prob = CablingProblem::new(5, vec![(2, 20)]);
        let path = vec![
            Edge((3, 3), (4, 3)),
            Edge((2, 3), (3, 3)),
            Edge((1, 3), (2, 3)),
            Edge((1, 2), (1, 3)),
            Edge((1, 1), (1, 2)),
            Edge((1, 0), (1, 1)),
        ];
        let problematic_edge = Edge((2, 3), (3, 3));
        let bend_first_edge = Edge((1, 2), (1, 3));
        let mut sol = Solution::new();
        let e = Edge((1, 3), (2, 3));
        let e2 = Edge((2, 3), (3, 3));
        assert_eq!(
            Edge::relocate_to_coord(&bend_first_edge, 3),
            Edge((3, 2), (3, 3))
        );
        assert_eq!(
            Edge::relocate_to_coord(&e, bend_first_edge.0.1),
            Edge((1, 2), (2, 2))
        );
        assert_eq!(
            Edge::relocate_to_coord(&e2, bend_first_edge.0.1),
            Edge((2, 2), (3, 2))
        );
        let net = prob.nets.iter().next().unwrap();
        let altered_path = fix_bend(&problematic_edge, net, &prob, path, &mut sol).unwrap();
        let altered_path_correct = vec![
            Edge((3, 3), (4, 3)),
            Edge((3, 2), (3, 3)),
            Edge((2, 2), (3, 2)),
            Edge((1, 2), (2, 2)),
            Edge((1, 1), (1, 2)),
            Edge((1, 0), (1, 1)),
        ];
        for i in 0..altered_path_correct.len() {
            assert_eq!(altered_path_correct[i], altered_path[i]);
        }
    }
    #[test]
    fn test_fix_bend_dynamic() {
        let prob = CablingProblem::new(5, vec![(2, 20)]);
        let path = vec![
            Edge((3, 3), (4, 3)),
            Edge((2, 3), (3, 3)),
            Edge((1, 3), (2, 3)),
            Edge((1, 2), (1, 3)),
            Edge((1, 1), (1, 2)),
            Edge((1, 0), (1, 1)),
        ];
        let problematic_edge = Edge((2, 3), (3, 3));
        let bend_first_edge = Edge((1, 2), (1, 3));
        let mut sol = Solution::new();
        let e = Edge((1, 3), (2, 3));
        let e2 = Edge((2, 3), (3, 3));
        assert_eq!(
            Edge::relocate_to_coord(&bend_first_edge, 3),
            Edge((3, 2), (3, 3))
        );
        assert_eq!(
            Edge::relocate_to_coord(&e, bend_first_edge.0.1),
            Edge((1, 2), (2, 2))
        );
        assert_eq!(
            Edge::relocate_to_coord(&e2, bend_first_edge.0.1),
            Edge((2, 2), (3, 2))
        );
        let net = prob.nets.iter().next().unwrap();
        let altered_path = fix_bend_dynamic(&problematic_edge, net, &prob, path, &mut sol).unwrap();
        let altered_path_correct = vec![
            Edge((3, 3), (4, 3)),
            Edge((3, 2), (3, 3)),
            Edge((2, 2), (3, 2)),
            Edge((1, 2), (2, 2)),
            Edge((1, 1), (1, 2)),
            Edge((1, 0), (1, 1)),
        ];
        for i in 0..altered_path_correct.len() {
            assert_eq!(altered_path_correct[i], altered_path[i]);
        }
    }
}
