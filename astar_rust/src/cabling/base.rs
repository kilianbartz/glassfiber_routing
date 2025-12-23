use crate::cabling::astar::CablingProblem;
///This module contains primitive basic data structures to model a cabling problem.
/// Included are
/// - Terminal, which is basically a pair (i32, i32),
/// - Edge (start=(i32, i32), (i32, i32))
/// - manhattan_dist((i32, i32), (i32, i32))
/// - Net(terminal, terminal)
use serde::Serialize;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct Terminal(pub i32, pub i32);

impl fmt::Display for Terminal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}
impl Terminal {
    pub fn from_pair(pair: (i32, i32)) -> Self {
        Self(pair.0, pair.1)
    }
    ///Computes the manhattan distance between two Terminals
    pub fn dist(&self, other: &Self) -> i32 {
        manhattan_dist(self.into(), other.into())
    }
    ///Outputs a Terminal's terminal number (see Figure 2, documentation pdf)
    pub fn to_number(&self, grid_size: i32) -> i32 {
        self.0 + grid_size * self.1 + 1
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EdgeDirection {
    Horizontal,
    Vertical,
}

#[derive(Copy, Clone, Debug, Serialize)]
pub struct Edge(pub (i32, i32), pub (i32, i32));

impl Edge {
    pub fn direction(&self) -> EdgeDirection {
        //x coord stays the same
        if self.0.0 == self.1.0 {
            return EdgeDirection::Vertical;
        }
        EdgeDirection::Horizontal
    }
    pub fn relocate_to_coord(old: &Self, coord: i32) -> Self {
        //relocates an edge to a new coord. If edge is horizontal, this is the y-level, for a vert it is the x-coord
        match old.direction() {
            EdgeDirection::Horizontal => Self((old.0.0, coord), (old.1.0, coord)),
            EdgeDirection::Vertical => Self((coord, old.0.1), (coord, old.1.1)),
        }
    }
    pub fn relocate_to_point(old: &Self, point: (i32, i32)) -> Self {
        match old.direction() {
            EdgeDirection::Vertical => Self::relocate_to_coord(old, point.0),
            EdgeDirection::Horizontal => Self::relocate_to_coord(old, point.1),
        }
    }
}

/// Outputs an edge's canonical form (a, b) with a <= b. This is used for comparisons and ensures that
/// Edge(c, d) = Edge(d, c)
fn normalize_edge(a: (i32, i32), b: (i32, i32)) -> ((i32, i32), (i32, i32)) {
    // The simplest way to define a canonical form is to
    // compare the two tuples and put the "smaller" one first.
    // Rust's default tuple comparison is lexicographical, which works perfectly here.
    if a <= b {
        (a, b) // (smaller, larger)
    } else {
        (b, a) // (smaller, larger)
    }
}
impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        // Get the two points from self
        let p1_self = self.0;
        let p2_self = self.1;

        // Get the two points from other
        let p1_other = other.0;
        let p2_other = other.1;

        // An edge self is equal to an edge other if:
        // Case 1: The points are in the same order
        // (p1_self == p1_other) AND (p2_self == p2_other)
        let same_order = (p1_self == p1_other) && (p2_self == p2_other);

        // Case 2: The points are in the reverse order
        // (p1_self == p2_other) AND (p2_self == p1_other)
        let reverse_order = (p1_self == p2_other) && (p2_self == p1_other);

        same_order || reverse_order

        // Alternative (using the canonical form for comparison, good for consistency):
        /*
        normalize_edge(p1_self, p2_self) == normalize_edge(p1_other, p2_other)
        */
    }
}

// 2. Implementation of Eq (requires PartialEq and confirms reflexivity)
impl Eq for Edge {}

// 3. Implementation of Hash for use in HashMaps/HashSets
impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // To ensure that Edge((a, b), (c, d)) hashes the same as Edge((c, d), (a, b)),
        // we **must** hash the canonical (normalized) representation of the edge.

        // Get the two points
        let p1 = self.0;
        let p2 = self.1;

        // Normalize the edge
        let normalized = normalize_edge(p1, p2);

        // Hash the normalized parts
        normalized.0.hash(state);
        normalized.1.hash(state);
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[[{}, {}], [{}, {}]]",
            self.0.0, self.0.1, self.1.0, self.1.1
        )
    }
}

impl Into<(i32, i32)> for Terminal {
    fn into(self) -> (i32, i32) {
        (self.0, self.1)
    }
}

impl Into<(i32, i32)> for &Terminal {
    fn into(self) -> (i32, i32) {
        (self.0, self.1)
    }
}

pub fn manhattan_dist(start: (i32, i32), target: (i32, i32)) -> i32 {
    (start.0 - target.0).abs() + (start.1 - target.1).abs()
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct Net(pub Terminal, pub Terminal);

impl fmt::Display for Net {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}
