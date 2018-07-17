use std::cmp::{max, min};
use std::fmt;
use std::str::FromStr;
extern crate permutohedron;
use permutohedron::Heap;

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Edge {
    Red,
    Green,
    None,
}

impl fmt::Debug for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Edge::Red => write!(f, "R"),
            Edge::Green => write!(f, "G"),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
pub struct Graph {
    pub vertices: usize,
    pub edges: Vec<Edge>, //top right of adjacency matrix TODO: consider both for faster accesses
}

impl Graph {
    pub fn get_edge(&self, n: usize, m: usize) -> &Edge {
        let n1 = min(n, m);
        let m1 = max(n, m);
        if n1 == m1 {
            return &Edge::None;
        } else {
            &self.edges[(m1 * m1 - m1) / 2 + n1]
        }
    }
    //creates a new complete graph with n vertices
    pub fn new(n: usize) -> Graph {
        let num_edges = n * (n - 1) / 2;
        let mut edges = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            edges.push(Edge::Red);
        }
        Graph { vertices: n, edges }
    }
    pub fn generate_children(&self) -> Vec<Graph> {
        let mut out = vec![];
        for mut n in 0..(2usize).pow(self.vertices as u32) {
            let mut child = self.clone();
            child.vertices += 1;
            for _ in 0..self.vertices {
                match n & 1 {
                    0 => child.edges.push(Edge::Red),
                    1 => child.edges.push(Edge::Green),
                    _ => unreachable!(),
                }
                n >>= 1;
            }
            out.push(child);
        }

        out
    }
}

pub enum GraphParseError {
    BadNumEdges(usize),
    BadChar(char),
}

impl fmt::Debug for GraphParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GraphParseError::BadNumEdges(n) => write!(
                f,
                "Error parsing the graph, expected triangular number of edges, got {}",
                n
            ),
            GraphParseError::BadChar(c) => {
                write!(f, "Error parsing the graph, expected R or G, found {}", c)
            }
        }
    }
}

impl FromStr for Graph {
    type Err = GraphParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let possible_lens = [1usize, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105];
        match possible_lens.binary_search(&s.len()) {
            Ok(n) => {
                let mut edges = vec![];
                for c in s.chars() {
                    match c {
                        'R' => edges.push(Edge::Red),
                        'G' => edges.push(Edge::Green),
                        bad_char => return Err(GraphParseError::BadChar(bad_char)),
                    }
                }
                Ok(Graph {
                    vertices: n + 2,
                    edges: edges,
                })
            }
            Err(e) => Err(GraphParseError::BadNumEdges(s.len())),
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graph {{ vertices: {} edges: ", self.vertices);
        for e in &self.edges {
            write!(f, "{:?}", e);
        }
        write!(f, " }}")
    }
}

#[derive(Debug, Clone)]
pub struct LabeledGraph {
    pub graph: Graph,
    pub labeling: Labeling,
}



/*
    It might make sense to write one isomorphism checking function that runs on two graphs,
    but for faster checks among a list, create another function that runs through the permutations once
    and then checks against each other

    fn iso_dedup(g: &LabeledGraph, others: &mut Vec<LabeledGraph>)
    it would run in place and basically run retain once for each permutation to check
    to be used with pop / retain if equal loop

    This may or may not be any faster than running though it all again and again
    It would have the downside of not being able to quit early if it finds an iso
    Although I suppose that doesn't matter, because it has to run all the way until there are
    none left or its done them all either way
    They get removed immediately on being found so that is not a downside
*/
impl LabeledGraph {
    fn bin(list: &Vec<u16>) -> Vec<Vec<usize>> {
        let mut temp: Vec<(u16, Vec<usize>)> = vec![];
        for (n, v) in list.iter().enumerate() {
            let mut found = false;
            for (label, bin) in temp.iter_mut() {
                if v == label {
                    bin.push(n);
                    found = true;
                    break;
                }
            }
            if !found {
                temp.push((*v, vec![n]));
            }
        }
        //temp.retain(|(label, bin)| bin.len() > 1);
        temp.sort_unstable_by_key(|(label, bin)| *label);
        temp.into_iter().map(|(label, bin)| bin).collect()
    }
    fn collapse(list: &Vec<Vec<usize>>, out: &mut Vec<usize>) {
        out.clear();
        for i in list.iter() {
            for j in i.iter() {
                out.push(*j);
            }
        }
    }
    /*
        Idea is to permute all vertices in g to see if one matches h
        First we put them into bins according to their labels
        Then we only permute bins with more than one element
    */
    fn rec_iso_check(
        depth: usize,
        max_depth: usize,
        orig_verts_g: &Vec<Vec<usize>>, //original bins, with more than one element
        verts_g: &mut Vec<Vec<usize>>,  //the current permutation hypothesis we're working on
        collapsed_verts_h: &Vec<usize>, //the vertices of h collapsed into a single vector
        collapsed_verts_g: &mut Vec<usize>,
        g: &LabeledGraph,
        h: &LabeledGraph,
    ) -> bool {
        if depth == max_depth {
            LabeledGraph::collapse(verts_g, collapsed_verts_g);
            for i in 0..collapsed_verts_h.len() - 1 {
                for j in 0..collapsed_verts_h.len() {
                    if     g.graph.get_edge(collapsed_verts_g[i], collapsed_verts_g[j])
                        != h.graph.get_edge(collapsed_verts_h[i], collapsed_verts_h[j]) {
                        return false;
                    }
                }
            }
            return true;
        }else {
            let mut c = orig_verts_g[depth].clone();
            let heap = Heap::new(&mut c);
            for i in heap {
                verts_g[depth] = i;
                if LabeledGraph::rec_iso_check(
                    depth + 1,
                    max_depth,
                    orig_verts_g,
                    verts_g,
                     collapsed_verts_h,
                    collapsed_verts_g,
                    g,
                    h
                ) {
                    return true;
                }
            }
            return false;
        }
    }
    /*
        This function does the setup of creating the bins, keeping track of the original, etc
    */
    pub fn is_color_iso(g: &LabeledGraph, h: &LabeledGraph) -> bool {
        if g.labeling != h.labeling {
            return false;
        } else {
            let orig_g_bins = LabeledGraph::bin(&g.labeling.labels);
            let mut g_bins = orig_g_bins.clone();
            if g_bins.len() > 0 {
                let h_bins = LabeledGraph::bin(&h.labeling.labels);
                let mut collapsed_verts_h = vec![];
                LabeledGraph::collapse(&h_bins, &mut collapsed_verts_h);
                let mut collapsed_verts_g = vec![];
                let max_depth = g_bins.len();

                LabeledGraph::rec_iso_check(
                    0,
                    max_depth,
                    &orig_g_bins,
                    &mut g_bins,
                    &collapsed_verts_h,
                    &mut collapsed_verts_g,
                    g,
                    h
                )

            }else{
                return true;
            }
        }
    }
}

impl Ord for LabeledGraph {
    fn cmp(&self, other: &Self) -> Ordering {
        self.labeling.cmp(&other.labeling)
    }
}
impl PartialOrd for LabeledGraph {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.labeling.cmp(&other.labeling))
    }
}
impl PartialEq for LabeledGraph {
    //Isomorphism algorithm, taking labels into account

    fn eq(&self, other: &LabeledGraph) -> bool {
        LabeledGraph::is_color_iso(self, other)
    }
}
impl Eq for LabeledGraph {}

#[derive(PartialEq, Eq, Debug, Ord, PartialOrd, Copy, Clone)]
pub enum LabelingVariant {
    Neighbors,
    Neighbors2,
    K3,
}

/*
        variant reflects the algorithm that produced the label
        hash is an encoding of set of labels for easy non ordered comparison
    */
#[derive(Eq, Debug, Clone)]
pub struct Labeling {
    variant: LabelingVariant,
    labels: Vec<u16>,
    hash: u64,
    pub complexity: u64,
}

impl PartialEq for Labeling {
    fn eq(&self, other: &Self) -> bool {
        self.variant == other.variant && self.hash == other.hash
    }
}

use std::cmp::{Ord, Ordering};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl Ord for Labeling {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.variant.cmp(&other.variant) {
            Ordering::Equal => match self.complexity.cmp(&other.complexity).reverse() {
                Ordering::Equal => self.hash.cmp(&other.hash),
                r => r,
            },
            r => r,
        }
    }
}

impl PartialOrd for Labeling {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Labeling {
    fn calc_comp<T: Eq>(list: Vec<T>) -> u64 {
        if list.len() < 2 {
            return 1;
        } else {
            let mut complexity = 1;
            let mut current = &list[0];
            let mut streak = 1;
            for i in list.iter().skip(1) {
                if i == current {
                    streak += 1;
                } else {
                    current = i;
                    complexity *= factorial(streak);
                    streak = 1;
                }
            }
            complexity *= factorial(streak);
            complexity
        }
    }
    //this assigns to each vertex the number of red edges incident to it
    pub fn neighbors(g: &Graph) -> Labeling {
        let mut labels = vec![];
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += 1,
                    _ => (),
                }
            }
            labels.push(red_edges);
        }
        let mut copy = labels.clone();
        copy.sort();

        let mut hasher = DefaultHasher::new();
        copy.hash(&mut hasher);

        Labeling {
            variant: LabelingVariant::Neighbors,
            labels,
            hash: hasher.finish(),
            complexity: Labeling::calc_comp(copy),
        }
    }
    pub fn neighbors2(g: &Graph) -> Labeling {
        let mut labels = Labeling::neighbors(g).labels;
        let mut new_labels = vec![];
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += labels[j],
                    _ => (),
                }
            }
            new_labels.push(red_edges);
        }
        let mut copy = new_labels.clone();
        copy.sort();

        let mut hasher = DefaultHasher::new();
        copy.hash(&mut hasher);
        Labeling {
            variant: LabelingVariant::Neighbors2,
            labels,
            hash: hasher.finish(),
            complexity: Labeling::calc_comp(copy),
        }
    }
    fn k3(g: &Graph) -> Labeling {
        unimplemented!()
    }
}

fn factorial(n: u64) -> u64 {
    if n < 2 {
        return 1;
    } else {
        let mut ans = 1;
        for i in 2..=n {
            ans *= i;
        }
        ans
    }
}
/*
        A list of graphs ready for checking for isomorphisms
        All have the same labeling variant and hash
    */
pub struct Chunk {
    pub graphs: Vec<LabeledGraph>
}

use std;
impl Chunk {
    //filters out duplicate graphs, in the isomorphism sense
    pub fn chunkify(mut v: Vec<LabeledGraph>) -> Vec<Chunk> {
        v.sort();
        let mut out = vec![];
        while let Some(i) = Chunk::find_split(&v) {
            out.push(Chunk {
                graphs: v.split_off(i)
            });
        }
        out.push(Chunk {graphs: v});
        out
    }
    fn find_split(v: &Vec<LabeledGraph>) -> Option<usize> {
        for i in (1..v.len()-1).rev() {
            if v[i].labeling != v[i-1].labeling {
                //println!("found split at {}", i);
                return Some(i);
            }
        }
        return None;
    }

    pub fn dedup(&mut self) {
        let mut cleaned = vec![];
        while let Some(g) = self.graphs.pop() {
            self.graphs.retain(|h| &g != h);
            cleaned.push(g);
        }
        std::mem::swap(&mut cleaned, &mut self.graphs);
    }


    fn collapse(list: &Vec<Vec<usize>>) -> Vec<usize> {
        let mut out = vec![];
        for i in list.iter() {
            for j in i.iter() {
                out.push(*j);
            }
        }
        out
    }

    pub fn dedup2(&mut self) {

    }

    pub fn chunk_dedup_setup(mut self) -> Chunk {

        let mut cleaned = vec![];
        let mut temp: Vec<(LabeledGraph, Vec<usize>)> = self.graphs.into_iter().map(|h| {
            let h_bins = LabeledGraph::bin(&h.labeling.labels);
            let h_bins_collapsed = Chunk::collapse(&h_bins);
            (h, h_bins_collapsed)
        }).collect();
        while let Some((g, mut collapsed_verts)) = temp.pop() {
            let orig_g_bins = LabeledGraph::bin(&g.labeling.labels);
            let mut g_bins = orig_g_bins.clone();
            let max_depth = g_bins.len();
            Chunk::rec_iso_check(
                0,
                max_depth,
                &orig_g_bins,
                &mut g_bins,
                &mut temp,
                &mut collapsed_verts,
                &g
            );
            cleaned.push(g);
        }
        Chunk {
            graphs: cleaned
        }
    }

    fn rec_iso_check(
        depth: usize,
        max_depth: usize,
        orig_verts_g: &Vec<Vec<usize>>, //original bins, with more than one element
        verts_g: &mut Vec<Vec<usize>>,  //the current permutation hypothesis we're working on
        others: &mut Vec<(LabeledGraph, Vec<usize>)>, //the vertices of h collapsed into a single vector
        collapsed_verts_g: &mut Vec<usize>,
        g: &LabeledGraph,
    ){
        if depth == max_depth {
            LabeledGraph::collapse(verts_g, collapsed_verts_g);
            others.retain(|h| {
                for i in 0..collapsed_verts_g.len() - 1 {
                    for j in 0..collapsed_verts_g.len() {
                        if     g.graph.get_edge(collapsed_verts_g[i], collapsed_verts_g[j])
                            != h.0.graph.get_edge(h.1[i], h.1[j]) {
                            return true;
                        }
                    }
                }
                return false;
            });
        }else {
            let mut c = orig_verts_g[depth].clone();
            let heap = Heap::new(&mut c);
            for i in heap {
                verts_g[depth] = i;
                Chunk::rec_iso_check(
                    depth + 1,
                    max_depth,
                    orig_verts_g,
                    verts_g,
                    others,
                    collapsed_verts_g,
                    g
                );
            }
        }
    }

}
