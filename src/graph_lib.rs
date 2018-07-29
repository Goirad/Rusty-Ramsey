use std::cmp::{max, min};
use std::fmt;
use std::str::FromStr;

extern crate permutohedron;

use permutohedron::Heap;
use std::mem;

use std::cmp::{Ord, Ordering};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Edge {
    Red,
    Green,
    None,
}

impl fmt::Debug for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Edge::Red => write!(f, "1"),
            Edge::Green => write!(f, "2"),
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
    pub fn generate_children2(&self, f: &Fn(&Graph) -> bool) -> Vec<Graph> {
        let mut out = vec![];
        let mut temp = self.clone();
        temp.vertices += 1;
        let x = self.vertices * (self.vertices - 1) / 2;
        for mut n in 0..(2usize).pow(self.vertices as u32) {
            for _ in 0..self.vertices {
                match n & 1 {
                    0 => temp.edges.push(Edge::Red),
                    1 => temp.edges.push(Edge::Green),
                    _ => unreachable!(),
                }
                n >>= 1;
            }
            if f(&temp) {
                let mut new = self.clone();
                new.vertices += 1;
                mem::swap(&mut temp, &mut new);
                out.push(new);
            }else{
                temp.edges.truncate(x);
            }
        }

        out
    }
}

impl<'a> IntoIterator for &'a Graph {
    type Item  = Graph;
    type IntoIter = GraphChildrenIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        GraphChildrenIterator {
            g: self,
            index: 0,
            max: (2usize).pow(self.vertices as u32),
        }
    }
}
pub struct GraphChildrenIterator<'a> {
    g: &'a Graph,
    index: usize,
    max: usize,
}

impl<'a> Iterator for GraphChildrenIterator<'a> {
    type Item = Graph;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.max {
            return None;
        }
        let mut n = self.index;
        let mut child = self.g.clone();
        child.vertices += 1;
        for _ in 0..self.g.vertices {
            match n & 1 {
                0 => child.edges.push(Edge::Red),
                1 => child.edges.push(Edge::Green),
                _ => unreachable!(),
            }
            n >>= 1;
        }
        self.index += 1;
        Some(child)
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
            Err(_) => Err(GraphParseError::BadNumEdges(s.len())),
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graph {{ vertices: {} edges: ", self.vertices)?;
        for e in &self.edges {
            write!(f, "{:?}", e)?;
        }
        write!(f, " }}")
    }
}


#[derive(PartialEq, Eq, Debug, Ord, PartialOrd, Copy, Clone)]
pub enum LabelingVariant {
    Neighbors,
    Neighbors2,
    Neighbors3,
    K3,
}

/*
        variant reflects the algorithm that produced the label
        hash is an encoding of set of labels for easy non ordered comparison
    */
#[derive(Debug, Clone)]
pub struct Labeling<T> {
    variant: LabelingVariant,
    labels: Vec<T>,
    hash: u64,   //labels, but sorted
    pub complexity: u64,
}

impl<T: Ord + Eq> PartialEq for Labeling<T> {
    fn eq(&self, other: &Self) -> bool {
        self.variant == other.variant && self.hash == other.hash
    }
}


impl<T: Eq + Ord> Eq for Labeling<T> {}



impl<T: Ord> Ord for Labeling<T> {
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

impl<T: Ord> PartialOrd for Labeling<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Labeling<u8> {
    pub fn neighbors(g: &Graph) -> Labeling<u8> {
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
            complexity: Labeling::calc_comp(&copy),
            hash: hasher.finish(),
        }
    }
}

impl Labeling<(u8, u16)> {
    pub fn neighbors2(g: &Graph) -> Labeling<(u8, u16)> {
        let labels = Labeling::neighbors(g).labels;
        let mut new_labels = vec![];
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += labels[j] as u16,
                    _ => (),
                }
            }
            new_labels.push((labels[i], red_edges));
        }
        let mut copy = new_labels.clone();
        copy.sort();

        let mut hasher = DefaultHasher::new();
        copy.hash(&mut hasher);
        Labeling {
            variant: LabelingVariant::Neighbors2,
            labels: new_labels,
            complexity: Labeling::calc_comp(&copy),
            hash: hasher.finish(),

        }
    }
}

impl Labeling<(u8, u16, u16)> {
    pub fn neighbors3(g: &Graph) -> Labeling<(u8, u16, u16)> {
        let labels = Labeling::neighbors2(g).labels;
        let mut new_labels = vec![];
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += labels[j].1 as u16,
                    _ => (),
                }
            }
            new_labels.push((labels[i].0, labels[i].1, red_edges));
        }
        let mut copy = new_labels.clone();
        copy.sort();

        let mut hasher = DefaultHasher::new();
        copy.hash(&mut hasher);
        Labeling {
            variant: LabelingVariant::Neighbors3,
            labels: new_labels,
            complexity: Labeling::calc_comp(&copy),
            hash: hasher.finish(),

        }
    }
    pub fn neighbors3_eff(g: &Graph) -> Labeling<(u8, u16, u16)> {
        let mut labels = vec![(0u8, 0u16, 0u16); g.vertices];
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += 1,
                    _ => (),
                }
            }
            labels[i].0 = red_edges;
        }
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += labels[j].0 as u16,
                    _ => (),
                }
            }
            labels[i].1 = red_edges;
        }
        for i in 0..g.vertices {
            let mut red_edges = 0;
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => red_edges += labels[j].1 as u16,
                    _ => (),
                }
            }
            labels[i].2 = red_edges;
        }
        let mut copy = labels.clone();
        copy.sort();
        let mut hasher = DefaultHasher::new();
        copy.hash(&mut hasher);

        Labeling {
            variant: LabelingVariant::Neighbors3,
            labels: labels,
            complexity: Labeling::calc_comp(&copy),
            hash: hasher.finish(),
        }
    }
}

impl Labeling<(u16, u8, u16, u16)> {
    //count number of k3s
    pub fn k3(g: &Graph) -> Labeling<(u16, u8, u16, u16)> {
        let old_labels = Labeling::neighbors3_eff(g).labels;
        let mut new_labels = vec![];
        for i in 0..g.vertices {
            let mut connected = vec![];
            for j in 0..g.vertices {
                match *g.get_edge(i, j) {
                    Edge::Red => connected.push(j),
                    _ => (),
                }
            }
            let mut k3s = 0u16;
            if connected.len() > 1 {
                for j in 0..connected.len()-1 {
                    for k in j..connected.len(){
                        match *g.get_edge(connected[j], connected[k]) {
                            Edge::Red => k3s += 1,
                            _ => (),
                        }
                    }
                }
            }

            let x = (k3s, old_labels[i].0, old_labels[i].1, old_labels[i].2);
            //println!("{:?}", x);
            new_labels.push(x);

        }
        let mut copy = new_labels.clone();
        copy.sort();
        let mut hasher = DefaultHasher::new();
        copy.hash(&mut hasher);

        Labeling {
            variant: LabelingVariant::K3,
            labels: new_labels,
            complexity: Labeling::calc_comp(&copy),
            hash: hasher.finish(),

        }
    }
}

impl<T: Eq> Labeling<T> {
    fn calc_comp(list: &Vec<T>) -> u64 {
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
}


#[derive(Debug, Clone)]
pub struct LabeledGraph<T> {
    pub graph: Graph,
    pub labeling: Labeling<T>,
}


impl<T: Ord + Copy> LabeledGraph<T> {
    fn bin(list: &Vec<T>) -> Vec<Vec<usize>> {
        let mut temp: Vec<(T, Vec<usize>)> = vec![];
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
        //temp.sort_unstable_by_key(|(label, verts)| verts.len());
        temp.sort_unstable_by(|(labelx, vertsx), (labely, vertsy)|{
            match vertsx.len().cmp(&vertsy.len()) {
                Ordering::Equal => {
                    labelx.cmp(labely)
                },
                c => c,
            }
        });
        temp.into_iter().map(|(_, bin)| bin).collect()
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
        g: &LabeledGraph<T>,
        h: &LabeledGraph<T>,
    ) -> bool {
        if depth == max_depth {
            collapse(verts_g, collapsed_verts_g);
            for i in 0..collapsed_verts_h.len() - 1 {
                for j in 0..collapsed_verts_h.len() {
                    if g.graph.get_edge(collapsed_verts_g[i], collapsed_verts_g[j])
                        != h.graph.get_edge(collapsed_verts_h[i], collapsed_verts_h[j])
                        {
                            return false;
                        }
                }
            }
            return true;
        } else {
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
                    h,
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
    pub fn is_color_iso(g: &LabeledGraph<T>, h: &LabeledGraph<T>) -> bool {
        if g.labeling != h.labeling {
            return false;
        } else {
            let orig_g_bins = LabeledGraph::bin(&g.labeling.labels);
            let mut g_bins = orig_g_bins.clone();
            if g_bins.len() > 0 {
                let h_bins = LabeledGraph::bin(&h.labeling.labels);
                let mut collapsed_verts_h = vec![];
                collapse(&h_bins, &mut collapsed_verts_h);
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
                    h,
                )
            } else {
                return true;
            }
        }
    }
}

fn collapse(list: &Vec<Vec<usize>>, out: &mut Vec<usize>) {
    out.clear();
    for i in list.iter() {
        for j in i.iter() {
            out.push(*j);
        }
    }
}

impl<T: Ord + Copy + Eq> Ord for LabeledGraph<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.labeling.cmp(&other.labeling)
    }
}

impl<T: Ord + Copy> PartialOrd for LabeledGraph<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.labeling.cmp(&other.labeling))
    }
}

impl<T: Ord + Eq + Copy> PartialEq for LabeledGraph<T> {
    //Isomorphism algorithm, taking labels into account
    fn eq(&self, other: &LabeledGraph<T>) -> bool {
        LabeledGraph::is_color_iso(self, other)
    }
}

impl<T: Ord + Eq + Copy> Eq for LabeledGraph<T> {}



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
pub struct Chunk<T> {
    pub graphs: Vec<LabeledGraph<T>>,
}

impl<T: Ord + Eq + Copy> Chunk<T> {
    //filters out duplicate graphs, in the isomorphism sense
    pub fn chunkify(mut v: Vec<LabeledGraph<T>>) -> Vec<Chunk<T>> {
        v.sort();
        let mut out = vec![];
        while let Some(i) = Chunk::find_split(&v) {
            out.push(Chunk {
                graphs: v.split_off(i),
            });
        }
        out.push(Chunk { graphs: v });
        out
    }
    fn find_split(v: &Vec<LabeledGraph<T>>) -> Option<usize> {
        for i in (1..v.len() - 1).rev() {
            if v[i].labeling != v[i - 1].labeling {
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
        mem::swap(&mut cleaned, &mut self.graphs);
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




    pub fn chunk_dedup(&mut self) {
        let mut graphs = vec![];
        mem::swap(&mut self.graphs, &mut graphs);


        let mut temp: Vec<(LabeledGraph<T>, Vec<usize>)> = graphs
            .into_iter()
            .map(|h| {
                let h_bins = LabeledGraph::bin(&h.labeling.labels);
                let h_bins_collapsed = h_bins.clone().into_iter().flatten().collect();
                (h, h_bins_collapsed)
            })
            .collect();




        while let Some((g, mut collapsed_verts)) = temp.pop() {
            let mut orig_g_bins = LabeledGraph::bin(&g.labeling.labels);
            let mut g_bins = orig_g_bins.clone();
            let max_depth = g_bins.len();
            Chunk::rec_iso_check(
                0,
                max_depth,
                &orig_g_bins,
                &mut g_bins,
                &mut temp,
                &mut collapsed_verts,
                &g,
            );
            self.graphs.push(g);
        }



    }

    fn rec_iso_check(
        depth: usize,
        max_depth: usize,
        orig_verts_g: &Vec<Vec<usize>>, //original bins, with more than one element
        verts_g: &mut Vec<Vec<usize>>,  //the current permutation hypothesis we're working on
        others: &mut Vec<(LabeledGraph<T>, Vec<usize>)>, //the vertices of h collapsed into a single vector
        collapsed_verts_g: &mut Vec<usize>,
        g: &LabeledGraph<T>,
    ) {
        if depth == max_depth {
            collapse(verts_g, collapsed_verts_g);
            others.retain(|h| {
                for i in 0..collapsed_verts_g.len() - 1 {
                    for j in 0..collapsed_verts_g.len() {
                        if g.graph.get_edge(collapsed_verts_g[i], collapsed_verts_g[j])
                            != h.0.graph.get_edge(h.1[i], h.1[j])
                            {
                                return true;
                            }
                    }
                }
                return false;
            });
        } else {
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
                    g,
                );
            }
        }
    }
}
