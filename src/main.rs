pub mod graph_lib;

use graph_lib::{Chunk, Edge, Graph, LabeledGraph, Labeling};

extern crate permutohedron;
extern crate rayon;

use rayon::prelude::*;

use std::time::{Duration, Instant};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

fn main() {
    let seed = graph_lib::Graph::new(1);
    let mut gs = vec![LabeledGraph {
        labeling: Labeling::neighbors2(&seed),
        graph: seed,
    }];
    for n in 2..12 {
        println!("Starting new tier {}", n);
        let mut next = Mutex::new(vec![]);
        gs.par_iter().for_each(|g| {
            let mut children = g.graph.generate_children();
            let mut children_labeled: Vec<LabeledGraph> = vec![];
            children.retain(|i| k4_free(i, &Edge::Red) && k4_free(i, &Edge::Green));

            children.into_iter().for_each(|c| {
                children_labeled.push(LabeledGraph {
                    labeling: Labeling::neighbors2(&c),
                    graph: c,
                });
            });
            next.lock().unwrap().append(&mut children_labeled);
        });
        let next = next.into_inner().unwrap();
        if next.len() == 0 {
            break;
        }
        println!("before: {}", next.len());
        let mut chunks = Chunk::chunkify(next);
        println!("chunks: {}", chunks.len());
        let mut out = vec![];
        let mut total_comp = 0;
        let mut total_processed = 0;

        let num_chunks_left = AtomicUsize::new(chunks.len());

        chunks.par_iter_mut().enumerate().for_each(|(i, c)| {

            let start = Instant::now();
            num_chunks_left.fetch_sub(1, Ordering::Relaxed);
            let comp = c.graphs[0].labeling.complexity;
            let num_old = c.graphs.len();

            c.chunk_dedup();
            c.graphs.shrink_to_fit();
            if comp > 10000 || num_old > 1000 {
                println!("starting with {:6}, {:7} left, comp {:8}, {:4} graphs -> {:4}, {:2}% reduction, {}",
                         i,
                         num_chunks_left.load(Ordering::Relaxed),
                         comp,
                         num_old,
                         c.graphs.len(),
                         (num_old - c.graphs.len()) * 100 / num_old,
                         fmt_dur(start.elapsed())
                );
            }
        });

        chunks.into_iter().for_each(|mut c| {
            out.append(&mut c.graphs);
        });

        println!("after: {}", out.len());
        gs = out;
    }
}

fn fmt_dur(d: Duration) -> String {
    let hours = d.as_secs() / 3600;
    let mins = d.as_secs() / 60 % 60;
    format!(
        "{:04}:{:02}:{:02}.{:03}",
        hours,
        mins,
        d.as_secs() % 60,
        d.subsec_nanos() / 1_000_000
    )
}

fn dedup<T: Eq>(v: &mut Vec<T>) {
    let mut old = vec![];
    std::mem::swap(v, &mut old);
    while let Some(g) = old.pop() {
        old.retain(|h| &g != h);
        v.push(g);
    }
}

fn k3_free(g: &Graph, col: &Edge) -> bool {
    if g.vertices < 3 {
        return true;
    }
    for i in 0..g.vertices - 2 {
        for j in i..g.vertices - 1 {
            for k in j..g.vertices {
                if g.get_edge(i, j) == col && g.get_edge(i, k) == col && g.get_edge(j, k) == col {
                    return false;
                }
            }
        }
    }
    true
}

fn k4_free(g: &Graph, col: &Edge) -> bool {
    if g.vertices < 4 {
        return true;
    }
    for i in 0..g.vertices - 3 {
        for j in i..g.vertices - 2 {
            for k in j..g.vertices - 1 {
                for l in k..g.vertices {
                    if g.get_edge(i, j) == col
                        && g.get_edge(i, k) == col
                        && g.get_edge(i, l) == col
                        && g.get_edge(j, k) == col
                        && g.get_edge(j, l) == col
                        && g.get_edge(k, l) == col
                    {
                        return false;
                    }
                }
            }
        }
    }
    true
}
