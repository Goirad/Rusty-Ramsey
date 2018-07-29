pub mod graph_lib;

use graph_lib::{Chunk, Edge, Graph, LabeledGraph, Labeling};

extern crate permutohedron;
extern crate rayon;
extern crate crossbeam;

use rayon::prelude::*;

use std::time::{Duration, Instant};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};


fn check(g: &Graph) -> bool {
    k4_free(g, &Edge::Red) && k4_free(g, &Edge::Green)
    //true
}

fn main() {
    let seed = graph_lib::Graph::new(1);
    let mut gs = vec![seed];
    for n in 2..20 {
        println!("Starting new tier {}", n);
        let mut next = Arc::new(Mutex::new(vec![]));

        if gs.len() > 1000 {
            let size = gs.len() / 4;
            crossbeam::scope(|scope| {
                for c in 0..4 {
                    let mygraphs = &gs;
                    let mynext = next.clone();
                    scope.spawn(move || {
                        let mut temp = vec![];
                        for (i, g) in mygraphs.iter().skip(c * size).take(size).enumerate() {
                            let children = g.generate_children2(&check);
                            let mut children_labeled: Vec<LabeledGraph<_>> = vec![];
                            if i % 10000 == 0 {
                                println!("thread {} done with {} / {}", c, i, size);
                            }
                            children.into_iter().for_each(|c| {
                                children_labeled.push(LabeledGraph {
                                    labeling: Labeling::k3(&c),
                                    graph: c,
                                });
                            });
                            temp.append(&mut children_labeled);
                        }
                        mynext.lock().unwrap().append(&mut temp);
                    });
                }
            });
        } else {
            gs.iter().for_each(|g| {
                let children = g.generate_children2(&check);
                let mut children_labeled: Vec<LabeledGraph<_>> = vec![];

                children.into_iter().for_each(|c| {
                    children_labeled.push(LabeledGraph {
                        labeling: Labeling::k3(&c),
                        graph: c,
                    });
                });
                next.lock().unwrap().append(&mut children_labeled);
            });
        }
        let next = Arc::try_unwrap(next).unwrap().into_inner().unwrap();

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

        let mut hard_ones = Mutex::new(vec![]);

        chunks.par_iter_mut().enumerate().for_each(|(i, c)| {
            let start = Instant::now();
            num_chunks_left.fetch_sub(1, Ordering::Relaxed);
            let comp = c.graphs[0].labeling.complexity;
            let num_old = c.graphs.len();

            if false && comp > 500_000 {
                let mut x = c.graphs.iter().map(|l| {
                    LabeledGraph {
                        graph: l.graph.clone(),
                        labeling: Labeling::k3(&l.graph),
                    }
                }).collect();
                hard_ones.lock().unwrap().append(&mut x);
                c.graphs.clear();
            } else {
                c.chunk_dedup();
                c.graphs.shrink_to_fit();
                if c.graphs.len() > 1 && comp > 10000 {
                    println!("{} left. Chunk of comp {} had {} graphs left after cleaning",
                             num_chunks_left.load(Ordering::Relaxed),
                             comp,
                             c.graphs.len()
                    );
                }
                /*if num_chunks_left.load(Ordering::Relaxed) % 10000 == 0 {
                    println!("starting with {:6}, {:7} left, comp {:8}, {:4} graphs -> {:4}, {:2}% reduction, {}",
                             i,
                             num_chunks_left.load(Ordering::Relaxed),
                             comp,
                             num_old,
                             c.graphs.len(),
                             (num_old - c.graphs.len()) * 100 / num_old,
                             fmt_dur(start.elapsed())
                    );
                }*/
            }
        });
        let hard_ones = hard_ones.into_inner().unwrap();
        if hard_ones.len() > 0 {
            let mut hard_chunks = Chunk::chunkify(hard_ones);
            let mut total_comp = 0;
            let mut total_processed = 0;

            let num_chunks_left = AtomicUsize::new(hard_chunks.len());
            println!("about to start hard chunks {}", hard_chunks.len());
            hard_chunks.par_iter_mut().enumerate().for_each(|(i, c)| {
                let start = Instant::now();
                num_chunks_left.fetch_sub(1, Ordering::Relaxed);
                let comp = c.graphs[0].labeling.complexity;
                let num_old = c.graphs.len();

                c.chunk_dedup();
                c.graphs.shrink_to_fit();
                if c.graphs.len() > 1 && comp > 10000 {
                    println!("{} left. Chunk of comp {} had {} graphs left after cleaning",
                             num_chunks_left.load(Ordering::Relaxed),
                             comp,
                             c.graphs.len()
                    );
                }
                /*if comp > 100000 || num_old > 10000 {
                    println!("starting with {:6}, {:7} left, comp {:8}, {:4} graphs -> {:4}, {:2}% reduction, {}",
                             i,
                             num_chunks_left.load(Ordering::Relaxed),
                             comp,
                             num_old,
                             c.graphs.len(),
                             (num_old - c.graphs.len()) * 100 / num_old,
                             fmt_dur(start.elapsed())
                    );
                    if comp > 100_000 && c.graphs.len() > 1 {
                        //println!("{:?}", c.graphs[0]);
                        //println!("{:?}", c.graphs[1]);
                    }
                }*/
            });

            hard_chunks.into_iter().for_each(|c| {
                out.append(&mut c.graphs.into_iter().map(|l| l.graph).collect());
            });
        }
        chunks.into_iter().for_each(|c| {
            out.append(&mut c.graphs.into_iter().map(|l| l.graph).collect());
        });


        println!("after: {}", out.len());
        gs = out;
    }
}

fn views<T>(v: &Vec<T>, n: usize) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let size = v.len() / n;
    let mut curr = 0;
    while curr + size < v.len() {
        out.push((curr, curr + size));
        curr = curr + size;
    }
    out.push((curr, v.len()));
    out
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
            if g.get_edge(i, j) == col {
                for k in j..g.vertices - 1 {
                    if g.get_edge(i, k) == col && g.get_edge(j, k) == col {
                        for l in k..g.vertices {
                            if     g.get_edge(i, l) == col
                                && g.get_edge(j, l) == col
                                && g.get_edge(k, l) == col
                                {
                                    return false;
                                }
                        }
                    }
                }
            }
        }
    }
    true
}

fn k5_free(g: &Graph, col: &Edge) -> bool {
    if g.vertices < 4 {
        return true;
    }
    for i in 0..g.vertices - 4 {
        for j in i..g.vertices - 3 {
            for k in j..g.vertices - 2 {
                for l in k..g.vertices - 1 {
                    for m in l..g.vertices {
                        if g.get_edge(i, j) == col
                            && g.get_edge(i, k) == col
                            && g.get_edge(i, l) == col
                            && g.get_edge(i, m) == col

                            && g.get_edge(j, k) == col
                            && g.get_edge(j, l) == col
                            && g.get_edge(j, m) == col

                            && g.get_edge(k, l) == col
                            && g.get_edge(k, m) == col

                            && g.get_edge(l, m) == col
                            {
                                return false;
                            }
                    }
                }
            }
        }
    }
    true
}