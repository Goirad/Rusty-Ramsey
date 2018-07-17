mod graph_lib;
use graph_lib::{Edge, Graph, LabeledGraph, Labeling, LabelingVariant, Chunk};
use std::str::FromStr;
extern crate permutohedron;

fn main() {
    let seed = graph_lib::Graph::new(1);
    let mut gs = vec![LabeledGraph {
        labeling: Labeling::neighbors(&seed),
        graph: seed,
    }];
    for n in 2..11 {
        println!("Starting new tier {}", n);
        let mut next = vec![];
        for (i, g) in gs.iter().enumerate() {
            if i % 1000 == 0 {
                println!("Done generating from {} graphs", i);
            }
            let mut children = g.graph.generate_children();
            let mut children_labeled: Vec<LabeledGraph> = vec![];
            children.retain(|i| k4_free(i, &Edge::Red) && k4_free(i, &Edge::Green));

            children
                .into_iter()
                .map(|c| {
                    children_labeled.push(LabeledGraph {
                        labeling: Labeling::neighbors2(&c),
                        graph: c,
                    });
                })
                .count();

            //dedup(&mut children_labeled);
            next.append(&mut children_labeled);
        }
        if next.len() == 0 { break; }
        println!("before: {}", next.len());
        let mut chunks = Chunk::chunkify(next);
        println!("chunks: {}", chunks.len());
        let mut out = vec![];
        let mut total_comp = 0;
        let mut total_processed = 0;

        for (i, mut c) in chunks.into_iter().enumerate() {
            total_comp += c.graphs[0].labeling.complexity;
            total_processed += c.graphs.len();
            if i % 1000 == 0 {

                println!("done with {} {} {} {}", i, total_comp / 100, total_processed, out.len());
                total_comp = 0;
            }
            let mut c = c.chunk_dedup_setup();
            //c.dedup();
            out.append(&mut c.graphs);
        }
        println!("after: {}", out.len());
        gs = out;

    }
}

fn dedup<T: Eq>(v: &mut Vec<T>) {
    let mut cleaned = vec![];
    while let Some(g) = v.pop() {
        v.retain(|h| &g != h);
        cleaned.push(g);
    }
    v.clear();
    v.append(&mut cleaned);
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
