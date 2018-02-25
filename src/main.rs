extern crate crossbeam;
extern crate graph_lib;
extern crate termion;
use graph_lib::Graph;
use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;

fn print_vv<T: std::fmt::Debug>(v: &Vec<Vec<T>>) {
    for i in v {
        println!("{:?}", i);
    }
}
use std::time::{Duration, Instant};

use std::fs::File;
use std::fs;
use std::path::Path;
use std::error::Error;
use std::io::{BufWriter, Write};

fn dump_graph_list(list: &Vec<Graph>, n: u32) {
    let path_str = format!("out/{}.txt", n);
    let path = Path::new(&path_str);
    let path_pretty = path.display();
    match fs::create_dir("out") {
        Err(_) => {} //println!("! {}", e.description()),
        Ok(_) => {}
    }
    let file = match File::create(&path) {
        Err(e) => panic!("couldn't create {} : {}", path_pretty, e.description()),
        Ok(file) => file,
    };
    let mut writer = BufWriter::new(&file);
    for g in list {
        let mut s = g.to_string();
        s.push('\n');
        writer.write(&s.into_bytes());
    }
    writer.flush();
}

struct ThreadUpdate {
    curr_comp: usize,
    thread_id: usize,
    curr_chunk_size: usize,
    running: bool,
    chunks_processed: u32,
}

fn fmt_dur(d: &Duration) -> String {
    let hours = d.as_secs() / 3600;
    let mins = d.as_secs() / 60 % 60;
    format!("{:04}:{:02}:{:02}", hours, mins, d.as_secs() % 60)
}

fn main() {
    let g = Graph::new(1);

    let mut rows = Vec::new();
    rows.push(vec![g]);

    for i in 1..7 {
        let mut now = Instant::now();
        let mut this_row;
        let cand = format!("out/{}.txt", i + 1);
        if Path::new(&cand).exists() {
            this_row = Graph::load_file(&cand, i + 1, 10000);
            println!("Found size {}, with {} graphs", i + 1, this_row.len());
            rows.push(this_row);
        } else {
            let mut this_row = Vec::new();
            //let mut count = 0;
            println!("Generating next size...");
            for j in rows[i - 1].iter() {
                let mut t = j.get_next_size();
                //t.sort();

                Graph::clean_isos(&mut t);
                this_row.append(&mut t);
            }
            if this_row.len() == 0 {
                break;
            }

            println!(
                "{} finding chunks among {} graphs...",
                fmt_dur(&now.elapsed()),
                this_row.len()
            );
            now = Instant::now();

            dump_graph_list(&this_row, (i + 1) as u32);
            let mut chunks = Graph::chunkify(&mut this_row);
            println!("{} chunks({}):", fmt_dur(&now.elapsed()), chunks.len());
            let chunk_ts = Arc::new(Mutex::new(chunks));
            //let row_ts = Arc::new(&this_row);

            Graph::process_tier(&mut this_row, chunk_ts);

            println!(
                "{} Writing cleaned list to disk...",
                fmt_dur(&now.elapsed())
            );

            //Graph::clean_isos(&mut this_row);
            /*
                I need - a stack of slices to process, that can be read from multiple threads
                       - a way to create n threads
                       - each thread should just assign isNulls, cleanup is later

            */
            dump_graph_list(&this_row, (i + 1) as u32);
            rows.push(this_row);

            println!(
                "{} {} contains {} graphs cleaned",
                fmt_dur(&now.elapsed()),
                i + 1,
                rows[i].len()
            );
        }
    }
}
