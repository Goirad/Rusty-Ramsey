
use std::cmp::{max, min, Ordering};


#[derive(PartialEq, Clone, Debug)]
enum Color {
    RED,
    GREEN,
    NONE,
}

impl Color {
    fn from_u64(n: u64) -> Option<Color>{
        match n {
            0 => Some(Color::GREEN),
            1 => Some(Color::RED),
            _ => None,
        }
    }
}


#[derive(Clone)]
struct Graph {
    num_verts: usize,
    edges: Vec<Color>,
    labeling: Vec<u32>,
    labeling_sorted: Vec<u32>,
    is_null: bool,
}
pub fn factorial(num: &usize) -> usize {
	let mut i = 1;
    let mut c: usize = num.clone();
    while c > 1 {
        i *= c;
        c -= 1;
    }
    i
}

pub fn dec_to_factorial(n: usize, dig: usize, out: &mut Vec<usize>) {
    let mut num = n;
    out.clear();
    for i in 0..dig {
        out.push(num % (i+1));
        num /= i+1;
    }
    out.reverse();
}
fn permute(list: &Vec<usize>, perm: &Vec<usize>, perm_scratch: &mut Vec<usize>, out: &mut Vec<usize>) {
    out.clear();
    perm_scratch.clone_from(list);
    for i in 0..list.len() {
        out.push(perm_scratch.remove(perm[i]));
    }
}

impl Graph {
    fn new(n: usize) -> Graph {
        let num_edges = n * (n - 1) / 2;
        let mut e = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            e.push(Color::RED);
        }
        let mut g = Graph {
            num_verts: n,
            edges: e,
            labeling: Vec::new(),
            labeling_sorted: Vec::new(),
            is_null: false,
        };
        g.labeling = g.label();
        g.labeling_sorted = g.label();
        g.labeling_sorted.sort();
        g
    }

    fn get_edge(&self, n: usize, m: usize) -> &Color {
        let n1 = min(n, m);
        let m1 = max(n, m);
        if n1 == m1 {
            return &Color::NONE;
        }
        //println!("{} {}", n1, m1);
        &self.edges[(m1 * m1 - m1) / 2 + n1]
    }

    fn from_string(s: String, n: usize) -> Graph {
        let mut g = Graph {num_verts: n, edges: Vec::new(), labeling: Vec::new(), labeling_sorted: Vec::new(), is_null: false};
        for c in s.into_bytes().iter() {
            g.edges.push(Color::from_u64((c - 48) as u64).unwrap());
        }
        g.labeling = g.label();

        g
    }

    fn label(&self) -> Vec<u32> {
        let mut l = Vec::new();
        for i in 0..self.num_verts {
            let mut k = 0;
            for j in 0..self.num_verts {
                if self.get_edge(i, j) == &Color::RED {
                    k += 1;
                }
            }
            l.push(k);
        }
        l
    }

    fn get_next_size(&self) -> Vec<Graph> {
        let mut next_size = Vec::new();
        for mut i in 0..(2 as u64).pow(self.num_verts as u32) {
            let mut g = self.clone();
            g.num_verts += 1;
            for _ in 0..self.num_verts {
                g.edges.push(Color::from_u64(i&1).unwrap());
                i>>=1;
            }

            if !g.has_k4(&Color::RED) && !g.has_k4(&Color::GREEN) {
                g.labeling = g.label();
                g.labeling_sorted = g.label();
                g.labeling_sorted.sort();
                next_size.push(g);
            }
        }

        return next_size;
    }

    fn has_k3(&self, col: &Color) -> bool {
        for i in 0..self.num_verts-2 {
            for j in i..self.num_verts-1 {
                for k in j..self.num_verts {
                    if
                        self.get_edge(i, j) == col &&
                        self.get_edge(i, k) == col &&

                        self.get_edge(j, k) == col
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }
    fn has_k4(&self, col: &Color) -> bool {
        if self.num_verts < 4 {
            return false;
        }
        for i in 0..self.num_verts-3 {
            for j in i..self.num_verts-2 {
                for k in j..self.num_verts-1 {
                    for l in k..self.num_verts {
                        if
                            self.get_edge(i, j) == col &&
                            self.get_edge(i, k) == col &&
                            self.get_edge(i, l) == col &&

                            self.get_edge(j, k) == col &&
                            self.get_edge(j, l) == col &&

                            self.get_edge(k, l) == col
                        {
                            return true;
                        }
                    }

                }
            }
        }

        return false;
    }

    fn collapse_verts(v: &Vec<Vec<usize>>, out: &mut Vec<usize>) {
        out.clear();
        for i in v.iter() {
            for j in i.iter() {
                out.push(*j);
            }
        }
    }
    fn rec_iso_check(depth: usize, orig_verts_g: &Vec<Vec<usize>>, verts_g: &mut Vec<Vec<usize>>, collapsed_verts_h: &Vec<usize>, g: &Graph, h: &Graph, mem_block: &mut IsoMemBlock) -> bool {

        if depth >= g.num_verts {

            Graph::collapse_verts(&verts_g, mem_block.collapsed_verts_g);


            for i in 0..g.num_verts-1 {
                for j in i+1..g.num_verts {
                    if g.get_edge(mem_block.collapsed_verts_g[i], mem_block.collapsed_verts_g[j]) != h.get_edge(collapsed_verts_h[i], collapsed_verts_h[j]) {
                        return false;
                    }
                }
            }
            return true;
        }else{

            if orig_verts_g[depth].len() > 0 {

                for i in 0..factorial(&orig_verts_g[depth].len()) {
                    //println!("Checking {} {} {} {} permutations", i, depth, g.num_verts, &orig_verts_g[depth].len());
                    dec_to_factorial(i, orig_verts_g[depth].len(), mem_block.perm);
                    //println!("permutation {}", i);
                    //println!("{:?}", mem_block.perm);
                    //println!("{:?}", verts_g[depth]);
                    permute(&orig_verts_g[depth], mem_block.perm, mem_block.perm_scratch, &mut verts_g[depth]);
                    //println!("{:?}", verts_g[depth]);
                    if Graph::rec_iso_check(depth+1, orig_verts_g, verts_g, collapsed_verts_h, g, h, mem_block) {
                        return true;
                    }
                }
                return false;
            }else{
                return Graph::rec_iso_check(depth+1, orig_verts_g, verts_g, collapsed_verts_h, g, h, mem_block);
            }
        }

    }

    fn is_color_iso(g: &Graph, h: &Graph) -> bool {
        if g.labeling_sorted == h.labeling_sorted {
            //TODO normalize vertex labelings
            let mut orig_verts_g = Vec::new();
            let mut verts_g = Vec::new();
            let mut verts_h = Vec::new();
            for _ in 0..g.num_verts {
                orig_verts_g.push(Vec::new());
                verts_g.push(Vec::new());
                verts_h.push(Vec::new());
            }

            for i in 0..g.num_verts {
                //println!("{}", i);
                orig_verts_g[g.labeling[i] as usize].push(i);
                verts_g[g.labeling[i] as usize].push(i);
                verts_h[h.labeling[i] as usize].push(i);
            }
            let mut collapsed_verts_h = Vec::new();
            Graph::collapse_verts(&verts_h, &mut collapsed_verts_h);
            let mut perm = Vec::new();
            let mut perm_scratch = Vec::new();
            let mut collapsed_verts_g = Vec::new();
            let mut mem_block = IsoMemBlock {collapsed_verts_g: &mut collapsed_verts_g, perm: &mut perm, perm_scratch: &mut perm_scratch};
            //println!();
            return Graph::rec_iso_check(0, &orig_verts_g, &mut verts_g, &collapsed_verts_h, g, h, &mut mem_block);
        }else{
            return false;
        }
    }
}

struct IsoMemBlock<'a> {
    //orig_verts_g:       &'a Vec<Vec<usize>>,
    //collapsed_verts_h:  &'a Vec<usize>,
    //verts_g:            &'a Vec<Vec<usize>>,
    collapsed_verts_g:  &'a mut Vec<usize>,
    perm:               &'a mut Vec<usize>,
    perm_scratch:       &'a mut Vec<usize>,
}

impl PartialOrd for Graph {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>{
        Some(self.labeling_sorted.cmp(&other.labeling_sorted))
    }
}
impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        return Graph::is_color_iso(self, other);
    }
}
impl Ord for Graph {
    fn cmp(&self, other: &Self) -> Ordering{
        self.labeling_sorted.cmp(&other.labeling_sorted)
    }
}
impl Eq for Graph {

}

fn print_vv(v: &Vec<Vec<usize>>) {
    for i in v {
        println!("{:?}", i);
    }
}



fn main() {
    let g = Graph::new(1);
    let mut rows = Vec::new();
    rows.push(vec!(g));

    for i in 1..9 {
        let mut this_row = Vec::new();
        //let mut count = 0;
        for j in rows[i-1].iter() {
            /*count += 1;
            if count % 1000 == 0 {
                println!("{}% done generating next size", 100.0 * count as f32/rows[i-1].len() as f32);
            }*/
            this_row.append(&mut j.get_next_size());
        }
        if this_row.len() == 0 {
            break;
        }
        println!("cleaning isos.. ({})", this_row.len());
        this_row.sort();
        let mut c = 1;
        {let mut h = &this_row[0];
        for i in 0..this_row.len() {
            if h.labeling_sorted != this_row[i].labeling_sorted {
                c += 1;
                h = &this_row[i];
            }
        }}
        println!("found {} chunks", c);
        for j in 0..this_row.len()-1 {
            if j%1000 == 0 {
                println!("{} {:?}", j, this_row[j].labeling_sorted);
            }
            if !this_row[j].is_null{
                for k in j+1..this_row.len() {
                    if this_row[k].labeling_sorted != this_row[j].labeling_sorted {
                        break;
                    }
                    if !this_row[k].is_null {
                        if this_row[j] == this_row[k] {
                            this_row[k].is_null = true;
                        }
                    }
                }
            }
        }

        for j in (0..this_row.len()).rev() {
            if this_row[j].is_null {
                this_row.remove(j);
            }
        }
        rows.push(this_row);
        println!("{} contains {} graphs", i+1, rows[i].len());
    }




}
