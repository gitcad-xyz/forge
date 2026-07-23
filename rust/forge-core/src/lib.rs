//! forge-core — the Rust port of the K1 exact planar kernel.
//!
//! Structure-for-structure with `src/forgekernel/{exact,brep,csg}.py`.
//! Every number is a BigRational; topological decisions use exact signs.
//! ref (the Python reference) is this port's oracle: the Python test
//! suite builds each case in BOTH and compares exact volume + a canonical
//! face-set. This crate never invents behaviour — it only makes ref fast.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use pyo3::prelude::*;
use std::collections::BTreeSet;

type R = BigRational;
type V = [R; 3];

fn ri(n: i64) -> R {
    BigRational::from_integer(BigInt::from(n))
}

fn add(a: &V, b: &V) -> V {
    [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
}
fn sub(a: &V, b: &V) -> V {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}
fn smul(s: &R, a: &V) -> V {
    [s * &a[0], s * &a[1], s * &a[2]]
}
fn dot(a: &V, b: &V) -> R {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}
fn cross(a: &V, b: &V) -> V {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}
fn is_zero(a: &V) -> bool {
    a[0].is_zero() && a[1].is_zero() && a[2].is_zero()
}

#[derive(Clone)]
struct Plane {
    n: V,
    d: R,
}
impl Plane {
    fn from_points(a: &V, b: &V, c: &V) -> Plane {
        let n = cross(&sub(b, a), &sub(c, a));
        Plane { d: dot(&n, a), n }
    }
    fn side(&self, p: &V) -> i32 {
        let s = dot(&self.n, p) - &self.d;
        if s.is_positive() {
            1
        } else if s.is_negative() {
            -1
        } else {
            0
        }
    }
    fn flipped(&self) -> Plane {
        Plane {
            n: [-&self.n[0], -&self.n[1], -&self.n[2]],
            d: -&self.d,
        }
    }
}

#[derive(Clone)]
struct Polygon {
    verts: Vec<V>,
    plane: Plane,
    source: String,
}
impl Polygon {
    fn new(verts: Vec<V>, source: String) -> Polygon {
        let plane = Plane::from_points(&verts[0], &verts[1], &verts[2]);
        Polygon { verts, plane, source }
    }
    fn with_plane(verts: Vec<V>, source: String, plane: Plane) -> Polygon {
        Polygon { verts, plane, source }
    }
    fn flipped(&self) -> Polygon {
        let mut v = self.verts.clone();
        v.reverse();
        Polygon::with_plane(v, self.source.clone(), self.plane.flipped())
    }
    fn area2(&self) -> R {
        let mut acc = [ri(0), ri(0), ri(0)];
        let v0 = &self.verts[0];
        for i in 1..self.verts.len() - 1 {
            acc = add(&acc, &cross(&sub(&self.verts[i], v0), &sub(&self.verts[i + 1], v0)));
        }
        dot(&acc, &acc)
    }
}

#[derive(Clone)]
struct Solid {
    polys: Vec<Polygon>,
}
impl Solid {
    fn new(polys: Vec<Polygon>) -> Solid {
        Solid {
            polys: polys.into_iter().filter(|p| !p.area2().is_zero()).collect(),
        }
    }

    fn cube(dx: R, dy: R, dz: R, prefix: &str) -> Solid {
        let o = ri(0);
        let vs = |i: usize| -> V {
            let x = [&o, &dx, &dx, &o, &o, &dx, &dx, &o][i].clone();
            let y = [&o, &o, &dy, &dy, &o, &o, &dy, &dy][i].clone();
            let z = [&o, &o, &o, &o, &dz, &dz, &dz, &dz][i].clone();
            [x, y, z]
        };
        let faces: [([usize; 4], &str); 6] = [
            ([0, 3, 2, 1], "bottom"),
            ([4, 5, 6, 7], "top"),
            ([0, 1, 5, 4], "front"),
            ([2, 3, 7, 6], "back"),
            ([1, 2, 6, 5], "right"),
            ([3, 0, 4, 7], "left"),
        ];
        let polys = faces
            .iter()
            .map(|(idx, name)| {
                Polygon::new(idx.iter().map(|&i| vs(i)).collect(), format!("{}.{}", prefix, name))
            })
            .collect();
        Solid::new(polys)
    }

    fn translated(&self, t: &V) -> Solid {
        Solid {
            polys: self
                .polys
                .iter()
                .map(|p| {
                    Polygon::with_plane(
                        p.verts.iter().map(|v| add(v, t)).collect(),
                        p.source.clone(),
                        Plane::from_points(&add(&p.verts[0], t), &add(&p.verts[1], t), &add(&p.verts[2], t)),
                    )
                })
                .collect(),
        }
    }

    fn volume6(&self) -> R {
        let mut acc = ri(0);
        for p in &self.polys {
            let v0 = &p.verts[0];
            for i in 1..p.verts.len() - 1 {
                acc += dot(v0, &cross(&p.verts[i], &p.verts[i + 1]));
            }
        }
        acc
    }

    /// Canonical face-set for the oracle: each polygon as its sorted vertex
    /// multiset ("n/d" strings), the whole set sorted. Order-independent —
    /// proves identical geometry regardless of internal BSP fragment order.
    fn canonical(&self) -> String {
        let mut faces: BTreeSet<String> = BTreeSet::new();
        for p in &self.polys {
            let mut vs: Vec<String> = p
                .verts
                .iter()
                .map(|v| format!("{}/{},{}/{},{}/{}", v[0].numer(), v[0].denom(), v[1].numer(), v[1].denom(), v[2].numer(), v[2].denom()))
                .collect();
            vs.sort();
            faces.insert(vs.join("|"));
        }
        faces.into_iter().collect::<Vec<_>>().join(";")
    }
}

// -- BSP boolean engine (ports csg.py) ---------------------------------------

#[derive(Default)]
struct Split {
    cof: Vec<Polygon>,
    cob: Vec<Polygon>,
    front: Vec<Polygon>,
    back: Vec<Polygon>,
}

/// Classify one polygon against a plane into coplanar-front/back and
/// strict front/back (owned vectors — the caller routes them, avoiding
/// the aliasing Python's in-place version relied on).
fn split(plane: &Plane, poly: &Polygon) -> Split {
    let mut r = Split::default();
    let (cof, cob, front, back) = (&mut r.cof, &mut r.cob, &mut r.front, &mut r.back);
    let sides: Vec<i32> = poly.verts.iter().map(|v| plane.side(v)).collect();
    let mut kind = 0;
    for &s in &sides {
        kind |= if s > 0 { 1 } else if s < 0 { 2 } else { 0 };
    }
    match kind {
        0 => {
            if dot(&plane.n, &poly.plane.n).is_positive() {
                cof.push(poly.clone());
            } else {
                cob.push(poly.clone());
            }
        }
        1 => front.push(poly.clone()),
        2 => back.push(poly.clone()),
        _ => {
            let mut f: Vec<V> = Vec::new();
            let mut b: Vec<V> = Vec::new();
            let n = poly.verts.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let (vi, vj) = (&poly.verts[i], &poly.verts[j]);
                let (si, sj) = (sides[i], sides[j]);
                if si >= 0 {
                    f.push(vi.clone());
                }
                if si <= 0 {
                    b.push(vi.clone());
                }
                if si * sj < 0 {
                    let t = (&plane.d - dot(&plane.n, vi)) / dot(&plane.n, &sub(vj, vi));
                    let x = add(vi, &smul(&t, &sub(vj, vi)));
                    f.push(x.clone());
                    b.push(x);
                }
            }
            if f.len() >= 3 {
                let p = Polygon::with_plane(f, poly.source.clone(), poly.plane.clone());
                if !p.area2().is_zero() {
                    front.push(p);
                }
            }
            if b.len() >= 3 {
                let p = Polygon::with_plane(b, poly.source.clone(), poly.plane.clone());
                if !p.area2().is_zero() {
                    back.push(p);
                }
            }
        }
    }
    r
}

struct Node {
    plane: Option<Plane>,
    front: Option<Box<Node>>,
    back: Option<Box<Node>>,
    polys: Vec<Polygon>,
}
impl Node {
    fn new() -> Node {
        Node { plane: None, front: None, back: None, polys: Vec::new() }
    }
    fn build(&mut self, polys: Vec<Polygon>) {
        if polys.is_empty() {
            return;
        }
        if self.plane.is_none() {
            self.plane = Some(polys[0].plane.clone());
        }
        let plane = self.plane.clone().unwrap();
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in &polys {
            let mut s = split(&plane, p);
            self.polys.append(&mut s.cof); // both coplanar sides stay here
            self.polys.append(&mut s.cob);
            front.append(&mut s.front);
            back.append(&mut s.back);
        }
        if !front.is_empty() {
            // reuse the existing child (build may be called again, as in
            // na.build(nb.all_polygons())) — never overwrite it
            if self.front.is_none() {
                self.front = Some(Box::new(Node::new()));
            }
            self.front.as_mut().unwrap().build(front);
        }
        if !back.is_empty() {
            if self.back.is_none() {
                self.back = Some(Box::new(Node::new()));
            }
            self.back.as_mut().unwrap().build(back);
        }
    }
    fn invert(&mut self) {
        for p in &mut self.polys {
            *p = p.flipped();
        }
        if let Some(pl) = &self.plane {
            self.plane = Some(pl.flipped());
        }
        if let Some(f) = &mut self.front {
            f.invert();
        }
        if let Some(b) = &mut self.back {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }
    fn clip_polygons(&self, polys: Vec<Polygon>) -> Vec<Polygon> {
        if self.plane.is_none() {
            return polys;
        }
        let plane = self.plane.clone().unwrap();
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in &polys {
            let mut s = split(&plane, p);
            front.append(&mut s.cof); // coplanar-front routes to front
            back.append(&mut s.cob); // coplanar-back routes to back
            front.append(&mut s.front);
            back.append(&mut s.back);
        }
        let front = match &self.front {
            Some(f) => f.clip_polygons(front),
            None => front,
        };
        let back = match &self.back {
            Some(b) => b.clip_polygons(back),
            None => Vec::new(),
        };
        let mut out = front;
        out.extend(back);
        out
    }
    fn clip_to(&mut self, other: &Node) {
        self.polys = other.clip_polygons(std::mem::take(&mut self.polys));
        if let Some(f) = &mut self.front {
            f.clip_to(other);
        }
        if let Some(b) = &mut self.back {
            b.clip_to(other);
        }
    }
    fn all_polygons(&self) -> Vec<Polygon> {
        let mut out = self.polys.clone();
        if let Some(f) = &self.front {
            out.extend(f.all_polygons());
        }
        if let Some(b) = &self.back {
            out.extend(b.all_polygons());
        }
        out
    }
}

fn node_of(s: &Solid) -> Node {
    let mut n = Node::new();
    n.build(s.polys.clone());
    n
}

fn boolean(op: &str, a: &Solid, b: &Solid) -> Solid {
    let mut na = node_of(a);
    let mut nb = node_of(b);
    match op {
        "union" => {
            na.clip_to(&nb);
            nb.clip_to(&na);
            nb.invert();
            nb.clip_to(&na);
            nb.invert();
            na.build(nb.all_polygons());
            Solid::new(na.all_polygons())
        }
        "cut" => {
            na.invert();
            na.clip_to(&nb);
            nb.clip_to(&na);
            nb.invert();
            nb.clip_to(&na);
            nb.invert();
            na.build(nb.all_polygons());
            na.invert();
            Solid::new(na.all_polygons())
        }
        "intersect" => {
            na.invert();
            nb.clip_to(&na);
            nb.invert();
            na.clip_to(&nb);
            nb.clip_to(&na);
            na.build(nb.all_polygons());
            na.invert();
            Solid::new(na.all_polygons())
        }
        _ => panic!("unknown op"),
    }
}

// -- pyo3 facade: parse dims as "num/den" strings, return exact results ------

fn parse_r(s: &str) -> R {
    let parts: Vec<&str> = s.split('/').collect();
    BigRational::new(parts[0].parse().unwrap(), parts[1].parse().unwrap())
}
fn r_str(r: &R) -> String {
    format!("{}/{}", r.numer(), r.denom())
}

/// Build a box, translate the second by (tx,ty,tz), apply the boolean op,
/// and return (signed_volume6_str, canonical_faces). All inputs "num/den".
#[pyfunction]
fn box_boolean(
    dx: &str, dy: &str, dz: &str,
    ex: &str, ey: &str, ez: &str,
    tx: &str, ty: &str, tz: &str,
    op: &str,
) -> (String, String) {
    let a = Solid::cube(parse_r(dx), parse_r(dy), parse_r(dz), "A");
    let b = Solid::cube(parse_r(ex), parse_r(ey), parse_r(ez), "B")
        .translated(&[parse_r(tx), parse_r(ty), parse_r(tz)]);
    let out = boolean(op, &a, &b);
    (r_str(&out.volume6()), out.canonical())
}

/// A single box's exact signed volume6 and canonical faces.
#[pyfunction]
fn box_form(dx: &str, dy: &str, dz: &str) -> (String, String) {
    let s = Solid::cube(parse_r(dx), parse_r(dy), parse_r(dz), "box");
    (r_str(&s.volume6()), s.canonical())
}

#[pymodule]
fn forgekernel_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(box_boolean, m)?)?;
    m.add_function(wrap_pyfunction!(box_form, m)?)?;
    Ok(())
}
