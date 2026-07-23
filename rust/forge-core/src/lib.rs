//! forge-core — Rust port of the exact planar kernel (K1), pyo3-exposed.
//!
//! Structure-for-structure with `src/forgekernel/{exact,brep,csg}.py`.
//! Every number is a BigRational; topological decisions use exact signs,
//! never epsilons. ref (the Python reference) is this port's oracle: the
//! Python suite builds each case in BOTH and compares exact volume + a
//! canonical face-set. `PySolid` is an opaque handle the Python seam
//! adapter holds and operates on, exactly like it holds ref solids.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

type R = BigRational;
type V = [R; 3];

fn ri(n: i64) -> R {
    BigRational::from_integer(BigInt::from(n))
}
fn half() -> R {
    BigRational::new(BigInt::from(1), BigInt::from(2))
}
fn add(a: &V, b: &V) -> V {
    [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
}
fn sub(a: &V, b: &V) -> V {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}
fn neg(a: &V) -> V {
    [-&a[0], -&a[1], -&a[2]]
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
fn norm1(a: &V) -> R {
    a[0].abs() + a[1].abs() + a[2].abs()
}
fn isqrt(n: &BigInt) -> BigInt {
    n.sqrt()
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
        Plane { n: neg(&self.n), d: -&self.d }
    }
    /// Integer coprime canonical (nx,ny,nz,d) with a sign convention.
    fn canonical(&self) -> [BigInt; 4] {
        let nums = [&self.n[0], &self.n[1], &self.n[2], &self.d];
        let mut den = BigInt::one();
        for v in nums.iter() {
            den = den.lcm(v.denom());
        }
        let mut ints: Vec<BigInt> = nums.iter().map(|v| v.numer() * (&den / v.denom())).collect();
        let mut g = BigInt::zero();
        for v in ints.iter() {
            g = g.gcd(v);
        }
        if !g.is_zero() {
            for v in ints.iter_mut() {
                *v /= &g;
            }
        }
        for v in ints.iter() {
            if !v.is_zero() {
                if v.is_negative() {
                    for w in ints.iter_mut() {
                        *w = -&*w;
                    }
                }
                break;
            }
        }
        [ints[0].clone(), ints[1].clone(), ints[2].clone(), ints[3].clone()]
    }
    fn coplanar_key(&self) -> [BigInt; 4] {
        let c = self.canonical();
        let neg: [BigInt; 4] = [-&c[0], -&c[1], -&c[2], -&c[3]];
        if c <= neg {
            c
        } else {
            neg
        }
    }
    /// Exact rational unit normal when |n|^2 is a perfect square.
    fn unit_normal(&self) -> Option<V> {
        let c = self.canonical();
        let nn = &c[0] * &c[0] + &c[1] * &c[1] + &c[2] * &c[2];
        let root = isqrt(&nn);
        if &root * &root != nn {
            return None;
        }
        let rr = BigRational::from_integer(root);
        Some([
            BigRational::from_integer(c[0].clone()) / &rr,
            BigRational::from_integer(c[1].clone()) / &rr,
            BigRational::from_integer(c[2].clone()) / &rr,
        ])
    }
}

fn to_f3(v: &V) -> [f64; 3] {
    [
        v[0].to_f64().unwrap_or(f64::NAN),
        v[1].to_f64().unwrap_or(f64::NAN),
        v[2].to_f64().unwrap_or(f64::NAN),
    ]
}

#[derive(Clone)]
struct Polygon {
    verts: Vec<V>,
    fverts: Vec<[f64; 3]>, // lazy: cached float approximations (predicate fast path)
    plane: Plane,
    source: String,
}
impl Polygon {
    fn new(verts: Vec<V>, source: String) -> Polygon {
        let plane = Plane::from_points(&verts[0], &verts[1], &verts[2]);
        let fverts = verts.iter().map(to_f3).collect();
        Polygon { verts, fverts, plane, source }
    }
    fn with_plane(verts: Vec<V>, source: String, plane: Plane) -> Polygon {
        let fverts = verts.iter().map(to_f3).collect();
        Polygon { verts, fverts, plane, source }
    }
    fn flipped(&self) -> Polygon {
        let mut v = self.verts.clone();
        v.reverse();
        let mut fv = self.fverts.clone();
        fv.reverse();
        Polygon { verts: v, fverts: fv, plane: self.plane.flipped(), source: self.source.clone() }
    }
    fn area2(&self) -> R {
        let mut acc = [ri(0), ri(0), ri(0)];
        let v0 = &self.verts[0];
        for i in 1..self.verts.len() - 1 {
            acc = add(&acc, &cross(&sub(&self.verts[i], v0), &sub(&self.verts[i + 1], v0)));
        }
        dot(&acc, &acc)
    }
    /// Float-filtered degeneracy test (hot path in Solid::new): a clearly
    /// non-zero float area normal proves non-degenerate without the exact
    /// area2; only near-zero falls back to exact. Answer is always exact.
    fn is_degenerate(&self) -> bool {
        let v0 = self.fverts[0];
        let mut acc = [0.0f64; 3];
        for i in 1..self.fverts.len() - 1 {
            let a = self.fverts[i];
            let b = self.fverts[i + 1];
            let e1 = [a[0] - v0[0], a[1] - v0[1], a[2] - v0[2]];
            let e2 = [b[0] - v0[0], b[1] - v0[1], b[2] - v0[2]];
            acc[0] += e1[1] * e2[2] - e1[2] * e2[1];
            acc[1] += e1[2] * e2[0] - e1[0] * e2[2];
            acc[2] += e1[0] * e2[1] - e1[1] * e2[0];
        }
        let mag = acc[0].abs() + acc[1].abs() + acc[2].abs();
        if mag.is_finite() && mag > 1e-7 {
            return false; // clearly a real face
        }
        self.area2().is_zero() // ambiguous → exact
    }
}

#[derive(Clone)]
struct Solid {
    polys: Vec<Polygon>,
}
impl Solid {
    fn new(polys: Vec<Polygon>) -> Solid {
        Solid {
            polys: polys.into_iter().filter(|p| !p.is_degenerate()).collect(),
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
        Solid::new(
            faces
                .iter()
                .map(|(idx, name)| {
                    Polygon::new(idx.iter().map(|&i| vs(i)).collect(), format!("{}.{}", prefix, name))
                })
                .collect(),
        )
    }

    fn prism(loop2d: Vec<[R; 2]>, h: R, prefix: &str) -> Solid {
        let mut lp = loop2d;
        if loop_area2(&lp).is_negative() {
            lp.reverse();
        }
        let tris = ear_clip(&lp);
        let mut polys = Vec::new();
        let z0 = ri(0);
        for (a, b, c) in &tris {
            polys.push(Polygon::new(
                vec![[a[0].clone(), a[1].clone(), z0.clone()],
                     [c[0].clone(), c[1].clone(), z0.clone()],
                     [b[0].clone(), b[1].clone(), z0.clone()]],
                format!("{}.bottom", prefix),
            ));
            polys.push(Polygon::new(
                vec![[a[0].clone(), a[1].clone(), h.clone()],
                     [b[0].clone(), b[1].clone(), h.clone()],
                     [c[0].clone(), c[1].clone(), h.clone()]],
                format!("{}.top", prefix),
            ));
        }
        let n = lp.len();
        for i in 0..n {
            let (p1, p2) = (&lp[i], &lp[(i + 1) % n]);
            polys.push(Polygon::new(
                vec![[p1[0].clone(), p1[1].clone(), z0.clone()],
                     [p2[0].clone(), p2[1].clone(), z0.clone()],
                     [p2[0].clone(), p2[1].clone(), h.clone()],
                     [p1[0].clone(), p1[1].clone(), h.clone()]],
                format!("{}.side{}", prefix, i),
            ));
        }
        Solid::new(polys)
    }

    fn prismatoid(bottom: Vec<[R; 2]>, z0: R, top: Vec<[R; 2]>, z1: R, prefix: &str) -> Solid {
        let mut b = bottom;
        let mut t = top;
        if loop_area2(&b).is_negative() {
            b.reverse();
            t.reverse();
        }
        let mut polys = Vec::new();
        for (a, bb, c) in ear_clip(&b) {
            polys.push(Polygon::new(
                vec![[a[0].clone(), a[1].clone(), z0.clone()],
                     [c[0].clone(), c[1].clone(), z0.clone()],
                     [bb[0].clone(), bb[1].clone(), z0.clone()]],
                format!("{}.bottom", prefix),
            ));
        }
        for (a, bb, c) in ear_clip(&t) {
            polys.push(Polygon::new(
                vec![[a[0].clone(), a[1].clone(), z1.clone()],
                     [bb[0].clone(), bb[1].clone(), z1.clone()],
                     [c[0].clone(), c[1].clone(), z1.clone()]],
                format!("{}.top", prefix),
            ));
        }
        let n = b.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let b0 = &b[i];
            let b1 = &b[j];
            let t0 = &t[i];
            let t1 = &t[j];
            polys.push(Polygon::new(
                vec![[b0[0].clone(), b0[1].clone(), z0.clone()],
                     [b1[0].clone(), b1[1].clone(), z0.clone()],
                     [t1[0].clone(), t1[1].clone(), z1.clone()]],
                format!("{}.side{}", prefix, i),
            ));
            polys.push(Polygon::new(
                vec![[b0[0].clone(), b0[1].clone(), z0.clone()],
                     [t1[0].clone(), t1[1].clone(), z1.clone()],
                     [t0[0].clone(), t0[1].clone(), z1.clone()]],
                format!("{}.side{}", prefix, i),
            ));
        }
        Solid::new(polys)
    }

    fn mapped(&self, f: &dyn Fn(&V) -> V) -> Solid {
        Solid {
            polys: self
                .polys
                .iter()
                .map(|p| {
                    let nv: Vec<V> = p.verts.iter().map(|v| f(v)).collect();
                    let pl = Plane::from_points(&nv[0], &nv[1], &nv[2]);
                    Polygon::with_plane(nv, p.source.clone(), pl)
                })
                .collect(),
        }
    }

    fn translated(&self, t: &V) -> Solid {
        self.mapped(&|v| add(v, t))
    }
    fn scaled(&self, fx: R, fy: R, fz: R) -> Solid {
        let s = self.mapped(&|v| [&v[0] * &fx, &v[1] * &fy, &v[2] * &fz]);
        if (&fx * &fy * &fz).is_negative() {
            Solid { polys: s.polys.iter().map(|p| p.flipped()).collect() }
        } else {
            s
        }
    }
    fn mirrored(&self, axis: usize) -> Solid {
        let m = self.mapped(&|v| {
            let mut w = v.clone();
            w[axis] = -&w[axis];
            w
        });
        Solid { polys: m.polys.iter().map(|p| p.flipped()).collect() }
    }
    fn rotated_quarter(&self, axis: usize, quarters: i64) -> Solid {
        let q = ((quarters % 4) + 4) % 4;
        let a = (axis + 1) % 3;
        let b = (axis + 2) % 3;
        self.mapped(&|v| {
            let mut w = v.clone();
            for _ in 0..q {
                let (na, nb) = (-&w[b], w[a].clone());
                w[a] = na;
                w[b] = nb;
            }
            w
        })
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
    fn volume(&self) -> R {
        self.volume6() / ri(6)
    }
    fn bbox(&self) -> (V, V) {
        let mut lo = self.polys[0].verts[0].clone();
        let mut hi = lo.clone();
        for p in &self.polys {
            for v in &p.verts {
                for k in 0..3 {
                    if v[k] < lo[k] {
                        lo[k] = v[k].clone();
                    }
                    if v[k] > hi[k] {
                        hi[k] = v[k].clone();
                    }
                }
            }
        }
        (lo, hi)
    }

    /// Exact closure test (T-junction tolerant): signed interval coverage
    /// on each Plücker-keyed carrier line must cancel. Empty == closed.
    fn watertight_bad(&self) -> usize {
        // key: (canon_dir, moment(cross(a,dir))) -> Vec<(lo,hi,sign)>
        let mut lines: BTreeMap<String, Vec<(R, R, i32)>> = BTreeMap::new();
        for p in &self.polys {
            let n = p.verts.len();
            for i in 0..n {
                let a = &p.verts[i];
                let b = &p.verts[(i + 1) % n];
                let d = sub(b, a);
                if let Some(cd) = canon_dir(&d) {
                    let mom = cross(a, &cd);
                    let key = format!(
                        "{}/{},{}/{},{}/{}|{}/{},{}/{},{}/{}",
                        cd[0].numer(), cd[0].denom(), cd[1].numer(), cd[1].denom(),
                        cd[2].numer(), cd[2].denom(),
                        mom[0].numer(), mom[0].denom(), mom[1].numer(), mom[1].denom(),
                        mom[2].numer(), mom[2].denom()
                    );
                    let ta = dot(a, &cd);
                    let tb = dot(b, &cd);
                    let (lo, hi, sign) = if ta < tb { (ta, tb, 1) } else { (tb, ta, -1) };
                    lines.entry(key).or_default().push((lo, hi, sign));
                }
            }
        }
        let mut bad = 0;
        for segs in lines.values() {
            let mut cuts: BTreeSet<R> = BTreeSet::new();
            for (lo, hi, _) in segs {
                cuts.insert(lo.clone());
                cuts.insert(hi.clone());
            }
            let cv: Vec<R> = cuts.into_iter().collect();
            for w in cv.windows(2) {
                let (lo, hi) = (&w[0], &w[1]);
                let mut cov = 0i32;
                for (slo, shi, s) in segs {
                    if slo <= lo && hi <= shi {
                        cov += s;
                    }
                }
                if cov != 0 {
                    bad += 1;
                    break;
                }
            }
        }
        bad
    }

    fn canonical(&self) -> String {
        let mut faces: BTreeSet<String> = BTreeSet::new();
        for p in &self.polys {
            let mut vs: Vec<String> = p
                .verts
                .iter()
                .map(|v| {
                    format!(
                        "{}/{},{}/{},{}/{}",
                        v[0].numer(), v[0].denom(), v[1].numer(), v[1].denom(),
                        v[2].numer(), v[2].denom()
                    )
                })
                .collect();
            vs.sort();
            faces.insert(vs.join("|"));
        }
        faces.into_iter().collect::<Vec<_>>().join(";")
    }

    fn logical_face_count(&self) -> usize {
        let mut keys: BTreeSet<String> = BTreeSet::new();
        for p in &self.polys {
            let c = p.plane.canonical();
            keys.insert(format!("{},{},{},{}", c[0], c[1], c[2], c[3]));
        }
        keys.len()
    }
}

fn loop_area2(lp: &[[R; 2]]) -> R {
    let mut acc = ri(0);
    let n = lp.len();
    for i in 0..n {
        let (a, b) = (&lp[i], &lp[(i + 1) % n]);
        acc += &a[0] * &b[1] - &b[0] * &a[1];
    }
    acc
}

fn ear_clip(lp: &[[R; 2]]) -> Vec<([R; 2], [R; 2], [R; 2])> {
    let orient = |a: &[R; 2], b: &[R; 2], c: &[R; 2]| -> R {
        (&b[0] - &a[0]) * (&c[1] - &a[1]) - (&b[1] - &a[1]) * (&c[0] - &a[0])
    };
    let inside = |p: &[R; 2], a: &[R; 2], b: &[R; 2], c: &[R; 2]| -> bool {
        orient(a, b, p).is_positive() && orient(b, c, p).is_positive() && orient(c, a, p).is_positive()
    };
    let mut pts: Vec<[R; 2]> = lp.to_vec();
    let mut tris = Vec::new();
    while pts.len() > 3 {
        let n = pts.len();
        let mut found = false;
        for i in 0..n {
            let a = &pts[(i + n - 1) % n];
            let b = &pts[i];
            let c = &pts[(i + 1) % n];
            if !orient(a, b, c).is_positive() {
                continue;
            }
            let mut any = false;
            for (j, p) in pts.iter().enumerate() {
                if j == i || j == (i + n - 1) % n || j == (i + 1) % n {
                    continue;
                }
                if inside(p, a, b, c) {
                    any = true;
                    break;
                }
            }
            if any {
                continue;
            }
            tris.push((a.clone(), b.clone(), c.clone()));
            pts.remove(i);
            found = true;
            break;
        }
        if !found {
            break;
        }
    }
    if pts.len() == 3 {
        tris.push((pts[0].clone(), pts[1].clone(), pts[2].clone()));
    }
    tris
}

fn canon_dir(d: &V) -> Option<V> {
    let mut den = BigInt::one();
    for v in d.iter() {
        den = den.lcm(v.denom());
    }
    let mut ints: Vec<BigInt> = d.iter().map(|v| v.numer() * (&den / v.denom())).collect();
    let mut g = BigInt::zero();
    for v in ints.iter() {
        g = g.gcd(v);
    }
    if g.is_zero() {
        return None;
    }
    for v in ints.iter_mut() {
        *v /= &g;
    }
    for v in ints.iter() {
        if !v.is_zero() {
            if v.is_negative() {
                for w in ints.iter_mut() {
                    *w = -&*w;
                }
            }
            break;
        }
    }
    Some([
        BigRational::from_integer(ints[0].clone()),
        BigRational::from_integer(ints[1].clone()),
        BigRational::from_integer(ints[2].clone()),
    ])
}

// -- chamfer (edge quarter + corner facets, ported from brep.py) -------------

fn parallelepiped(base: &V, e1: &V, e2: &V, e3: &V, source: &str) -> Solid {
    let v0 = base.clone();
    let v1 = add(base, e1);
    let v2 = add(&add(base, e1), e2);
    let v3 = add(base, e2);
    let vs = [
        v0.clone(), v1.clone(), v2.clone(), v3.clone(),
        add(&v0, e3), add(&v1, e3), add(&v2, e3), add(&v3, e3),
    ];
    let faces: [([usize; 4], &str); 6] = [
        ([0, 3, 2, 1], "b"), ([4, 5, 6, 7], "t"), ([0, 1, 5, 4], "f"),
        ([2, 3, 7, 6], "k"), ([1, 2, 6, 5], "r"), ([3, 0, 4, 7], "l"),
    ];
    let s = Solid::new(
        faces.iter().map(|(idx, nm)| Polygon::new(idx.iter().map(|&i| vs[i].clone()).collect(), format!("{}.{}", source, nm))).collect(),
    );
    if s.volume().is_positive() {
        s
    } else {
        Solid { polys: s.polys.iter().map(|p| p.flipped()).collect() }
    }
}

fn tetra(p0: &V, p1: &V, p2: &V, p3: &V, source: &str) -> Solid {
    let s = Solid::new(vec![
        Polygon::new(vec![p0.clone(), p1.clone(), p2.clone()], format!("{}.a", source)),
        Polygon::new(vec![p0.clone(), p2.clone(), p3.clone()], format!("{}.b", source)),
        Polygon::new(vec![p0.clone(), p3.clone(), p1.clone()], format!("{}.c", source)),
        Polygon::new(vec![p1.clone(), p3.clone(), p2.clone()], format!("{}.d", source)),
    ]);
    if s.volume().is_positive() {
        s
    } else {
        Solid { polys: s.polys.iter().map(|p| p.flipped()).collect() }
    }
}

fn solve3(rows: &[V; 3], rhs: &[R; 3]) -> Option<V> {
    let det = dot(&rows[0], &cross(&rows[1], &rows[2]));
    if det.is_zero() {
        return None;
    }
    let col = |i: usize| -> V { [rows[0][i].clone(), rows[1][i].clone(), rows[2][i].clone()] };
    let r = [rhs[0].clone(), rhs[1].clone(), rhs[2].clone()];
    let ax = dot(&r, &cross(&col(1), &col(2))) / &det;
    let ay = dot(&col(0), &cross(&r, &col(2))) / &det;
    let az = dot(&col(0), &cross(&col(1), &r)) / &det;
    Some([ax, ay, az])
}

struct Edge {
    point: V,
    dir: V,
    tmin: R,
    tmax: R,
    plane_a: Plane,
    plane_b: Plane,
}

fn logical_edges(s: &Solid) -> Vec<Edge> {
    // group boundary segments by carrier line; an edge = 2 distinct planes meet
    let mut lines: BTreeMap<String, (Vec<[BigInt; 4]>, Vec<Plane>, R, R, V, V)> = BTreeMap::new();
    for p in &s.polys {
        let n = p.verts.len();
        for i in 0..n {
            let a = &p.verts[i];
            let b = &p.verts[(i + 1) % n];
            if let Some(cd) = canon_dir(&sub(b, a)) {
                let mom = cross(a, &cd);
                let key = format!(
                    "{}/{},{}/{},{}/{}|{}/{},{}/{},{}/{}",
                    cd[0].numer(), cd[0].denom(), cd[1].numer(), cd[1].denom(), cd[2].numer(), cd[2].denom(),
                    mom[0].numer(), mom[0].denom(), mom[1].numer(), mom[1].denom(), mom[2].numer(), mom[2].denom()
                );
                let ta = dot(a, &cd);
                let tb = dot(b, &cd);
                let (lo, hi) = if ta < tb { (ta.clone(), tb.clone()) } else { (tb.clone(), ta.clone()) };
                let e = lines.entry(key).or_insert_with(|| {
                    (Vec::new(), Vec::new(), lo.clone(), hi.clone(), a.clone(), cd.clone())
                });
                let pk = p.plane.canonical();
                if !e.0.contains(&pk) {
                    e.0.push(pk);
                    e.1.push(p.plane.clone());
                }
                if lo < e.2 {
                    e.2 = lo;
                }
                if hi > e.3 {
                    e.3 = hi;
                }
            }
        }
    }
    let mut out = Vec::new();
    for (_, (_, planes, tmin, tmax, point, dir)) in lines {
        if planes.len() == 2 {
            out.push(Edge {
                point,
                dir,
                tmin,
                tmax,
                plane_a: planes[0].clone(),
                plane_b: planes[1].clone(),
            });
        }
    }
    out
}

fn unit_dir(cd: &V) -> Option<V> {
    let nn = &cd[0] * &cd[0] + &cd[1] * &cd[1] + &cd[2] * &cd[2];
    let num = nn.numer().clone();
    let den = nn.denom().clone();
    let rn = isqrt(&num);
    let rd = isqrt(&den);
    if &rn * &rn != num || &rd * &rd != den {
        return None;
    }
    let root = BigRational::new(rn, rd);
    Some([&cd[0] / &root, &cd[1] / &root, &cd[2] / &root])
}

fn chamfer(s: &Solid, dist: R) -> Result<Solid, String> {
    let edges = logical_edges(s);
    let (lo, hi) = s.bbox();
    let extent = (&hi[0] - &lo[0]) + (&hi[1] - &lo[1]) + (&hi[2] - &lo[2]) + ri(1);
    // edge planar cuts
    let mut out = s.clone();
    for e in &edges {
        let pa = &e.plane_a;
        let pb = &e.plane_b;
        let na = pa.unit_normal();
        let nb = pb.unit_normal();
        if na.is_none() || nb.is_none() {
            return Err("chamfer: non-rational face normal (K2)".into());
        }
        let u = &e.dir;
        let p0 = &e.point;
        let mut ca = cross(u, na.as_ref().unwrap());
        if pb.side(&add(p0, &ca)) > 0 {
            ca = neg(&ca);
        }
        let mut cb = cross(u, nb.as_ref().unwrap());
        if pa.side(&add(p0, &cb)) > 0 {
            cb = neg(&cb);
        }
        if pb.side(&add(p0, &ca)) >= 0 || pa.side(&add(p0, &cb)) >= 0 {
            continue; // reflex
        }
        let qa = add(p0, &smul(&dist, &ca));
        let qb = add(p0, &smul(&dist, &cb));
        let span = sub(&qb, &qa);
        if is_zero(&span) {
            continue;
        }
        let mid = smul(&half(), &add(&qa, &qb));
        let toward = sub(p0, &mid);
        let e1 = smul(&(&extent / norm1(u)), u);
        let e2 = smul(&(&extent / norm1(&span)), &span);
        let e3 = smul(&ri(2), &toward);
        let base = sub(&sub(&mid, &smul(&half(), &e1)), &smul(&half(), &e2));
        let tool = parallelepiped(&base, &e1, &e2, &e3, "chamfer");
        out = boolean_impl("cut", &out, &tool);
    }
    // corner facets (industrial semantics)
    let mut at_vertex: BTreeMap<String, Vec<(V, Plane, Plane)>> = BTreeMap::new();
    for e in &edges {
        let nn = dot(&e.dir, &e.dir);
        let t0 = dot(&e.point, &e.dir);
        for (t_end, sign) in [(&e.tmin, 1i64), (&e.tmax, -1i64)] {
            let v = add(&e.point, &smul(&((t_end - &t0) / &nn), &e.dir));
            let key = format!(
                "{}/{},{}/{},{}/{}",
                v[0].numer(), v[0].denom(), v[1].numer(), v[1].denom(), v[2].numer(), v[2].denom()
            );
            at_vertex
                .entry(key)
                .or_default()
                .push((smul(&ri(sign), &e.dir), e.plane_a.clone(), e.plane_b.clone()));
        }
    }
    for (_, incident) in at_vertex {
        if incident.len() != 3 {
            continue;
        }
        let v = {
            let cd = &incident[0].0;
            let nn = dot(cd, cd);
            // recover vertex from any edge: point + ((t_end - t0)/nn) already
            // baked; but we stored dir*sign, not v. Recompute from key instead:
            // easier: reconstruct as intersection later. We approximate v via
            // the first incident edge endpoint captured through solve below.
            let _ = nn;
            // fall through: v is needed; recompute from the three chamfer planes.
            None::<V>
        };
        let _ = v;
        let units: Vec<Option<V>> = incident.iter().map(|(cd, _, _)| unit_dir(cd)).collect();
        if units.iter().any(|u| u.is_none()) {
            continue;
        }
        let u: Vec<V> = units.into_iter().map(|x| x.unwrap()).collect();
        // recover the corner vertex: intersection of the three ORIGINAL face
        // planes incident here (plane_a of each edge shares the corner)
        // Use the three chamfer planes' apex approach from brep.py.
        let m: Vec<V> = (0..3).map(|k| add(&u[(k + 1) % 3], &u[(k + 2) % 3])).collect();
        // we need v; reconstruct from edge endpoints: the vertex is where the
        // three edges meet — take it from the stored incident via the dir and
        // the original edge. Simpler: solve original face planes.
        // Gather the 3 distinct original planes around this corner:
        let mut planes: Vec<Plane> = Vec::new();
        for (_, pa, pb) in &incident {
            for pl in [pa, pb] {
                if !planes.iter().any(|q| q.coplanar_key() == pl.coplanar_key()) {
                    planes.push(pl.clone());
                }
            }
        }
        if planes.len() != 3 {
            continue;
        }
        let vrows = [planes[0].n.clone(), planes[1].n.clone(), planes[2].n.clone()];
        let vrhs = [planes[0].d.clone(), planes[1].d.clone(), planes[2].d.clone()];
        let vtx = match solve3(&vrows, &vrhs) {
            Some(x) => x,
            None => continue,
        };
        let rhs: Vec<R> = (0..3).map(|k| dot(&m[k], &add(&vtx, &smul(&dist, &u[(k + 1) % 3])))).collect();
        let mrows = [m[0].clone(), m[1].clone(), m[2].clone()];
        let mrhs = [rhs[0].clone(), rhs[1].clone(), rhs[2].clone()];
        let apex = match solve3(&mrows, &mrhs) {
            Some(x) => x,
            None => continue,
        };
        // facet corner points: (chamfer_i, chamfer_j, face_k) solves
        let mut pts: Vec<V> = Vec::new();
        let mut ok = true;
        for k in 0..3 {
            let i = (k + 1) % 3;
            let j = (k + 2) % 3;
            let keys_i: Vec<[BigInt; 4]> = vec![incident[i].1.coplanar_key(), incident[i].2.coplanar_key()];
            let shared = if keys_i.contains(&incident[j].1.coplanar_key()) {
                Some(incident[j].1.clone())
            } else if keys_i.contains(&incident[j].2.coplanar_key()) {
                Some(incident[j].2.clone())
            } else {
                None
            };
            match shared {
                Some(sh) => {
                    let prows = [m[i].clone(), m[j].clone(), sh.n.clone()];
                    let prhs = [rhs[i].clone(), rhs[j].clone(), sh.d.clone()];
                    match solve3(&prows, &prhs) {
                        Some(pt) => pts.push(pt),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let tool = tetra(&pts[0], &pts[1], &pts[2], &apex, "corner");
        if tool.volume().is_zero() {
            continue;
        }
        out = boolean_impl("cut", &out, &tool);
    }
    Ok(out)
}

// -- BSP boolean engine ------------------------------------------------------

#[derive(Default)]
struct Split {
    cof: Vec<Polygon>,
    cob: Vec<Polygon>,
    front: Vec<Polygon>,
    back: Vec<Polygon>,
}
/// Float error-filtered plane side: decide the sign in f64 with a static
/// forward-error bound (Shewchuk-style); fall back to the exact rational
/// only when the float magnitude is within the bound of zero. The answer
/// is ALWAYS the exact sign — the filter only skips slow exact arithmetic
/// on the common, unambiguous case (the chamfer/deep-boolean hot path).
fn side_filtered(plane: &Plane, pf: &[f64; 4], fp: &[f64; 3], p: &V) -> i32 {
    let t0 = pf[0] * fp[0];
    let t1 = pf[1] * fp[1];
    let t2 = pf[2] * fp[2];
    let s = t0 + t1 + t2 - pf[3];
    // conservative static bound: a few ulps of the summed magnitudes
    let bound = 16.0 * f64::EPSILON * (t0.abs() + t1.abs() + t2.abs() + pf[3].abs() + 1.0);
    if s.is_finite() {
        if s > bound {
            return 1;
        }
        if s < -bound {
            return -1;
        }
    }
    plane.side(p) // ambiguous or non-finite → exact fallback
}

/// Cache a plane's f64 coefficients once, so `split` (called per polygon
/// against the same node plane) never reconverts BigRationals in the loop.
fn plane_floats(plane: &Plane) -> [f64; 4] {
    [
        plane.n[0].to_f64().unwrap_or(f64::NAN),
        plane.n[1].to_f64().unwrap_or(f64::NAN),
        plane.n[2].to_f64().unwrap_or(f64::NAN),
        plane.d.to_f64().unwrap_or(f64::NAN),
    ]
}

/// Classify `poly` against `plane` and route it into coplanar/front/back.
/// Takes the polygon BY VALUE so the common whole-on-one-side cases MOVE it
/// (no BigRational vertex clone); only a genuinely straddling polygon (kind
/// 3) is decomposed into two freshly built pieces.
fn split(plane: &Plane, pf: &[f64; 4], poly: Polygon) -> Split {
    let mut r = Split::default();
    let sides: Vec<i32> = (0..poly.verts.len())
        .map(|i| side_filtered(plane, pf, &poly.fverts[i], &poly.verts[i]))
        .collect();
    let mut kind = 0;
    for &s in &sides {
        kind |= if s > 0 { 1 } else if s < 0 { 2 } else { 0 };
    }
    match kind {
        0 => {
            if dot(&plane.n, &poly.plane.n).is_positive() {
                r.cof.push(poly);
            } else {
                r.cob.push(poly);
            }
        }
        1 => r.front.push(poly),
        2 => r.back.push(poly),
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
                    r.front.push(p);
                }
            }
            if b.len() >= 3 {
                let p = Polygon::with_plane(b, poly.source.clone(), poly.plane.clone());
                if !p.area2().is_zero() {
                    r.back.push(p);
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
        let pf = plane_floats(&plane);
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in polys {
            let mut s = split(&plane, &pf, p);
            self.polys.append(&mut s.cof);
            self.polys.append(&mut s.cob);
            front.append(&mut s.front);
            back.append(&mut s.back);
        }
        if !front.is_empty() {
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
        let pf = plane_floats(&plane);
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in polys {
            let mut s = split(&plane, &pf, p);
            front.append(&mut s.cof);
            back.append(&mut s.cob);
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
fn boolean_impl(op: &str, a: &Solid, b: &Solid) -> Solid {
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

// -- pyo3 facade: PySolid opaque handle --------------------------------------

fn parse_r(s: &str) -> R {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() == 2 {
        BigRational::new(parts[0].parse().unwrap(), parts[1].parse().unwrap())
    } else {
        BigRational::from_integer(parts[0].parse().unwrap())
    }
}
fn r_str(r: &R) -> String {
    format!("{}/{}", r.numer(), r.denom())
}

#[pyclass]
struct PySolid {
    inner: Solid,
}

#[pymethods]
impl PySolid {
    fn translate(&self, x: &str, y: &str, z: &str) -> PySolid {
        PySolid { inner: self.inner.translated(&[parse_r(x), parse_r(y), parse_r(z)]) }
    }
    fn scale(&self, fx: &str, fy: &str, fz: &str) -> PySolid {
        PySolid { inner: self.inner.scaled(parse_r(fx), parse_r(fy), parse_r(fz)) }
    }
    fn mirror(&self, axis: usize) -> PySolid {
        PySolid { inner: self.inner.mirrored(axis) }
    }
    fn rotate_quarter(&self, axis: usize, quarters: i64) -> PySolid {
        PySolid { inner: self.inner.rotated_quarter(axis, quarters) }
    }
    fn boolean(&self, op: &str, other: &PySolid) -> PySolid {
        PySolid { inner: boolean_impl(op, &self.inner, &other.inner) }
    }
    fn chamfer(&self, dist: &str) -> PyResult<PySolid> {
        chamfer(&self.inner, parse_r(dist))
            .map(|s| PySolid { inner: s })
            .map_err(PyValueError::new_err)
    }
    fn volume(&self) -> f64 {
        self.inner.volume().to_f64().unwrap()
    }
    fn volume6_str(&self) -> String {
        r_str(&self.inner.volume6())
    }
    fn canonical(&self) -> String {
        self.inner.canonical()
    }
    fn watertight_ok(&self) -> bool {
        self.inner.watertight_bad() == 0
    }
    fn logical_faces(&self) -> usize {
        self.inner.logical_face_count()
    }
    fn bbox(&self) -> Vec<f64> {
        let (lo, hi) = self.inner.bbox();
        vec![
            lo[0].to_f64().unwrap(), lo[1].to_f64().unwrap(), lo[2].to_f64().unwrap(),
            hi[0].to_f64().unwrap(), hi[1].to_f64().unwrap(), hi[2].to_f64().unwrap(),
        ]
    }
}

#[pyfunction]
fn make_box(dx: &str, dy: &str, dz: &str, prefix: &str) -> PySolid {
    PySolid { inner: Solid::cube(parse_r(dx), parse_r(dy), parse_r(dz), prefix) }
}

#[pyfunction]
fn make_prism(loop_flat: Vec<String>, h: &str, prefix: &str) -> PySolid {
    // loop_flat = [x0,y0,x1,y1,...] as "n/d" strings
    let mut lp: Vec<[R; 2]> = Vec::new();
    let mut i = 0;
    while i + 1 < loop_flat.len() {
        lp.push([parse_r(&loop_flat[i]), parse_r(&loop_flat[i + 1])]);
        i += 2;
    }
    PySolid { inner: Solid::prism(lp, parse_r(h), prefix) }
}

#[pyfunction]
fn make_prismatoid(bottom: Vec<String>, z0: &str, top: Vec<String>, z1: &str, prefix: &str) -> PySolid {
    let rd = |flat: &[String]| -> Vec<[R; 2]> {
        let mut v = Vec::new();
        let mut i = 0;
        while i + 1 < flat.len() {
            v.push([parse_r(&flat[i]), parse_r(&flat[i + 1])]);
            i += 2;
        }
        v
    };
    PySolid { inner: Solid::prismatoid(rd(&bottom), parse_r(z0), rd(&top), parse_r(z1), prefix) }
}

// legacy oracle helpers (kept for the original tests)
#[pyfunction]
fn box_boolean(
    dx: &str, dy: &str, dz: &str, ex: &str, ey: &str, ez: &str,
    tx: &str, ty: &str, tz: &str, op: &str,
) -> (String, String) {
    let a = Solid::cube(parse_r(dx), parse_r(dy), parse_r(dz), "A");
    let b = Solid::cube(parse_r(ex), parse_r(ey), parse_r(ez), "B")
        .translated(&[parse_r(tx), parse_r(ty), parse_r(tz)]);
    let out = boolean_impl(op, &a, &b);
    (r_str(&out.volume6()), out.canonical())
}
#[pyfunction]
fn box_form(dx: &str, dy: &str, dz: &str) -> (String, String) {
    let s = Solid::cube(parse_r(dx), parse_r(dy), parse_r(dz), "box");
    (r_str(&s.volume6()), s.canonical())
}

// -- K3 Rust port: de Boor NURBS eval + SSI subdivision detection -------------
// Ported from forgekernel/nurbs.py and ssi.py AFTER the Python semantics
// were oracle-settled; Python ref stays the executable specification.

fn deboor_h(p: usize, u_knots: &[R], pts: &[Vec<R>], u: &R) -> Vec<R> {
    let n = pts.len() - 1;
    let dim = pts[0].len();
    let k = if *u >= u_knots[n + 1] {
        n
    } else if *u <= u_knots[p] {
        p
    } else {
        let (mut lo, mut hi) = (p, n + 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if *u < u_knots[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    };
    let mut d: Vec<Vec<R>> = (0..=p).map(|j| pts[k - p + j].clone()).collect();
    for r in 1..=p {
        for j in (r..=p).rev() {
            let i = k - p + j;
            let denom = &u_knots[i + p - r + 1] - &u_knots[i];
            let a = if denom.is_zero() {
                R::zero()
            } else {
                (u - &u_knots[i]) / denom
            };
            let b = R::one() - &a;
            d[j] = (0..dim)
                .map(|c| &b * &d[j - 1][c] + &a * &d[j][c])
                .collect();
        }
    }
    d[p].clone()
}

#[pyfunction]
#[pyo3(signature = (degree, cps, knots, t, weights=None))]
fn nurbs_curve_eval(
    degree: usize, cps: Vec<Vec<String>>, knots: Vec<String>, t: &str,
    weights: Option<Vec<String>>,
) -> Vec<String> {
    let u_knots: Vec<R> = knots.iter().map(|s| parse_r(s)).collect();
    let w: Vec<R> = match &weights {
        Some(ws) => ws.iter().map(|s| parse_r(s)).collect(),
        None => vec![R::one(); cps.len()],
    };
    let pts: Vec<Vec<R>> = cps
        .iter()
        .zip(w.iter())
        .map(|(pt, wi)| {
            let mut h: Vec<R> = pt.iter().map(|c| wi * &parse_r(c)).collect();
            h.push(wi.clone());
            h
        })
        .collect();
    let h = deboor_h(degree, &u_knots, &pts, &parse_r(t));
    (0..3).map(|c| r_str(&(&h[c] / &h[3]))).collect()
}

#[pyfunction]
#[pyo3(signature = (pu, pv, net, ku, kv, u, v, weights=None))]
fn nurbs_surface_eval(
    pu: usize, pv: usize, net: Vec<Vec<Vec<String>>>, ku: Vec<String>,
    kv: Vec<String>, u: &str, v: &str, weights: Option<Vec<Vec<String>>>,
) -> Vec<String> {
    let uk: Vec<R> = ku.iter().map(|s| parse_r(s)).collect();
    let vk: Vec<R> = kv.iter().map(|s| parse_r(s)).collect();
    let vr = parse_r(v);
    // homogeneous rows, de Boor down v per row then across u
    let col: Vec<Vec<R>> = net
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let hrow: Vec<Vec<R>> = row
                .iter()
                .enumerate()
                .map(|(j, pt)| {
                    let wi = match &weights {
                        Some(w) => parse_r(&w[i][j]),
                        None => R::one(),
                    };
                    let mut h: Vec<R> =
                        pt.iter().map(|c| &wi * &parse_r(c)).collect();
                    h.push(wi);
                    h
                })
                .collect();
            deboor_h(pv, &vk, &hrow, &vr)
        })
        .collect();
    let h = deboor_h(pu, &uk, &col, &parse_r(u));
    (0..3).map(|c| r_str(&(&h[c] / &h[3]))).collect()
}

// -- SSI subdivision detection (the hot loop) ---------------------------------

#[derive(Clone)]
struct RPatch {
    net: Vec<Vec<Vec<R>>>, // (p+1)x(q+1) of dim-3 or dim-4 points
    u0: R,
    u1: R,
    v0: R,
    v1: R,
}

impl RPatch {
    fn dim(&self) -> usize {
        self.net[0][0].len()
    }
    fn bbox(&self) -> ([R; 3], [R; 3]) {
        let d = self.dim();
        let mut lo: Option<[R; 3]> = None;
        let mut hi: Option<[R; 3]> = None;
        for row in &self.net {
            for pt in row {
                let cart: [R; 3] = if d == 3 {
                    [pt[0].clone(), pt[1].clone(), pt[2].clone()]
                } else {
                    [&pt[0] / &pt[3], &pt[1] / &pt[3], &pt[2] / &pt[3]]
                };
                match (&mut lo, &mut hi) {
                    (Some(l), Some(h)) => {
                        for c in 0..3 {
                            if cart[c] < l[c] {
                                l[c] = cart[c].clone();
                            }
                            if cart[c] > h[c] {
                                h[c] = cart[c].clone();
                            }
                        }
                    }
                    _ => {
                        lo = Some(cart.clone());
                        hi = Some(cart);
                    }
                }
            }
        }
        (lo.unwrap(), hi.unwrap())
    }
    fn split_rows(rows: &[Vec<Vec<R>>], dim: usize) -> (Vec<Vec<Vec<R>>>, Vec<Vec<Vec<R>>>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        let half = R::new(BigInt::one(), BigInt::from(2));
        for row in rows {
            let mut pts: Vec<Vec<R>> = row.clone();
            let n = pts.len();
            let mut lo = vec![pts[0].clone()];
            let mut hi = vec![pts[n - 1].clone()];
            for r in 1..n {
                for i in 0..n - r {
                    pts[i] = (0..dim)
                        .map(|c| (&pts[i][c] + &pts[i + 1][c]) * &half)
                        .collect();
                }
                lo.push(pts[0].clone());
                hi.push(pts[n - r - 1].clone());
            }
            hi.reverse();
            left.push(lo);
            right.push(hi);
        }
        (left, right)
    }
    fn split4(&self) -> Vec<RPatch> {
        let dim = self.dim();
        let half = R::new(BigInt::one(), BigInt::from(2));
        // split in u: transpose so rows run along u
        let nu = self.net.len();
        let nv = self.net[0].len();
        let cols: Vec<Vec<Vec<R>>> = (0..nv)
            .map(|j| (0..nu).map(|i| self.net[i][j].clone()).collect())
            .collect();
        let (l, r) = Self::split_rows(&cols, dim);
        let um = (&self.u0 + &self.u1) * &half;
        let untrans = |m: &Vec<Vec<Vec<R>>>| -> Vec<Vec<Vec<R>>> {
            let a = m[0].len();
            (0..a)
                .map(|i| (0..m.len()).map(|j| m[j][i].clone()).collect())
                .collect()
        };
        let mk = |net: Vec<Vec<Vec<R>>>, u0: R, u1: R, v0: R, v1: R| RPatch {
            net, u0, u1, v0, v1,
        };
        let a = mk(untrans(&l), self.u0.clone(), um.clone(), self.v0.clone(), self.v1.clone());
        let b = mk(untrans(&r), um, self.u1.clone(), self.v0.clone(), self.v1.clone());
        let mut out = Vec::with_capacity(4);
        for p in [a, b] {
            let (pl, pr) = Self::split_rows(&p.net, dim);
            let vm = (&p.v0 + &p.v1) * &half;
            out.push(mk(pl, p.u0.clone(), p.u1.clone(), p.v0.clone(), vm.clone()));
            out.push(mk(pr, p.u0.clone(), p.u1.clone(), vm, p.v1.clone()));
        }
        out
    }
}

fn boxes_overlap_r(a: &([R; 3], [R; 3]), b: &([R; 3], [R; 3])) -> bool {
    (0..3).all(|c| a.0[c] <= b.1[c] && b.0[c] <= a.1[c])
}

fn parse_net(net: &[Vec<Vec<String>>]) -> Vec<Vec<Vec<R>>> {
    net.iter()
        .map(|row| row.iter().map(|pt| pt.iter().map(|c| parse_r(c)).collect()).collect())
        .collect()
}

/// Subdivision detection: returns surviving leaf-pair parameter boxes
/// (au0,au1,av0,av1,bu0,bu1,bv0,bv1) as exact rational strings. Empty
/// result = certified non-intersection. Clustering/refinement stay in
/// Python (they are not the hot loop).
#[pyfunction]
#[pyo3(signature = (net_a, net_b, depth, a_box=None, b_box=None))]
fn ssi_pairs(
    net_a: Vec<Vec<Vec<String>>>, net_b: Vec<Vec<Vec<String>>>, depth: usize,
    a_box: Option<Vec<String>>, b_box: Option<Vec<String>>,
) -> Vec<Vec<String>> {
    let parse_box = |b: &Option<Vec<String>>| -> (R, R, R, R) {
        match b {
            Some(v) => (parse_r(&v[0]), parse_r(&v[1]), parse_r(&v[2]), parse_r(&v[3])),
            None => (R::zero(), R::one(), R::zero(), R::one()),
        }
    };
    let (au0, au1, av0, av1) = parse_box(&a_box);
    let (bu0, bu1, bv0, bv1) = parse_box(&b_box);
    let a = RPatch { net: parse_net(&net_a), u0: au0, u1: au1, v0: av0, v1: av1 };
    let b = RPatch { net: parse_net(&net_b), u0: bu0, u1: bu1, v0: bv0, v1: bv1 };
    let mut pairs = vec![(a, b)];
    for _ in 0..depth {
        let mut nxt = Vec::new();
        for (pa, pb) in &pairs {
            if !boxes_overlap_r(&pa.bbox(), &pb.bbox()) {
                continue;
            }
            let subs_a = pa.split4();
            let subs_b = pb.split4();
            let boxes_b: Vec<_> = subs_b.iter().map(|s| s.bbox()).collect();
            for sa in subs_a {
                let ba = sa.bbox();
                for (sb, bb) in subs_b.iter().zip(boxes_b.iter()) {
                    if boxes_overlap_r(&ba, bb) {
                        nxt.push((sa.clone(), sb.clone()));
                    }
                }
            }
        }
        pairs = nxt;
        if pairs.is_empty() {
            return Vec::new();
        }
    }
    pairs
        .iter()
        .map(|(pa, pb)| {
            vec![
                r_str(&pa.u0), r_str(&pa.u1), r_str(&pa.v0), r_str(&pa.v1),
                r_str(&pb.u0), r_str(&pb.u1), r_str(&pb.v0), r_str(&pb.v1),
            ]
        })
        .collect()
}

#[pymodule]
fn forgekernel_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolid>()?;
    m.add_function(wrap_pyfunction!(make_box, m)?)?;
    m.add_function(wrap_pyfunction!(make_prism, m)?)?;
    m.add_function(wrap_pyfunction!(make_prismatoid, m)?)?;
    m.add_function(wrap_pyfunction!(box_boolean, m)?)?;
    m.add_function(wrap_pyfunction!(box_form, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surface_eval, m)?)?;
    m.add_function(wrap_pyfunction!(ssi_pairs, m)?)?;
    Ok(())
}
