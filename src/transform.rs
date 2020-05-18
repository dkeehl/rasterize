use crate::geometry::{Vector3f, Point3f};
use std::ops::{Mul, Index, IndexMut};
use std::fmt;
use std::cmp::Ordering::{self, *};

#[derive(Clone)]
pub struct SquareMatrix<const N: usize> {
    data: [[f32; N]; N]
}

fn float_ord(a: f32, b: f32) -> Ordering {
    debug_assert!(!a.is_nan() && !b.is_nan());
    if a < b {
        Less
    } else if a > b {
        Greater
    } else {
        Equal
    }
}

impl<const N: usize> SquareMatrix<N> {
    pub fn lu_decompose(&self) -> Option<LU<N>> {
        fn permute_row<const N: usize>(
            m: &mut SquareMatrix<N>, p: &mut SquareMatrix<N>, at: usize)
        {
            let i = (at..N).max_by(|&i, &j| float_ord(m[i][at].abs(), m[j][at].abs()));
            if let Some(i) = i {
                m.data.swap(at, i);
                p.data.swap(at, i);
            }
        }

        let mut lu = self.clone();
        let mut p = identity();
        for i in 1..N {
            permute_row(&mut lu, &mut p, i - 1);
            let a = lu[i - 1][i - 1];
            let a_inv = if a == 0.0 {
                return None;
            } else {
                a.recip()
            };

            for r in i..N {
                lu[r][i - 1] *= a_inv;
                if lu[r][i - 1] != 0.0 {
                    for c in i..N {
                        lu[r][c] -= lu[i - 1][c] * lu[r][i - 1];
                    }
                }
            }
        }
        Some(LU { lu, p })
    }

    fn diff(&self, other: &SquareMatrix<N>) -> (f32, f32) {
        let mut sum = 0.0;
        let mut sqr_sum = 0.0;
        for i in 0..N {
            for j in 0..N {
                let d = self[i][j] - other[i][j];
                sum += d;
                sqr_sum += d * d;
            }
        }
        let c = (N * N) as f32;
        (sum / c, sqr_sum / c)
    }
}

pub type Matrix4x4 = SquareMatrix<4>;

impl Matrix4x4 {
    pub fn transpose(&self) -> Matrix4x4 {
        let a = &self.data;
        m([[a[0][0], a[1][0], a[2][0], a[3][0]],
           [a[0][1], a[1][1], a[2][1], a[3][1]],
           [a[0][2], a[1][2], a[2][2], a[3][2]],
           [a[0][3], a[1][3], a[2][3], a[3][3]] ])
    }

    pub fn inverse_transpose(&self) -> Option<Matrix4x4> {
        let lu = self.lu_decompose()?;
        let x = lu.solve(&[1.0, 0.0, 0.0, 0.0]);
        let y = lu.solve(&[0.0, 1.0, 0.0, 0.0]);
        let z = lu.solve(&[0.0, 0.0, 1.0, 0.0]);
        let w = lu.solve(&[0.0, 0.0, 0.0, 1.0]);
        Some(m([x, y, z, w]))
    }

    pub fn inverse(&self) -> Option<Matrix4x4> {
        self.inverse_transpose().map(|m| m.transpose())
    }
}

impl<const N: usize> fmt::Display for SquareMatrix<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..N {
            write!(f, "[");
            for j in 0..N {
                write!(f, "{:>8.3} ", self[i][j]);
            }
            write!(f, "]\n");
        }
        Ok(())
    }
}

pub struct LU<const N: usize> {
    lu: SquareMatrix<N>,
    p: SquareMatrix<N>,
}

impl<const N: usize> LU<N> {
    pub fn solve(&self, b: &[f32; N]) -> [f32; N] {
        let mut ret = &self.p * b;
        for i in 0..N {
            for j in 0..i {
                ret[i] -= self.lu[i][j] * ret[j];
            }
        }
        for i in (0..N).rev() {
            for j in (i + 1..N).rev() {
                ret[i] -= self.lu[i][j] * ret[j];
            }
            ret[i] /= self.lu[i][i];
        }
        ret
    }
}

pub fn m<const N: usize>(data: [[f32; N]; N]) -> SquareMatrix<N> {
    SquareMatrix { data }
}

pub fn identity<const N: usize>() -> SquareMatrix<N> {
    let mut data = [[0.0; N]; N];
    for i in 0..N {
        data[i][i] = 1.0
    }
    SquareMatrix { data }
}

impl<'a, const N: usize> Mul<&'a [f32; N]> for &SquareMatrix<N> {
    type Output = [f32; N];

    fn mul(self, rhs: &'a [f32; N]) -> Self::Output {
        let mut ret = [0.0; N];
        for (x, row) in ret.iter_mut().zip(self.data.iter()) {
            *x = row.iter().zip(rhs.iter()).fold(0.0, |acc, (a, b)| acc + a * b);
        }
        ret
    }
}

impl Mul for &Matrix4x4 {
    type Output = Matrix4x4;

    fn mul(self, other: &Matrix4x4) -> Self::Output {
        let a = &self.data;
        let b = &other.data;
        let elem = |i: usize, j: usize| -> f32 {
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j]
        };
        m([[elem(0, 0), elem(0, 1), elem(0, 2), elem(0, 3)],
           [elem(1, 0), elem(1, 1), elem(1, 2), elem(1, 3)],
           [elem(2, 0), elem(2, 1), elem(2, 2), elem(2, 3)],
           [elem(3, 0), elem(3, 1), elem(3, 2), elem(3, 3)]])
    }
}

impl<const N: usize> Index<usize> for SquareMatrix<N> {
    type Output = [f32; N];

    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl<const N: usize> IndexMut<usize> for SquareMatrix<N> {
    fn index_mut(&mut self, i: usize) -> &mut [f32; N] {
        &mut self.data[i]
    }
}

#[derive(Clone)]
pub struct Transform {
    m: Matrix4x4
}

impl Transform {
    pub fn new(m: Matrix4x4) -> Transform {
        Transform { m }
    }

    pub fn identity() -> Transform {
        Transform { m: identity() }
    }

    pub fn translate(x: f32, y: f32, z: f32) -> Transform {
        let m = m([
            [1.0, 0.0, 0.0,  x ],
            [0.0, 1.0, 0.0,  y ],
            [0.0, 0.0, 1.0,  z ],
            [0.0, 0.0, 0.0, 1.0] ]);
        Transform { m }
    }

    pub fn inverse_transpose(&self) -> Option<Transform> {
        self.m.inverse_transpose().map(|m| Transform { m })
    }

    pub fn trans<T: Trans>(&self, t: T) -> <T as Trans>::Output {
        t.trans(self)
    }
}

impl Mul for Transform {
    type Output = Transform;

    fn mul(self, other: Transform) -> Self::Output {
        let m = &self.m * &other.m;
        Transform { m }
    }
}

pub trait Trans {
    type Output;
    fn trans(self, t: &Transform) -> Self::Output;
}

impl Trans for &Vector3f {
    type Output = Vector3f;

    fn trans(self, t: &Transform) -> Vector3f {
        let m = &t.m.data;
        let f = |i: usize| m[i][0] * self[0] + m[i][1] * self[1] + m[i][2] * self[2];
        [f(0), f(1), f(2)].into()
    }
}

impl Trans for Vector3f {
    type Output = Vector3f;

    fn trans(self, t: &Transform) -> Vector3f {
        <&Vector3f>::trans(&self, t)
    }
}

impl Trans for &Point3f {
    type Output = Point3f;

    fn trans(self, t: &Transform) -> Point3f {
        let m = &t.m.data;
        let f = |i: usize| m[i][0] * self[0] + m[i][1] * self[1] + m[i][2] * self[2] + m[i][3];
        let w = f(3).recip();
        [f(0) * w, f(1) * w, f(2) * w].into()
    }
}

impl Trans for Point3f {
    type Output = Point3f;

    fn trans(self, t: &Transform) -> Point3f {
        <&Point3f>::trans(&self, t)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn inverse_test() {
        let matrix = m([
            [0.3, -0.2, 10.0, 0.0],
            [3.0, -0.1, -0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 7.0, -0.3, 0.0],
        ]);
        let inv = matrix.inverse().unwrap();
        let matrix = &matrix * &inv;
        let i = identity();
        println!("{:?}", matrix.diff(&i));
    }

    #[test]
    fn solve_test() {
        let matrix = m([
            [3.0, -0.1, -0.2],
            [0.1, 7.0, -0.3],
            [0.3, -0.2, 10.0] ]);
        let res = matrix.lu_decompose().unwrap().solve(&[7.85, -19.3, 71.4]);
        // [3.0, -2.5, 7]
        println!("{:?}", res);
    }
}

