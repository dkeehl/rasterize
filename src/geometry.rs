use std::ops::{Index, IndexMut, Add, Sub, Mul, Div, Neg};
use std::mem::{self, MaybeUninit};
use std::f32;
use std::fmt::{self, Display};

fn map<T, U, F, const N:usize>(ts: &[T; N], f: F) -> [U; N]
where
    F: Fn(&T) -> U,
    U: 'static + Copy,
{
    let mut ret = MaybeUninit::<[U; N]>::uninit();
    unsafe {
        for (i, t) in ts.iter().enumerate() {
            let ptr = ret.as_mut_ptr() as *mut U;
            let ptr = ptr.offset(i as isize);
            ptr.write(f(t));
        }
        ret.assume_init()
    }
}

pub trait Dot<T=Self> {
    type Output;
    fn dot(self, other: T) -> Self::Output;
}

impl<'a> Dot<&'a [f32; 3]> for &Point3f {
    type Output = f32;

    fn dot(self, other: &'a [f32; 3]) -> Self::Output {
        other[0] * self[0] + other[1] * self[1] + other[2] * self[2]
    }
}

impl<'a, T> Dot<&'a Point3f> for &[T; 3] 
where
    for<'r> &'r T: Mul<f32, Output=T>,
    T: Add<Output=T>,
{
    type Output = T;

    fn dot(self, rhs: &'a Point3f) -> Self::Output {
        &self[0] * rhs[0] + &self[1] * rhs[1] + &self[2] * rhs[2]
    }
}

impl Dot for &Vector3f {
    type Output = f32;

    fn dot(self, other: &Vector3f) -> Self::Output {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

macro_rules! array_wrapper {
    ($name:ident) => {
        #[derive(Clone)]
        pub struct $name<T, const N: usize> {
            pub data: [T; N]
        }

        impl<T, const N: usize> Index<usize> for $name<T, N> {
            type Output = T;

            fn index(&self, i: usize) -> &T {
                &self.data[i]
            }
        }

        impl<T, const N: usize> IndexMut<usize> for $name<T, N> {
            fn index_mut(&mut self, i: usize) -> &mut T {
                &mut self.data[i]
            }
        }

        impl<T> $name<T, 3> {
            pub fn new(x: T, y: T, z: T) -> Self {
                $name { data: [x, y, z] }
            }

            pub fn x(&self) -> &T {
                &self[0]
            }

            pub fn y(&self) -> &T {
                &self[1]
            }

            pub fn z(&self) -> &T {
                &self[2]
            }
        }

        impl<T> $name<T, 2> {
            pub fn x(&self) -> &T {
                &self[0]
            }

            pub fn y(&self) -> &T {
                &self[1]
            }
        }

        impl<T, const N: usize> From<[T; N]> for $name<T, N> {
            fn from(data: [T; N]) -> Self {
                $name { data }
            }
        }

        impl<T> From<(T, T, T)> for $name<T, 3> {
            fn from(data: (T, T, T)) -> Self {
                let (x, y, z) = data;
                [x, y, z].into()
            }
        }

        impl<T> From<(T, T)> for $name<T, 2> {
            fn from(data: (T, T)) -> Self {
                let (x, y) = data;
                [x, y].into()
            }
        }

        impl<const N: usize> Mul<f32> for &$name<f32, N> {
            type Output = $name<f32, N>;

            fn mul(self, scaler: f32) -> Self::Output {
                let data = map(&self.data, |x| x * scaler);
                data.into()
            }
        }

        impl<const N: usize> Mul<f32> for $name<f32, N> {
            type Output = $name<f32, N>;

            fn mul(self, scaler: f32) -> Self::Output {
                <&$name<f32, N>>::mul(&self, scaler)
            }
        }

        impl<const N: usize> Neg for &$name<f32, N> {
            type Output = $name<f32, N>;

            fn neg(self) -> Self::Output {
                let data = map(&self.data, |x| -x);
                data.into()
            }
        }

        impl<const N: usize> Neg for $name<f32, N> {
            type Output = $name<f32, N>;

            fn neg(self) -> Self::Output {
                <&$name<f32, N>>::neg(&self)
            }
        }
    }
}

array_wrapper!(Vector);
array_wrapper!(Point);

impl<'a, T, const N: usize> From<&'a Point<T, N>> for &'a Vector<T, N> {
    fn from(p: &'a Point<T, N>) -> &'a Vector<T, N> {
        unsafe { mem::transmute(p) }
    }
}

pub type Point2i = Point<i32, 2>;
pub type Point3f = Point<f32, 3>;
pub type Point2f = Point<f32, 2>;
pub type Vector2i = Vector<i32, 2>;
pub type Vector3f = Vector<f32, 3>;
pub type Vector2f = Vector<f32, 2>;
pub type Normal = Vector<f32, 3>;

impl Copy for Point2i {}
impl Copy for Point2f {}

macro_rules! derive {
    ($tr:ident, $f:ident, $lhs:ty, $rhs:ty, $o:ty) => {
        impl $tr<$rhs> for $lhs {
            type Output = $o;

            fn $f(self, rhs: $rhs) -> Self::Output {
                <&$lhs>::$f(&self, &rhs)
            }
        }

        impl $tr<$rhs> for &$lhs {
            type Output = $o;

            fn $f(self, rhs: $rhs) -> Self::Output {
                <&$lhs>::$f(self, &rhs)
            }
        }

        impl<'a> $tr<&'a $rhs> for $lhs {
            type Output = $o;

            fn $f(self, rhs: &'a $rhs) -> Self::Output {
                <&$lhs>::$f(&self, rhs)
            }
        }
    }
}

macro_rules! derive_sub {
    ($t:ty, $o:ty) => { derive!(Sub, sub, $t, $t, $o); };
    ($lhs:ty => $rhs:ty => $o:ty) => { derive!(Sub, sub, $lhs, $rhs, $o); }
}

macro_rules! derive_add {
    ($t:ty, $o:ty) => { derive!(Add, add, $t, $t, $o); };
    ($lhs:ty => $rhs:ty => $o:ty) => { derive!(Add, add, $lhs, $rhs, $o); }
}

impl<T> Sub for &Point<T, 2>
where
    for<'a> &'a T: Sub<Output=T>
{
    type Output = Vector<T, 2>;

    fn sub(self, other: Self) -> Self::Output {
        let [ref x0, ref y0] = self.data;
        let [ref x1, ref y1] = other.data;
        [x0 - x1, y0 - y1].into()
    }
}

derive_sub!(Point2i, Vector2i);

impl<T> Sub for &Point<T, 3>
where
    for<'a> &'a T: Sub<Output=T>
{
    type Output = Vector<T, 3>;

    fn sub(self, other: Self) -> Self::Output {
        let [ref x0, ref y0, ref z0] = self.data;
        let [ref x1, ref y1, ref z1] = other.data;
        [x0 - x1, y0 - y1, z0 - z1].into()
    }
}

derive_sub!(Point3f, Vector3f);

impl<'a, T> Add<&'a Vector<T, 3>> for &Point<T, 3>
where
    for<'r> &'r T: Add<Output=T>
{
    type Output = Point<T, 3>;

    fn add(self, rhs: &'a Vector<T, 3>) -> Self::Output {
        let [ref x0, ref y0, ref z0] = self.data;
        let [ref x1, ref y1, ref z1] = rhs.data;
        [x0 + x1, y0 + y1, z0 + z1].into()
    }
}

derive_add!(Point3f => Vector3f => Point3f);

impl<T> Sub for &Vector<T, 3>
where
    for<'a> &'a T: Sub<Output=T>
{
    type Output = Vector<T, 3>;

    fn sub(self, other: Self) -> Self::Output {
        let [ref x0, ref y0, ref z0] = self.data;
        let [ref x1, ref y1, ref z1] = other.data;
        [x0 - x1, y0 - y1, z0 - z1].into()
    }
}

derive_sub!(Vector3f, Vector3f);

impl<T> Add for &Vector<T, 3>
where
    for<'a> &'a T: Add<Output=T>
{
    type Output = Vector<T, 3>;

    fn add(self, other: Self) -> Self::Output {
        let [ref x0, ref y0, ref z0] = self.data;
        let [ref x1, ref y1, ref z1] = other.data;
        [x0 + x1, y0 + y1, z0 + z1].into()
    }
}

derive_add!(Vector3f, Vector3f);

impl Vector3f {
    pub fn cross(&self, other: &Vector3f) -> Vector3f {
        ( self.y() * other.z() - self.z() * other.y(),
          self.z() * other.x() - self.x() * other.z(),
          self.x() * other.y() - self.y() * other.x() ).into()
    }

    pub fn len_squared(&self) -> f32 {
        self.dot(self)
    }

    pub fn len(&self) -> f32 {
        self.len_squared().sqrt()
    }

    pub fn normalize(self) -> Vector3f {
        let len = self.len();
        let mut ret = self;
        if len != 0.0 {
            let len_inv = len.recip();
            for x in ret.data.iter_mut() {
                *x *= len_inv;
            }
        }
        ret
    }
}

impl<T> Display for Point<T, 3>
where
    T: Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self[0], self[1], self[2])
    }
}

pub struct Bounds<T, const N: usize> {
    min: Point<T, N>,
    max: Point<T, N>,
}

impl<T, const N: usize> Bounds<T, N> {
    pub fn destruct(self) -> (Point<T, N>, Point<T, N>) {
        (self.min, self.max)
    }
}

impl<T: PartialOrd + Copy, const N: usize> Bounds<T, N> {
    pub fn new(p0: Point<T, N>, p1: Point<T, N>) -> Self {
        let mut min = p0;
        let mut max = p1;
        for i in 0..N {
            if min[i] > max[i] {
                mem::swap(&mut min[i], &mut max[i]);
            }
        }
        Bounds { min, max }
    }

    pub fn add_point(&mut self, p: Point<T, N>) {
        for i in 0..N {
            if p[i] < self.min[i] {
                self.min[i] = p[i];
            }
            if p[i] > self.max[i] {
                self.max[i] = p[i];
            }
        }
    }
}

type Bounds2<T> = Bounds<T, 2>;
type Bounds3<T> = Bounds<T, 3>;

pub type Bounds2i = Bounds2<i32>;
pub type Bounds2f = Bounds2<f32>;
pub type Bounds3f = Bounds3<f32>;

impl Bounds2i {
    pub fn points_inclusive(&self) -> PointsInclusive<'_> {
        assert!(self.min[0] <= self.max[0] && self.min[1] <= self.max[1]);
        PointsInclusive {
            bounds: self,
            x: self.min[0],
            y: self.min[1],
        }
    }
}

pub struct PointsInclusive<'a> {
    bounds: &'a Bounds2i,
    x: i32,
    y: i32,
}

impl<'a> Iterator for PointsInclusive<'a> {
    type Item=Point2i;

    fn next(&mut self) -> Option<Self::Item> {
        // FIXME Consider overflow
        if self.y > self.bounds.max[1] {
            None
        } else {
            let p = (self.x, self.y).into();
            if self.x == self.bounds.max[0] {
                self.x = self.bounds.min[0];
                self.y += 1;
            } else {
                self.x += 1;
            }
            Some(p)
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn map_test() {
        let a = [1, 2, 3];
        let b = super::map(&a, |n| n + 1);
        assert_eq!(b, [2, 3, 4]);
    }
}

