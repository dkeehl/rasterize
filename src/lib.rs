#![feature(const_generics)]
#![allow(unused, incomplete_features)]

mod tga; 
#[macro_use]
mod geometry;
mod transform;
pub mod model;
pub mod gui;

pub use tga::{TGAColor, TGAImage, Buffer, Rate};
pub use geometry::{Point2i, Vector2i, Point2f, Point3f, Vector3f, Normal, Bounds2i, Bounds2f,
    Dot, Point};
use model::Model;
use transform::{m, Transform, Trans};

pub fn line(p0: Point2i, p1: Point2i, img: &mut Buffer, color: TGAColor) {
    let slope = Slope::new(p0[0], p0[1], p1[0], p1[1]);
    for p in slope.points() {
        img.set(&p, color);
    }
}

struct Slope {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    swap_axis: bool,
}

impl Slope {
    fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Slope {
        let (x0, x1, y0, y1, swap_axis) = if (x0 - x1).abs() > (y0 - y1).abs() {
            (x0, x1, y0, y1, false)
        } else {
            (y0, y1, x0, x1, true)
        };
        // make sure p0 on the left of p1
        let (x0, y0, x1, y1) = if x0 < x1 {
            (x0, y0, x1, y1)
        } else {
            (x1, y1, x0, y0)
        };
        Slope { x0, x1, y0, y1, swap_axis }
    }

    fn points(self) -> Points {
        let x_pos = self.x0;
        let y_pos = self.y0;
        let dx = self.x1 - self.x0;
        let dy = self.y1 - self.y0;
        let y_step = if dy > 0 { 1 } else { -1 };
        Points {
            slope: self,
            x_pos,
            y_pos,
            dx,
            y_step,
            error: 0,
            e_step: dy.abs() * 2,
        }
    }
}

struct Points {
    slope: Slope,
    x_pos: i32,
    y_pos: i32,
    dx: i32,
    y_step: i32,
    error: i32,
    e_step: i32,
}

impl Iterator for Points {
    type Item = Point2i;

    fn next(&mut self) -> Option<Self::Item> {
        if self.x_pos > self.slope.x1 {
            None
        } else {
            let ret = if self.slope.swap_axis {
                (self.y_pos, self.x_pos)
            } else {
                (self.x_pos, self.y_pos)
            };
            self.x_pos += 1;
            self.error += self.e_step;
            if self.error > self.dx {
                self.y_pos += self.y_step;
                self.error -= self.dx * 2;
            }
            Some(ret.into())
        }
    }
}

fn barycentric_i(a: &Point2i, b: &Point2i, c: &Point2i, p: &Point2i) -> Point3f {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let det = ab.x() * ac.y() - ab.y() * ac.x();
    if det != 0 {
        let u = ap.x() * ac.y() - ap.y() * ac.x();
        let v = ap.y() * ab.x() - ap.x() * ab.y();
        let x = (det - u - v) as f32 / det as f32;
        let det = det as f32;
        let y = u as f32 / det;
        let z = v as f32 / det;
        (x, y, z).into()
    } else {
        (-1.0, 1.0, 1.0).into()
    }
}

pub fn triangle<T: Shader>(pts: &[Point3f; 3], shader: &T, buf: &mut Buffer) {
    let a = buf.screen_to_raster(&pts[0]);
    let b = buf.screen_to_raster(&pts[1]);
    let c = buf.screen_to_raster(&pts[2]);
    let mut bounds = Bounds2i::new(a, b);
    bounds.add_point(c);
    for p in bounds.points_inclusive() {
        let b = barycentric_i(&a, &b, &c, &p);
        let include = b[0] >= 0.0 && b[1] >= 0.0 && b[2] >= 0.0;
        if include {
            if let Some(mut pos) = buf.position(&p) {
                let z = pts[0][2] * b[0] + pts[1][2] * b[1] + pts[2][2] * b[2];
                if z > pos.z_buffer_get() {
                    if let Some(color) = shader.fragment(&b) {
                        pos.set_color(color);
                        pos.z_buffer_set(z);
                    }
                }
            }
        }
    }
}

pub struct Camera {
    world_to_screen: Transform,
    world_normal_to_screen: Transform,
    world_to_camera: Transform,
    r: f32,
}

impl Camera {
    pub fn new(pos: Point3f, look_at: Vector3f, up: Vector3f, eye: f32) -> Option<Camera>
    {
        // look_at should be finite and nonzero
        let z = -look_at.normalize();
        let x = up.cross(&z).normalize();
        // x should be finite and nonzero
        let y = z.cross(&x).normalize();
        let basis_change = Transform::new(m([
            [x[0], x[1], x[2], 0.0],
            [y[0], y[1], y[2], 0.0],
            [z[0], z[1], z[2], 0.0],
            [ 0.0,  0.0,  0.0, 1.0] ]));
        let origin_change = Transform::translate(-pos[0], -pos[1], -pos[2]);
        let world_to_camera = basis_change * origin_change;
        let r = -eye.recip();
        let camera_to_screen = camera_to_screen(eye);
        let world_to_screen = camera_to_screen * world_to_camera.clone();
        let world_normal_to_screen = world_to_screen.inverse_transpose()?;
        Some(Camera {
            world_to_screen,
            world_normal_to_screen,
            world_to_camera,
            r,
        })
    }

    // Given a barycentric coordinate in screen space, fix it to the camera/world space.
    // pts are in camera space
    // TODO Here we need only z coordinates of the points
    pub fn fix_baryc(&self, baryc: &Point3f, pts: &[Point3f; 3]) -> Point3f {
        let a = baryc[0] / (self.r * pts[0].z() + 1.0);
        let b = baryc[1] / (self.r * pts[1].z() + 1.0);
        let c = baryc[2] / (self.r * pts[2].z() + 1.0);
        let sum = (a + b + c).recip();
        p!(a * sum, b * sum, c * sum)
    }
}

// _eye_ the z position in camera space, equals ctan(fov / 2.0)
fn camera_to_screen(eye: f32) -> Transform {
    let e32 = -eye.recip();
    Transform::new(m([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, e32, 1.0] ]))
}

pub fn clamp<T: PartialOrd>(v: T, min: T, max: T) -> T {
    assert!(min <= max);
    let mut ret = v;
    if ret < min {
        ret = min;
    } else if ret > max {
        ret = max;
    }
    ret
}

pub trait Shader {
    fn vertex(&mut self, f: usize, v: usize) -> Point3f;

    fn fragment(&self, baryc: &Point3f) -> Option<TGAColor>;
}

pub struct Plane<'a> {
    model: &'a Model,
    camera: &'a Camera,
    varying_i: [f32; 3],
    light_dir: Vector3f,
}

impl<'a> Shader for Plane<'a>
{
    fn vertex(&mut self, f: usize, v: usize) -> Point3f {
        let vt = self.model.vertex(f, v);
        let intensity = vt.normal.unwrap().dot(&self.light_dir).max(0.0);
        self.varying_i[v] = intensity;
        self.camera.world_to_screen.trans(vt.position)
    }

    fn fragment(&self, _: &Point3f) -> Option<TGAColor> {
        let i = self.varying_i.iter().sum::<f32>() / 3.0 * 255.0;
        let i = i as u8;
        let color = TGAColor::rgba(i, i, i, 255);
        Some(color)
    }
}

pub struct Gouraud<'a, 'b, D> {
    varying_p: [Point3f; 3],
    varying_intensity: [f32; 3],
    varying_u: [f32; 3],
    varying_v: [f32; 3],
    i: usize,
    light_dir: Vector3f,
    model: &'a Model,
    camera: &'b Camera,
    diffuse: D,
}

impl<'a, 'b, D> Gouraud<'a, 'b, D> {
    pub fn new(model: &'a Model, camera: &'b Camera, light_dir: Vector3f, diffuse: D) -> Self {
        Gouraud {
            varying_p: [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()],
            varying_intensity: [0.0; 3],
            varying_u: [0.0; 3],
            varying_v: [0.0; 3],
            i: 0,
            light_dir: light_dir.normalize(),
            model,
            camera,
            diffuse,
        }
    }
}

impl<'a, 'b, D: Texture<TGAColor>> Shader for Gouraud<'a, 'b, D> {
    fn vertex(&mut self, f: usize, v: usize) -> Point3f {
        let vt = self.model.vertex(f, v);
        let intensity = vt.normal.unwrap().dot(&self.light_dir).max(0.0);
        self.varying_intensity[self.i] = intensity;
        let uv = vt.texture.unwrap();
        self.varying_u[self.i] = uv[0];
        self.varying_v[self.i] = uv[1];
        self.varying_p[self.i] = self.camera.world_to_camera.trans(vt.position);
        self.i = (self.i + 1) % 3;
        self.camera.world_to_screen.trans(vt.position)
    }

    fn fragment(&self, baryc: &Point3f) -> Option<TGAColor> {
        let baryc = self.camera.fix_baryc(baryc, &self.varying_p);
        let i = self.varying_intensity.dot(&baryc);
        let i = clamp(i, 0.0, 255.0);
        let u = self.varying_u.dot(&baryc);
        let v = self.varying_v.dot(&baryc);
        let c = self.diffuse.eval(u, v);
        let color = c * Rate::unchecked(i);
        Some(color)
    }
}

pub trait Texture<T> {
    fn eval(&self, u: f32, v: f32) -> T;
}

impl Texture<TGAColor> for TGAImage {
    fn eval(&self, u: f32, v: f32) -> TGAColor {
        let uv = self.ndc_to_raster(u, v);
        self.get(&uv)
    }
}

impl Texture<Normal> for TGAImage {
    fn eval(&self, u: f32, v: f32) -> Normal {
        let uv = self.ndc_to_raster(u, v);
        let c = self.get(&uv).unpack();
        let to_f32 = |i| i as f32 / 255.0 * 2.0 - 1.0;
        [to_f32(c.r), to_f32(c.g), to_f32(c.b)].into()
    }
}

impl Texture<i32> for TGAImage {
    fn eval(&self, u: f32, v: f32) -> i32 {
        let uv = self.ndc_to_raster(u, v);
        let c = self.get(&uv).unpack();
        c.b as i32
    }
}

pub struct Constant<T> {
    val: T 
}

impl<T: Clone> Texture<T> for Constant<T> {
    fn eval(&self, _: f32, _: f32) -> T {
        self.val.clone()
    }
}

impl<T> Constant<T> {
    pub fn new(val: T) -> Self {
        Constant { val }
    }
}

pub struct Shadow<'a> {
    camera: &'a Camera,
    model: &'a Model,
    varying_tri: [Point3f; 3],
}

impl<'a> Shadow<'a> {
    pub fn new(camera: &'a Camera, model: &'a Model) -> Self {
        Shadow {
            camera,
            model,
            varying_tri: [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()],
        }
    }
}

impl<'a> Shader for Shadow<'a> {
    fn vertex(&mut self, fi: usize, vi: usize) -> Point3f {
        let vt = self.model.vertex(fi, vi);
        let ret = self.camera.world_to_screen.trans(vt.position);
        self.varying_tri[vi] = ret.clone();
        ret
    }

    fn fragment(&self, baryc: &Point3f) -> Option<TGAColor> {
        let p = barycentric_point(&self.varying_tri, baryc);
        let y = clamp(255.0 * p.z(), 0.0, 255.0) as u8;
        Some(TGAColor::rgba(y, y, y, 255))
    }
}

fn barycentric_point(pts: &[Point3f; 3], baryc: &Point3f) -> Point3f {
    let f = |i| <&Vector3f>::from(&pts[i]) * baryc[i];
    Point::from([0.0, 0.0, 0.0]) + (f(0) + f(1) + f(2))
}

pub struct Phong<'a, D, N, S, A> {
    i: usize,
    varying_u: [f32; 3],
    varying_v: [f32; 3],
    varying_p: [Point3f; 3],
    varying_p_cam: [Point3f; 3],
    varying_n: [Normal; 3],
    model: &'a Model,
    camera: &'a Camera,
    light_dir: Vector3f,
    diffuse: D,
    normal_map: N,
    spec: S,
    ambient: A,
    light: &'a Camera,
    shadow_z: &'a Buffer,
}

impl<'a> Phong<'a, (), (), (), ()> {
    pub fn new(
        model: &'a Model,
        camera: &'a Camera,
        light_dir: Vector3f,
        ambient: f32,
        light: &'a Camera,
        shadow_z: &'a Buffer,
        diffuse_path: &str, nm_path: &str, spec_path: &str) -> Option<impl Shader + 'a>
    {
        let diffuse = TGAImage::read_file(diffuse_path)?;
        let normal_map = TGAImage::read_file(nm_path)?;
        let spec = TGAImage::read_file(spec_path)?;
        Some(Phong {
            i: 0,
            varying_u: [0.0; 3],
            varying_v: [0.0; 3],
            varying_p: [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()],
            varying_p_cam: [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()],
            varying_n: [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()],
            model,
            camera,
            light_dir: light_dir.normalize(),
            diffuse,
            normal_map,
            spec,
            ambient: Constant::new(ambient),
            light,
            shadow_z,
        })
    }
}

impl<'a, D, N, S, A> Shader for Phong<'a, D, N, S, A>
where
    D: Texture<TGAColor>,
    N: Texture<Normal>,
    S: Texture<i32>,
    A: Texture<f32>,
{
    fn vertex(&mut self, fi: usize, vi: usize) -> Point3f {
        let vt = self.model.vertex(fi, vi);
        let uv = vt.texture.unwrap();
        self.varying_u[self.i] = uv[0];
        self.varying_v[self.i] = uv[1];
        self.varying_p[self.i] = vt.position.clone();
        self.varying_n[self.i] = vt.normal.unwrap().clone();
        self.varying_p_cam[self.i] = self.camera.world_to_camera.trans(vt.position);
        self.i = (self.i + 1) % 3;
        self.camera.world_to_screen.trans(vt.position)
    }

    fn fragment(&self, baryc: &Point3f) -> Option<TGAColor> {
        let baryc = self.camera.fix_baryc(baryc, &self.varying_p_cam);
        let u = self.varying_u.dot(&baryc);
        let v = self.varying_v.dot(&baryc);
        let c = self.diffuse.eval(u, v).unpack();

        let pts = &self.varying_p;
        // reconstruct p in world space
        let p = barycentric_point(pts, &baryc);
        let p = self.light.world_to_screen.trans(&p);
        let pos = self.shadow_z.screen_to_raster(&p);
        let direct = self.shadow_z.z_buffer_get(&pos)
            .map(|z| p[2] + 0.12 >= z)
            .unwrap_or(false);
        let shadow = 0.3 + 0.7 * direct as u8 as f32;

        let gn = self.varying_n.dot(&baryc).normalize();
        let [t, b] = darboux(&gn, pts, self.varying_u, self.varying_v).unwrap();

        let local = self.normal_map.eval(u, v);
        let n = (t * local[0] + b * local[1] + gn * local[2]).normalize();
        let d = n.dot(&self.light_dir).max(0.0);
        let r = &n * (n.dot(&self.light_dir) * 2.0) - &self.light_dir;
        let r = self.camera.world_to_screen.trans(r).normalize();
        let a = self.spec.eval(u, v);
        let s = r.z().min(0.0).powi(a);
        let amb = self.ambient.eval(u, v);
        const SPECULAR_COE: f32 = 0.0;
        const DIFFUSE_COE: f32 = 1.2;
        let f = |x| {
            let x = amb + x as f32 * shadow * (DIFFUSE_COE * d + SPECULAR_COE * s);
            clamp(x, 0.0, 255.0) as u8
        };
        Some(TGAColor::rgba(f(c.r), f(c.g), f(c.b), c.a))
    }
}

fn darboux(normal: &Normal, pts: &[Point3f; 3], u: [f32; 3], v: [f32; 3])
    -> Option<[Normal; 2]>
{
    let du1 = u[1] - u[0];
    let du2 = u[2] - u[0];
    let dv1 = v[1] - v[0];
    let dv2 = v[2] - v[0];
    let lu = m([[du1, dv1], [du2, dv2]]).lu_decompose()?;
    let mut tmp: [Normal; 2] = [[0.0; 3].into(), [0.0; 3].into()];
    for i in 0..3 {
        let [j, k] = lu.solve(
            &[pts[1][i] - pts[0][i], pts[2][i] - pts[0][i]]);
        tmp[0][i] = j;
        tmp[1][i] = k;
    }
    let [dpdu, _] = tmp;
    let b = normal.cross(&dpdu).normalize();
    let t = b.cross(normal);
    Some([t, b])
}

#[cfg(test)]
mod test {
    use super::Camera;

    #[test]
    fn new_camera() {
        let c = 3.0;
        let cam = Camera::new(
            (0.0, 0.0, 0.0).into(),
            (0.0, 0.0, -1.0).into(),
            (0.0, 1.0, 0.0).into(),
            c).unwrap();
        let p: crate::geometry::Point3f = (0.163784, -0.126996, 0.470144).into();
        let p_ = cam.world_to_screen.trans(&p);
        let scaler = (1.0 - p[2] / c).recip();
        let x = p[0] * scaler;
        let y = p[1] * scaler;
        assert_eq!(p_[0], x);
        assert_eq!(p_[1], y);
    }
}

