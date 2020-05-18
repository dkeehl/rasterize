#![allow(unused)]
use rasterizer::{Point3f, Vector3f, Buffer, Camera, Shader, *};
use rasterizer::model::Model;

fn main() {
    let img = lesson_3();
    // let img = shadow_test();
    img.write_file("output.tga").unwrap();
}

const W: u16 = 800;
const H: u16 = 800;

fn lesson_3() -> Buffer {
    let model = Model::read_file("obj/african_head/african_head.obj").unwrap();
    let up = Vector3f::new(0.0, 1.0, 0.0);
    let light_dir = Vector3f::new(1.0, 1.0, 1.0);
    let center = Point3f::new(0.0, 0.0, 0.0);
    let light = Camera::new(center.clone(), -&light_dir, up.clone(), f32::INFINITY).unwrap();
    let mut shadow = Buffer::new(W, H);
    let mut shader = rasterizer::Shadow::new(&light, &model);

    draw(&mut shader, &model, &mut shadow);

    let mut buffer = Buffer::new(W, H);
    let cam = Camera::new(center, [-1.0, -1.0, -3.0].into(), up, 4.0).unwrap();

    let mut shader = rasterizer::Phong2::new(&model, &cam, light_dir, 20.0, &light, &shadow,
        "obj/african_head/african_head_diffuse.tga",
        "obj/african_head/african_head_nm_tangent.tga",
        "obj/african_head/african_head_spec.tga").unwrap();
    // let mut shader = rasterizer::Gouraud::new(&model, &cam, light_dir);
    draw(&mut shader, &model, &mut buffer);
    buffer
}

fn draw<T: Shader>(shader: &mut T, model: &Model, buffer: &mut Buffer) {
    let mut pts = [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()];
    for fi in 0..model.polygons().len() {
        for (vi, v) in pts.iter_mut().enumerate() {
            *v = shader.vertex(fi, vi);
        }
        triangle(&pts, shader, buffer);
    }
}

fn shadow_test() -> Buffer {
    let model = Model::read_file("obj/african_head/african_head.obj").unwrap();
    let up = Vector3f::new(0.0, 1.0, 0.0);
    let light_dir = Vector3f::new(1.0, 1.0, 1.0);
    let center = Point3f::new(0.0, 0.0, 0.0);
    let light = Camera::new(center, -light_dir, up, f32::INFINITY).unwrap();
    let mut shadow = Buffer::new(W, H);
    let mut shader = rasterizer::Shadow::new(&light, &model);

    draw(&mut shader, &model, &mut shadow);
    shadow
}

