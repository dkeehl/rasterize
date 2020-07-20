#![allow(unused)]
use rasterizer::*;
use rasterizer::model::Model;

fn main() {
    // let img = floor();
    // let img = head();
    let img = plane();
    img.write_file("output.tga").unwrap();
    // rasterizer::gui::view(W as u32, H as u32, img.into_raw_data())
}

const W: u16 = 640;
const H: u16 = 640;

fn draw<T: Shader>(shader: &mut T, model: &Model, buffer: &mut Buffer) {
    let mut pts = [[0.0; 3].into(), [0.0; 3].into(), [0.0; 3].into()];
    for (fi, p) in model.polygons().iter().enumerate() {
        for t in p.triangles() {
            for (i, v) in pts.iter_mut().enumerate() {
                *v = shader.vertex(fi, t[i]);
            }
            // println!("{}:\na: {}\nb: {}\nc: {}", fi, pts[0], pts[1], pts[2]);
            triangle(&pts, shader, buffer);
        }
    }
}

fn head() -> Buffer {
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

    let mut shader = rasterizer::Phong::new(&model, &cam, light_dir, 20.0, &light, &shadow,
        "obj/african_head/african_head_diffuse.tga",
        "obj/african_head/african_head_nm_tangent.tga",
        "obj/african_head/african_head_spec.tga").unwrap();
    draw(&mut shader, &model, &mut buffer);
    buffer
}

fn head0() -> Buffer {
    let model = Model::read_file("obj/african_head/african_head.obj").unwrap();
    let up = Vector3f::new(0.0, 1.0, 0.0);
    let light_dir = Vector3f::new(1.0, 1.0, 1.0);
    let center = Point3f::new(0.0, 0.0, 0.0);

    let mut buffer = Buffer::new(W, H);
    let cam = Camera::new(center, [0.0, 0.0, -1.0].into(), up, 4.0).unwrap();

    // let texture = Constant::new(TGAColor::rgba(255, 255, 255, 255));
    let texture = TGAImage::read_file("obj/african_head/african_head_diffuse.tga").unwrap();
    let mut shader = rasterizer::Gouraud::new(&model, &cam, light_dir, texture);
    draw(&mut shader, &model, &mut buffer);
    buffer
}

fn plane() -> Buffer {
    let model = Model::read_file("obj/StarSparrow/StarSparrow01.obj").unwrap();
    let up = Vector3f::new(0.0, 1.0, 0.0);
    let light_dir = Vector3f::new(1.0, 1.0, 1.0);
    let center = Point3f::new(0.0, 20.0, 20.0);
    let mut buffer = Buffer::new(W, H);
    let cam = Camera::new(center, [0.0, -1.0, -1.0].into(), up, 4.0).unwrap();

    // let texture = Constant::new(TGAColor::rgba(255, 255, 255, 255));
    let texture = TGAImage::read_file("obj/StarSparrow/Textures/StarSparrow_Red.tga").unwrap();
    let mut shader = rasterizer::Gouraud::new(&model, &cam, light_dir, texture);
    draw(&mut shader, &model, &mut buffer);
    buffer
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

fn floor() -> Buffer {
    let model = Model::read_file("obj/floor.obj").unwrap();
    let up = Vector3f::new(0.0, 1.0, 0.0);
    let light_dir = Vector3f::new(1.0, 1.0, 1.0);
    let center = Point3f::new(0.0, 0.0, 0.0);
    let mut buffer = Buffer::new(W, H);
    let cam = Camera::new(center, [0.0, -1.0, -1.0].into(), up, 4.0).unwrap();

    let texture = TGAImage::read_file("obj/floor_diffuse.tga").unwrap();
    let mut shader = rasterizer::Gouraud::new(&model, &cam, light_dir, texture);
    draw(&mut shader, &model, &mut buffer);
    buffer
}

