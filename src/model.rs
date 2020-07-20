use crate::geometry::{Point3f, Point2f, Vector3f, Normal};
use obj::raw::object::{self, RawObj, Polygon::*};
use obj::raw::parse_obj;
use std::fs::File;
use std::io::BufReader;

pub struct Model {
    polygons: Vec<Polygon>,
    positions: Vec<Point3f>,
    textures: Vec<Point2f>,
    normals: Vec<Normal>,
}

impl Model {
    pub fn read_file(path: &str) -> Option<Model> {
        let input = BufReader::new(File::open(path).ok()?);
        let raw = parse_obj(input).unwrap();
        Some(Model::from_raw(raw))
    }

    pub fn from_raw(raw: RawObj) -> Self {
        let polygons = raw.polygons.iter().map(make_polygon).collect();
        let positions = raw.positions.iter().map(|&(x, y, z, _)| (x, y, z).into()).collect();
        let textures = raw.tex_coords.iter().map(|&(u, v, _)| (u, v).into()).collect();
        let normals = raw.normals.iter().map(|&n| n.into()).collect();
        Model { polygons, positions, textures, normals }
    }

    pub fn polygons(&self) -> &[Polygon] {
        &self.polygons[..]
    }

    pub fn vertex(&self, f: usize, v: usize) -> VertexVal<'_> {
        let i = &self.polygons[f].vertice[v];
        self.index(i)
    }

    pub fn index(&self, i: &Vertex) -> VertexVal<'_> {
        let position = &self.positions[i.position];
        let texture = i.texture.map(|i| &self.textures[i]);
        let normal = i.normal.map(|i| &self.normals[i]);
        VertexVal { position, texture, normal }
    }

    fn inspect(&self) {
        let mut counts = [0; 6];
        for p in self.polygons() {
            let c = p.vertice().len();
            if c >= 5 {
                counts[5] += 1;
            } else {
                counts[c] += 1;
            }
        }
        let total = self.polygons().len();
        println!("total {} polygons, {:?}", total, counts);
    }
}

pub struct VertexVal<'a> {
    pub position: &'a Point3f,
    pub texture: Option<&'a Point2f>,
    pub normal: Option<&'a Normal>,
}

pub struct Vertex {
    position: usize,
    normal: Option<usize>,
    texture: Option<usize>,
}

impl Vertex {
    fn new(position: usize, texture: Option<usize>, normal: Option<usize>) -> Self {
        Vertex { position, normal, texture }
    }
}

pub struct Polygon {
    vertice: Vec<Vertex>
}

impl Polygon {
    pub fn vertice(&self) -> &[Vertex] {
        &self.vertice[..]
    }

    pub fn assume_triangle<'a>(&self, model: &'a Model) -> [VertexVal<'a>; 3] {
        [model.index(&self.vertice[0]),
        model.index(&self.vertice[1]),
        model.index(&self.vertice[2])]
    }

    pub fn triangles(&self) -> Triangles<'_> {
        Triangles {
            vertice: self.vertice(),
            v_1: 1,
        }
    }
}

pub struct Triangles<'a> {
    vertice: &'a [Vertex],
    v_1: usize,
}

impl<'a> Iterator for Triangles<'a> {
    type Item=[usize; 3];

    fn next(&mut self) -> Option<Self::Item> {
        let v_2 = self.v_1 + 1;
        if v_2 >= self.vertice.len() {
            None
        } else {
            let ret = [0, self.v_1, v_2];
            self.v_1 = v_2;
            Some(ret)
        }
    }
}

fn make_polygon(poly: &object::Polygon) -> Polygon {
    let vertice = match poly {
        P(v)   => v.iter().map(|&p| Vertex::new(p, None, None)).collect(),
        PT(v)  => v.iter().map(|&(p, t)| Vertex::new(p, Some(t), None)).collect(),
        PN(v)  => v.iter().map(|&(p, n)| Vertex::new(p, None, Some(n))).collect(),
        PTN(v) => v.iter().map(|&(p, t, n)| Vertex::new(p, Some(t), Some(n))).collect(),
    };
    Polygon { vertice }
}

#[cfg(test)]
mod test {
    #[test]
    fn model_info() {
        super::Model::read_file("obj/african_head/african_head.obj")
            .unwrap()
            .inspect()
    }
}

