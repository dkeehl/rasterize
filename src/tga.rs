use crate::geometry::{Point2i, Point3f};
use tinytga::Tga;
use std::io::{self, Read, Write};
use std::fs::File;
use std::ops::Mul;

pub struct TGAImage {
    pub width: u16,
    pub height: u16,
    data: Vec<TGAColor>,
}

impl TGAImage {
    pub fn read_file(path: &str) -> Option<TGAImage> {
        let mut buf = vec![];
        let mut file = File::open(path).ok()?;
        file.read_to_end(&mut buf);
        TGAImage::from_raw(&buf[..])
    }

    pub fn from_raw(raw: &[u8]) -> Option<TGAImage> {
        let tga = Tga::from_slice(raw).ok()?;
        let width = tga.width();
        let height = tga.height();
        let data: Vec<_> = tga.into_iter().map(|p| TGAColor { val: p.color }).collect();
        assert_eq!(width as usize * height as usize, data.len());
        Some(TGAImage { width, height, data })
    }

    pub fn ndc_to_raster(&self, x: f32, y: f32) -> Point2i {
        let x = (x * self.width as f32).floor() as i32;
        let y = (y * self.height as f32).floor() as i32;
        (x, y).into()
    }

    pub fn set(&mut self, p: &Point2i, c: TGAColor) -> bool {
        let width = self.width as i32;
        let height = self.height as i32;
        if p[0] < 0 || p[0] >= width || p[1] < 0 || p[1] >= height {
            false
        } else {
            let i = (p[1] * width + p[0]) as usize;
            self.data[i] = c;
            true
        }
    }

    pub fn get(&self, p: &Point2i) -> TGAColor {
        let width = self.width as i32;
        let height = self.height as i32;
        if p[0] < 0 || p[0] >= width || p[1] < 0 || p[1] >= height {
            BLACK
        } else {
            let i = (p[1] * width + p[0]) as usize;
            self.data[i]
        }
    }

    pub fn flip_horizontally(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;
        let mid = width / 2;

        for i in 0..height {
            let row = &mut self.data[i * width..(i + 1) * width];
            for j in 0..mid {
                row.swap(j, width - 1 - j);
            }
        }
    }

    pub fn flip_vertically(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;
        let mid = height / 2;
        let mut data = &mut self.data[..];

        for i in 0..mid {
            let (l, r) = data.split_at_mut(width);
            let (m, r) = r.split_at_mut(r.len() - width);
            l.swap_with_slice(r);
            data = m;
        }
    }

    pub fn write_file(&self, path: &str) -> io::Result<()> {
        let mut f = File::create(path)?;
        let header = TGAHeader::true_color(self.width, self.height);
        f.write(header.serialize().as_slice())?;
        let len = self.data.len();
        let ptr = self.data.as_ptr() as *const u8;
        let slice = unsafe {
            std::slice::from_raw_parts(ptr, len * std::mem::size_of::<TGAColor>())
        };
        f.write_all(slice)
    }
}

pub struct Buffer {
    img: TGAImage,
    z_buffer: Vec<f32>,
}

impl Buffer {
    pub fn new(width: u16, height: u16) -> Buffer {
        let size = width as usize * height as usize;
        let img = TGAImage {
            width,
            height,
            data: vec![BLACK; size],
        };
        let z_buffer = vec![std::f32::NEG_INFINITY; size];
        Buffer { img, z_buffer }
    }

    pub fn position(&mut self, p: &Point2i) -> Option<Position<'_>> {
        let width = self.img.width as i32;
        let height = self.img.height as i32;
        if p[0] < 0 || p[0] >= width || p[1] < 0 || p[1] >= height {
            None
        } else {
            let pos = (p[1] * width + p[0]) as usize;
            Some(Position {
                buf: self,
                pos
            })
        }
    }

    pub fn set(&mut self, p: &Point2i, c: TGAColor) -> bool {
        self.img.set(p, c)
    }

    fn pos(&self, x: f32, y: f32) -> Point2i {
        self.img.ndc_to_raster((x + 1.0) * 0.5, (1.0 - y) * 0.5)
    }

    pub fn world_to_raster(&self, p_world: &Point3f) -> Point2i {
        self.pos(p_world[0], p_world[1])
    }

    pub fn screen_to_raster(&self, p_world: &Point3f) -> Point2i {
        self.pos(p_world[0], p_world[1])
    }

    pub fn get(&self, p: &Point2i) -> TGAColor {
        self.img.get(p)
    }

    pub fn write_file(&self, path: &str) -> io::Result<()> {
        self.img.write_file(path)
    }
    
    pub fn z_buffer_get(&self, pos: &Point2i) -> Option<f32> {
        self.z_buffer.get((pos[0] * self.img.width as i32 +  pos[1]) as usize).map(|x| *x)
    }
}

pub struct Position<'a> {
    buf: &'a mut Buffer,
    pos: usize,
}

impl<'a> Position<'a> {
    pub fn z_buffer_set(&mut self, z: f32) {
        self.buf.z_buffer[self.pos] = z
    }

    pub fn z_buffer_get(&self) -> f32 {
        self.buf.z_buffer[self.pos]
    }

    pub fn set_color(&mut self, c: TGAColor) {
        self.buf.img.data[self.pos] = c
    }
}

#[derive(Clone, Copy)]
pub struct TGAColor {
    val: u32,
}

impl TGAColor {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> TGAColor {
        let val = (a as u32) << 24 | (r as u32) << 16 |
            (g as u32) << 8  | b as u32;
        TGAColor { val }
    }

    pub fn unpack(self) -> Rgba {
        let [a, r, g, b] = self.val.to_be_bytes();
        Rgba { r, g, b, a }
    }
}

impl Mul<Rate> for TGAColor {
    type Output = TGAColor;

    fn mul(self, s: Rate) -> TGAColor {
        let [a, r, g, b] = self.val.to_be_bytes();
        let f = |n| (n as f32 * s.get()) as u8;
        TGAColor::rgba(f(r), f(g), f(b), f(a))
    }
}

#[derive(Clone, Copy)]
pub struct Rate(f32);

impl Rate {
    pub fn unchecked(n: f32) -> Rate {
        Rate(n)
    }

    pub fn new(n: f32) -> Option<Rate> {
        if n >= 0.0 && n <= 1.0 {
            Some(Rate(n))
        } else {
            None
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

const BLACK: TGAColor = TGAColor::rgba(0, 0, 0, 255);

#[derive(Clone, Copy)]
enum ImageType {
    Empty = 0,
    ColorMapped = 1,
    Truecolor = 2,
    Monochrome = 3,
    RleColorMapped = 9,
    RleTruecolor = 10,
    RleMonochrome = 11,
}

#[derive(Clone, Copy)]
enum PixelDepth {
    B8 = 8,
    B16 = 16,
    B24 = 24,
    B32 = 32,
}

// https://www.fileformat.info/format/tga/egff.htm
struct TGAHeader {
    id_len: u8,
    has_color_map: bool,
    image_type: ImageType,
    color_map_start: u16,
    color_map_len: u16,
    color_map_depth: u8,
    x_offset: u16,
    y_offset: u16,
    width: u16,
    height: u16,
    pixel_depth: PixelDepth,
    img_descriptor: u8,
}

impl TGAHeader {
    fn true_color(width: u16, height: u16) -> Self {
        TGAHeader {
            id_len: 0,
            has_color_map: false,
            image_type: ImageType::Truecolor,
            color_map_start: 0,
            color_map_len: 0,
            color_map_depth: 0,
            x_offset: 0,
            y_offset: 0,
            width,
            height,
            pixel_depth: PixelDepth::B32,
            img_descriptor: 0x20,
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut ret = Vec::with_capacity(18);
        ret.push(self.id_len);
        ret.push(self.has_color_map as u8);
        ret.push(self.image_type as u8);
        ret.extend_from_slice(&self.color_map_start.to_le_bytes());
        ret.extend_from_slice(&self.color_map_len.to_le_bytes());
        ret.push(self.color_map_depth);
        ret.extend_from_slice(&self.x_offset.to_le_bytes());
        ret.extend_from_slice(&self.y_offset.to_le_bytes());
        ret.extend_from_slice(&self.width.to_le_bytes());
        ret.extend_from_slice(&self.height.to_le_bytes());
        ret.push(self.pixel_depth as u8);
        ret.push(self.img_descriptor);
        ret
    }
}

#[cfg(test)]
mod test {
    use super::TGAColor;

    #[test]
    fn color_unpack() {
        let c = TGAColor::rgba(1, 2, 3, 4).unpack();
        assert_eq!(c.r, 1);
        assert_eq!(c.g, 2);
        assert_eq!(c.b, 3);
        assert_eq!(c.a, 4);
    }
}

