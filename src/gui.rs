use orbtk::prelude::*;

pub fn view(w: u32, h: u32, data: Vec<u32>) {
    Application::new()
        .window(move |ctx| {
            let mut img = Image::new(w, h);
            img.draw(&data[..]);
            
            Window::create()
                .title("Preview")
                .position((100.0, 100.0))
                .size(w as f64, h as f64)
                .child(ImageWidget::create().image(img).build(ctx))
                .build(ctx)
        })
        .run();
}

