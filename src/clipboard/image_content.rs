use std::{
    fs::File,
    io::{self, BufReader, Read},
};

use arboard::{Clipboard, ImageData};
use image::{ImageBuffer, ImageFormat, Rgb, Rgba, buffer::ConvertBuffer};

use crate::common::filename::add_suffix;

pub fn save_to_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut clipboard = Clipboard::new().unwrap();
    let fname: String = add_suffix(fname, ".jpg", || !fname.contains('.'));
    if let Ok(image) = clipboard.get_image() {
        let data = image.bytes;
        let image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
            image.width as u32,
            image.height as u32,
            data.to_vec(),
        )
        .ok_or("failed to create image")?;
        let image: ImageBuffer<Rgb<u8>, Vec<u8>> = image.convert();
        image.save(fname.as_str())?;
        println!("save to file: {fname}");
        Ok(())
    } else {
        Err(Box::new(io::Error::new(io::ErrorKind::Other, "no image")))
    }
}

fn open_by_content(path: &str) -> Result<image::DynamicImage, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Peek at first 8 bytes to detect format
    let mut header = [0; 8];
    reader.read_exact(&mut header)?;

    let format = if header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        ImageFormat::Png
    } else if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
        ImageFormat::Jpeg
    } else {
        return Err("Unsupported image format".into());
    };

    // Rewind by creating a new reader with header + rest
    use std::io::Cursor;
    let mut full_data = Vec::new();
    full_data.extend_from_slice(&header);
    reader.read_to_end(&mut full_data)?;

    let cursor = Cursor::new(full_data);
    Ok(image::load(cursor, format)?)
}

pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut clipboard = Clipboard::new()?;
    match open_by_content(fname) {
        Ok(img) => {
            let img_rgba = img.to_rgba8();
            let width = img_rgba.width();
            let height = img_rgba.height();
            let bytes = img_rgba.into_raw();

            let image_data = ImageData {
                width: width as usize,
                height: height as usize,
                bytes: std::borrow::Cow::Owned(bytes),
            };

            clipboard.set_image(image_data)?;

            Ok(())
        }
        Err(err) => {
            eprintln!("{err:?}");
            Err(err)
        }
    }
}
