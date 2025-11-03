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

    // Peek at first 12 bytes to detect format (some formats need more bytes)
    let mut header = [0; 12];
    reader.read_exact(&mut header)?;

    let format = if header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        // PNG magic bytes
        ImageFormat::Png
    } else if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
        // JPEG magic bytes
        ImageFormat::Jpeg
    } else if header.starts_with(&[0x47, 0x49, 0x46, 0x38, 0x37, 0x61]) || 
              header.starts_with(&[0x47, 0x49, 0x46, 0x38, 0x39, 0x61]) {
        // GIF magic bytes (GIF87a or GIF89a)
        ImageFormat::Gif
    } else if header.starts_with(&[0x52, 0x49, 0x46, 0x46]) && 
              header[8..12] == [0x57, 0x45, 0x42, 0x50] {
        // WebP magic bytes (RIFF....WEBP)
        ImageFormat::WebP
    } else if header.starts_with(&[0x42, 0x4D]) {
        // BMP magic bytes
        ImageFormat::Bmp
    } else if header.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || 
              header.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        // TIFF magic bytes (little-endian or big-endian)
        ImageFormat::Tiff
    } else if header.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        // ICO magic bytes
        ImageFormat::Ico
    } else if header[0..4] == [0x71, 0x6F, 0x69, 0x66] {
        // QOI magic bytes "qoif"
        ImageFormat::Qoi
    } else {
        return Err(format!(
            "Unsupported image format. Supported formats: PNG, JPEG, GIF, WebP, BMP, TIFF, ICO, QOI. Header bytes: {:02X?}", 
            &header[..8]
        ).into());
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

