use std::{
    fs::File,
    io::{self, BufReader, Read, Write},
};

use arboard::{Clipboard, ImageData};
use image::{ImageBuffer, ImageFormat, Rgb, Rgba, buffer::ConvertBuffer};

use crate::commonw::filename::add_suffix;

