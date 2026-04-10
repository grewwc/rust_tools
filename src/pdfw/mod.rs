use std::{
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

use regex::Regex;

use image::{DynamicImage, ImageBuffer, Luma, Rgb};

#[derive(Debug, Clone)]
pub struct PdfParseOptions {
    pub extract_text: bool,
    pub pages: Option<Vec<u32>>,
}

impl Default for PdfParseOptions {
    fn default() -> Self {
        Self {
            extract_text: true,
            pages: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ParsedPdf {
    pub path: PathBuf,
    pub page_count: usize,
    pub title: Option<String>,
    pub author: Option<String>,
    pub subject: Option<String>,
    pub keywords: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug)]
pub enum PdfParseError {
    OpenFailed(std::io::Error),
    ParseFailed(lopdf::Error),
    ExtractTextFailed(String),
}

impl fmt::Display for PdfParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenFailed(err) => write!(f, "open pdf failed: {err}"),
            Self::ParseFailed(err) => write!(f, "parse pdf failed: {err}"),
            Self::ExtractTextFailed(err) => write!(f, "extract pdf text failed: {err}"),
        }
    }
}

impl Error for PdfParseError {}

pub fn parse_pdf(
    path: impl AsRef<Path>,
    opts: PdfParseOptions,
) -> Result<ParsedPdf, PdfParseError> {
    let path = path.as_ref();
    let canonical = path.canonicalize().map_err(PdfParseError::OpenFailed)?;

    let doc = lopdf::Document::load(&canonical).map_err(PdfParseError::ParseFailed)?;
    let page_count = doc.get_pages().len();

    let info = extract_info_dict(&doc);
    let mut title = info
        .as_ref()
        .and_then(|d| d.get(b"Title").ok())
        .and_then(|o| resolve_to_string(&doc, o).ok())
        .filter(|s| !s.trim().is_empty());
    let mut author = info
        .as_ref()
        .and_then(|d| d.get(b"Author").ok())
        .and_then(|o| resolve_to_string(&doc, o).ok())
        .filter(|s| !s.trim().is_empty());
    let mut subject = info
        .as_ref()
        .and_then(|d| d.get(b"Subject").ok())
        .and_then(|o| resolve_to_string(&doc, o).ok())
        .filter(|s| !s.trim().is_empty());
    let mut keywords = info
        .as_ref()
        .and_then(|d| d.get(b"Keywords").ok())
        .and_then(|o| resolve_to_string(&doc, o).ok())
        .filter(|s| !s.trim().is_empty());

    let xmp = extract_xmp_metadata(&doc);
    if title.is_none() {
        title = xmp.title;
    }
    if author.is_none() {
        author = xmp.author;
    }
    if subject.is_none() {
        subject = xmp.subject;
    }
    if keywords.is_none() {
        keywords = xmp.keywords;
    }

    let text = if opts.extract_text {
        let page_numbers = match opts.pages.as_deref() {
            Some(pages) => {
                let mut out = pages
                    .iter()
                    .copied()
                    .filter(|p| *p >= 1 && (*p as usize) <= page_count)
                    .collect::<Vec<_>>();
                out.sort_unstable();
                out.dedup();
                out
            }
            None => (1..=page_count as u32).collect(),
        };
        if page_numbers.is_empty() {
            return Err(PdfParseError::ExtractTextFailed(
                "no valid pages selected".to_string(),
            ));
        }

        let extracted = match doc.extract_text(&page_numbers) {
            Ok(t) => t,
            Err(err) => {
                if opts.pages.is_some() {
                    return Err(PdfParseError::ExtractTextFailed(err.to_string()));
                }
                pdf_extract::extract_text(&canonical).map_err(|e| {
                    PdfParseError::ExtractTextFailed(format!("{err}; fallback failed: {e}"))
                })?
            }
        };
        Some(extracted)
    } else {
        None
    };

    Ok(ParsedPdf {
        path: canonical,
        page_count,
        title,
        author,
        subject,
        keywords,
        text,
    })
}

fn extract_info_dict(doc: &lopdf::Document) -> Option<lopdf::Dictionary> {
    let info_ref = doc.trailer.get(b"Info").ok()?.as_reference().ok()?;
    let info_obj = doc.get_object(info_ref).ok()?;
    match info_obj {
        lopdf::Object::Dictionary(d) => Some(d.clone()),
        lopdf::Object::Reference(r) => doc.get_dictionary(*r).ok().cloned(),
        _ => None,
    }
}

fn resolve_to_string(doc: &lopdf::Document, obj: &lopdf::Object) -> Result<String, lopdf::Error> {
    let resolved = match obj {
        lopdf::Object::Reference(r) => doc.get_object(*r)?.clone(),
        other => other.clone(),
    };
    match &resolved {
        lopdf::Object::String(_, _) => Ok(lopdf::decode_text_string(&resolved).unwrap_or_default()),
        _ => Ok(object_to_string(&resolved).unwrap_or_default()),
    }
}

fn object_to_string(obj: &lopdf::Object) -> Option<String> {
    match obj {
        lopdf::Object::String(bytes, _) => Some(String::from_utf8_lossy(bytes).to_string()),
        lopdf::Object::Name(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
        lopdf::Object::Integer(v) => Some(v.to_string()),
        lopdf::Object::Real(v) => Some(v.to_string()),
        _ => None,
    }
}

#[derive(Default)]
struct XmpMetadata {
    title: Option<String>,
    author: Option<String>,
    subject: Option<String>,
    keywords: Option<String>,
}

fn extract_xmp_metadata(doc: &lopdf::Document) -> XmpMetadata {
    let Ok(catalog) = doc.catalog() else {
        return XmpMetadata::default();
    };

    let Ok(meta_obj) = catalog.get(b"Metadata") else {
        return XmpMetadata::default();
    };

    let meta_obj = match meta_obj {
        lopdf::Object::Reference(r) => match doc.get_object(*r) {
            Ok(o) => o.clone(),
            Err(_) => return XmpMetadata::default(),
        },
        other => other.clone(),
    };

    let lopdf::Object::Stream(stream) = meta_obj else {
        return XmpMetadata::default();
    };

    let bytes = stream.get_plain_content().unwrap_or_default();
    let xmp = String::from_utf8_lossy(&bytes);
    let xmp = xmp.as_ref();

    XmpMetadata {
        title: xmp_capture_title(xmp),
        author: xmp_capture_author(xmp),
        subject: xmp_capture_subject(xmp),
        keywords: xmp_capture_keywords(xmp),
    }
}

fn xmp_capture_title(xmp: &str) -> Option<String> {
    let patterns = [
        r"(?s)<dc:title[^>]*>.*?<rdf:li[^>]*>(.*?)</rdf:li>",
        r"(?s)<pdf:Title[^>]*>(.*?)</pdf:Title>",
    ];
    xmp_capture_first(xmp, &patterns)
}

fn xmp_capture_author(xmp: &str) -> Option<String> {
    let patterns = [
        r"(?s)<dc:creator[^>]*>.*?<rdf:li[^>]*>(.*?)</rdf:li>",
        r"(?s)<pdf:Author[^>]*>(.*?)</pdf:Author>",
    ];
    xmp_capture_first(xmp, &patterns)
}

fn xmp_capture_subject(xmp: &str) -> Option<String> {
    let patterns = [
        r"(?s)<dc:description[^>]*>.*?<rdf:li[^>]*>(.*?)</rdf:li>",
        r"(?s)<pdf:Subject[^>]*>(.*?)</pdf:Subject>",
    ];
    xmp_capture_first(xmp, &patterns)
}

fn xmp_capture_keywords(xmp: &str) -> Option<String> {
    let patterns = [
        r"(?s)<pdf:Keywords[^>]*>(.*?)</pdf:Keywords>",
        r"(?s)<dc:subject[^>]*>.*?<rdf:li[^>]*>(.*?)</rdf:li>",
    ];
    xmp_capture_first(xmp, &patterns)
}

fn xmp_capture_first(xmp: &str, patterns: &[&str]) -> Option<String> {
    for pat in patterns {
        let Ok(re) = Regex::new(pat) else {
            continue;
        };
        let Some(caps) = re.captures(xmp) else {
            continue;
        };
        let Some(m) = caps.get(1) else {
            continue;
        };
        let v = xml_unescape(m.as_str()).trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    None
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

pub fn extract_page_images(
    path: impl AsRef<Path>,
) -> Result<Vec<(u32, Vec<DynamicImage>)>, PdfParseError> {
    let doc = lopdf::Document::load(path.as_ref()).map_err(PdfParseError::ParseFailed)?;
    let pages = doc.get_pages();
    let mut out = Vec::new();
    for page_number in 1..=pages.len() as u32 {
        let Some(page_id) = pages.get(&page_number) else {
            continue;
        };
        let images = extract_images_from_page(&doc, *page_id);
        out.push((page_number, images));
    }
    Ok(out)
}

pub fn ocr_pdf_to_markdown(
    path: impl AsRef<Path>,
    langs: &[&str],
) -> Result<String, PdfParseError> {
    ocr_pdf_to_markdown_pages(path, langs, None)
}

pub fn ocr_pdf_to_markdown_pages(
    path: impl AsRef<Path>,
    langs: &[&str],
    pages: Option<&[u32]>,
) -> Result<String, PdfParseError> {
    let doc = lopdf::Document::load(path.as_ref()).map_err(PdfParseError::ParseFailed)?;
    let page_map = doc.get_pages();
    let mut out = String::new();
    let selected_pages: Vec<u32> = match pages {
        Some(ps) => {
            let mut out = ps
                .iter()
                .copied()
                .filter(|p| page_map.contains_key(p))
                .collect::<Vec<_>>();
            out.sort_unstable();
            out.dedup();
            out
        }
        None => (1..=page_map.len() as u32).collect(),
    };

    for page_number in selected_pages {
        let Some(page_id) = page_map.get(&page_number) else {
            continue;
        };

        let text = doc
            .extract_text(&[page_number])
            .unwrap_or_default()
            .trim()
            .to_string();
        if !text.is_empty() {
            out.push_str(&format!("## Page {page_number}\n\n"));
            out.push_str(&text);
            out.push_str("\n\n");
            continue;
        }

        let images = extract_images_from_page(&doc, *page_id);
        if images.is_empty() {
            continue;
        }

        let mut page_text = String::new();
        for img in images {
            let t = ocr_image_to_text(&img, langs).unwrap_or_default();
            if !t.trim().is_empty() {
                if !page_text.is_empty() {
                    page_text.push('\n');
                }
                page_text.push_str(t.trim());
            }
        }

        if !page_text.trim().is_empty() {
            out.push_str(&format!("## Page {page_number}\n\n"));
            out.push_str(page_text.trim());
            out.push_str("\n\n");
        }
    }
    Ok(out)
}

fn extract_images_from_page(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> Vec<DynamicImage> {
    let mut images = Vec::new();
    let Ok((resource_dict, resource_ids)) = doc.get_page_resources(page_id) else {
        return images;
    };
    let mut resource_dicts: Vec<&lopdf::Dictionary> = Vec::new();
    if let Some(resources) = resource_dict {
        resource_dicts.push(resources);
    }
    for id in resource_ids {
        if let Ok(dict) = doc.get_dictionary(id) {
            resource_dicts.push(dict);
        }
    }

    let content = doc.get_and_decode_page_content(page_id).ok();
    let mut names = Vec::new();
    if let Some(content) = content {
        for op in &content.operations {
            if op.operator != "Do" {
                continue;
            }
            let Some(first) = op.operands.first() else {
                continue;
            };
            if let Ok(name) = first.as_name() {
                names.push(name.to_vec());
            }
        }
    }

    if !names.is_empty() {
        for name in names {
            if let Some(obj) = resolve_xobject(doc, &resource_dicts, &name) {
                collect_images_from_xobject(doc, obj, 0, &mut images);
            }
        }
        if !images.is_empty() {
            return images;
        }
    }

    for resources in resource_dicts {
        collect_images_from_resources(doc, resources, 0, &mut images);
    }
    images
}

fn resolve_xobject<'a>(
    doc: &'a lopdf::Document,
    resource_dicts: &[&'a lopdf::Dictionary],
    name: &[u8],
) -> Option<&'a lopdf::Object> {
    for resources in resource_dicts {
        let xobj = match resources.get(b"XObject") {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dict = match xobj {
            lopdf::Object::Dictionary(d) => Some(d),
            lopdf::Object::Reference(r) => doc.get_dictionary(*r).ok(),
            _ => None,
        };
        let Some(dict) = dict else {
            continue;
        };
        if let Ok(obj) = dict.get(name) {
            return Some(obj);
        }
    }
    None
}

fn collect_images_from_resources(
    doc: &lopdf::Document,
    resources: &lopdf::Dictionary,
    depth: usize,
    images: &mut Vec<DynamicImage>,
) {
    if depth >= 8 {
        return;
    }
    let xobject_obj = match resources.get(b"XObject") {
        Ok(v) => v,
        Err(_) => return,
    };
    let xobject = match xobject_obj {
        lopdf::Object::Dictionary(d) => Some(d),
        lopdf::Object::Reference(r) => doc.get_dictionary(*r).ok(),
        _ => None,
    };
    let Some(xobject) = xobject else {
        return;
    };
    for (_, obj) in xobject.iter() {
        collect_images_from_xobject(doc, obj, depth + 1, images);
    }
}

fn collect_images_from_xobject(
    doc: &lopdf::Document,
    obj: &lopdf::Object,
    depth: usize,
    images: &mut Vec<DynamicImage>,
) {
    if depth >= 8 {
        return;
    }
    let stream = match obj {
        lopdf::Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| match o {
            lopdf::Object::Stream(s) => Some(s.clone()),
            _ => None,
        }),
        lopdf::Object::Stream(s) => Some(s.clone()),
        _ => None,
    };
    let Some(stream) = stream else {
        return;
    };

    let subtype = stream
        .dict
        .get(b"Subtype")
        .ok()
        .and_then(|o| o.as_name().ok());

    match subtype {
        Some(b"Image") => {
            if let Some(img) = decode_xobject_image(&stream) {
                images.push(img);
            }
        }
        Some(b"Form") => {
            let resources = stream.dict.get(b"Resources").ok();
            let dict = match resources {
                Some(lopdf::Object::Dictionary(d)) => Some(d),
                Some(lopdf::Object::Reference(r)) => doc.get_dictionary(*r).ok(),
                _ => None,
            };
            if let Some(dict) = dict {
                collect_images_from_resources(doc, dict, depth + 1, images);
            }
        }
        _ => {}
    }
}

fn decode_xobject_image(stream: &lopdf::Stream) -> Option<DynamicImage> {
    let filter = stream.dict.get(b"Filter").ok();
    let filter_name = filter.as_ref().and_then(|o| o.as_name().ok()).or_else(|| {
        filter
            .as_ref()
            .and_then(|o| o.as_array().ok())
            .and_then(|arr| arr.first())
            .and_then(|o| o.as_name().ok())
    });

    if let Some(name) = filter_name {
        if name == b"DCTDecode" {
            return image::load_from_memory(&stream.content).ok();
        }
        if name == b"JPXDecode" {
            return image::load_from_memory(&stream.content).ok();
        }
        if name == b"FlateDecode" {
            return decode_flate_image(stream);
        }
    }

    let decoded = stream.get_plain_content().ok()?;
    image::load_from_memory(&decoded).ok()
}

fn decode_flate_image(stream: &lopdf::Stream) -> Option<DynamicImage> {
    let data = stream.get_plain_content().ok()?;
    let width = stream
        .dict
        .get(b"Width")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .and_then(|v| u32::try_from(v).ok())?;
    let height = stream
        .dict
        .get(b"Height")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .and_then(|v| u32::try_from(v).ok())?;
    let bpc = stream
        .dict
        .get(b"BitsPerComponent")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(8);
    if bpc != 8 {
        return None;
    }

    let cs = stream.dict.get(b"ColorSpace").ok();
    let cs_name = cs.as_ref().and_then(|o| o.as_name().ok()).or_else(|| {
        cs.as_ref()
            .and_then(|o| o.as_array().ok())
            .and_then(|arr| arr.first())
            .and_then(|o| o.as_name().ok())
    });

    match cs_name {
        Some(b"DeviceGray") => {
            let expected = (width as usize) * (height as usize);
            if data.len() < expected {
                return None;
            }
            let buf: ImageBuffer<Luma<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width, height, data[..expected].to_vec())?;
            Some(DynamicImage::ImageLuma8(buf))
        }
        Some(b"DeviceRGB") | None => {
            let expected = (width as usize) * (height as usize) * 3;
            if data.len() < expected {
                return None;
            }
            let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width, height, data[..expected].to_vec())?;
            Some(DynamicImage::ImageRgb8(buf))
        }
        _ => None,
    }
}

pub fn ocr_image_to_text(img: &DynamicImage, langs: &[&str]) -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        macos_vision_ocr(img, langs)
    }
    #[cfg(not(target_os = "macos"))]
    {
        return tesseract_ocr(img, langs);
    }
}

#[cfg(not(target_os = "macos"))]
fn tesseract_ocr(img: &DynamicImage, langs: &[&str]) -> Result<String, String> {
    use std::{io::Write, process::Command};

    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;

    let input_path =
        std::env::temp_dir().join(format!("rust_tools_ocr_{}.png", uuid::Uuid::new_v4()));
    {
        let mut f = std::fs::File::create(&input_path).map_err(|e| e.to_string())?;
        f.write_all(&png).map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
    }

    let mut cmd = Command::new("tesseract");
    cmd.arg(&input_path).arg("stdout");
    if let Some(lang) = build_tesseract_lang_arg(langs) {
        cmd.arg("-l").arg(lang);
    }

    let out = cmd.output().map_err(|e| {
        let _ = std::fs::remove_file(&input_path);
        if e.kind() == std::io::ErrorKind::NotFound {
            "tesseract not found in PATH (install tesseract-ocr)".to_string()
        } else {
            e.to_string()
        }
    })?;
    let _ = std::fs::remove_file(&input_path);

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        if stderr.is_empty() {
            return Err(format!("tesseract failed: {}", out.status));
        }
        return Err(format!("tesseract failed: {}", stderr));
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = Vec::new();
    for line in text.lines() {
        let normalized = normalize_ocr_line(line);
        if !normalized.is_empty() {
            lines.push(normalized);
        }
    }
    Ok(lines.join("\n"))
}

#[cfg(any(not(target_os = "macos"), test))]
fn build_tesseract_lang_arg(langs: &[&str]) -> Option<String> {
    if langs.is_empty() {
        return None;
    }
    let mut items = Vec::new();
    for l in langs {
        let l = l.trim();
        if l.is_empty() {
            continue;
        }
        items.push(map_lang_for_tesseract(l));
    }
    if items.is_empty() {
        return None;
    }
    Some(items.join("+"))
}

#[cfg(any(not(target_os = "macos"), test))]
fn map_lang_for_tesseract(lang: &str) -> String {
    let key = lang.trim().to_ascii_lowercase();
    match key.as_str() {
        "zh-hans" | "zh_cn" | "zh-cn" | "zh" | "zh-hans-cn" => "chi_sim".to_string(),
        "zh-hant" | "zh_tw" | "zh-tw" | "zh-hant-tw" => "chi_tra".to_string(),
        "en" | "en-us" | "en_us" | "en-gb" | "en_gb" => "eng".to_string(),
        "ja" | "ja-jp" | "ja_jp" => "jpn".to_string(),
        "ko" | "ko-kr" | "ko_kr" => "kor".to_string(),
        _ => lang.trim().to_string(),
    }
}

#[cfg(target_os = "macos")]
fn macos_vision_ocr(img: &DynamicImage, langs: &[&str]) -> Result<String, String> {
    use std::{ffi::CString, ptr::NonNull};

    use objc2::runtime::AnyObject;
    use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSString};
    use objc2_vision::{
        VNImageOption, VNRecognizeTextRequest, VNRecognizeTextRequestRevision3, VNRequest,
        VNRequestTextRecognitionLevel, VNSequenceRequestHandler,
    };

    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;

    let nsdata = unsafe { NSData::dataWithBytes_length(png.as_ptr().cast(), png.len()) };
    let options: objc2::rc::Retained<NSDictionary<VNImageOption, AnyObject>> = NSDictionary::new();
    let handler = unsafe { VNSequenceRequestHandler::new() };

    let request = VNRecognizeTextRequest::new();
    unsafe {
        request.setRevision(VNRecognizeTextRequestRevision3);
    }
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
    request.setUsesLanguageCorrection(true);
    request.setAutomaticallyDetectsLanguage(langs.is_empty());
    request.setMinimumTextHeight(0.0);

    if !langs.is_empty() {
        let mut items = Vec::new();
        for l in langs {
            let c = CString::new(*l).map_err(|e| e.to_string())?;
            let ptr = NonNull::new(c.as_ptr().cast_mut()).ok_or("invalid language")?;
            let s = unsafe { NSString::stringWithUTF8String(ptr) }.ok_or("invalid language")?;
            items.push(s);
        }
        let arr = NSArray::from_retained_slice(&items);
        request.setRecognitionLanguages(&arr);
    }

    let request_super: objc2::rc::Retained<VNRequest> = request.clone().into_super().into_super();
    let requests = NSArray::from_retained_slice(&[request_super]);
    unsafe {
        handler
            .performRequests_onImageData_error(&requests, &nsdata)
            .map_err(|e: objc2::rc::Retained<NSError>| format!("{e:?}"))?;
    }
    drop(options);

    let results = request.results().unwrap_or_default();
    let mut lines = Vec::new();
    for obs in results.iter() {
        let candidates: objc2::rc::Retained<NSArray<AnyObject>> =
            unsafe { objc2::msg_send![&*obs, topCandidates: 1usize] };
        let Some(first) = candidates.firstObject() else {
            continue;
        };
        let s: objc2::rc::Retained<NSString> = unsafe { objc2::msg_send![&*first, string] };
        let rust = nsstring_to_string(&s);
        let rust = rust.trim().to_string();
        let normalized = normalize_ocr_line(&rust);
        if normalized.is_empty() {
            continue;
        }
        lines.push(normalized);
    }

    Ok(lines.join("\n"))
}

#[cfg(target_os = "macos")]
fn nsstring_to_string(s: &objc2_foundation::NSString) -> String {
    let ptr = s.UTF8String();
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().to_string() }
}

fn normalize_ocr_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let lowered = trimmed
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    if lowered == "html" || lowered == "'html" || lowered == "’html" || lowered == "“html" {
        return String::new();
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{Document, Object, Stream};

    #[test]
    fn parse_pdf_extracts_text_for_simple_pdf() {
        let mut doc = Document::with_version("1.5");

        let pages_id = doc.new_object_id();
        let page_id = doc.new_object_id();
        let font_id = doc.new_object_id();
        let contents_id = doc.new_object_id();
        let info_id = doc.new_object_id();

        let mut font_dict = lopdf::Dictionary::new();
        font_dict.set("Type", "Font");
        font_dict.set("Subtype", "Type1");
        font_dict.set("BaseFont", "Helvetica");
        doc.objects.insert(font_id, Object::Dictionary(font_dict));

        let content = b"BT /F1 24 Tf 100 700 Td (Hello PDF) Tj ET".to_vec();
        doc.objects.insert(
            contents_id,
            Object::Stream(Stream::new(lopdf::Dictionary::new(), content)),
        );

        let mut info_dict = lopdf::Dictionary::new();
        info_dict.set("Title", Object::string_literal("Test PDF"));
        doc.objects.insert(info_id, Object::Dictionary(info_dict));

        let mut font_map = lopdf::Dictionary::new();
        font_map.set("F1", font_id);
        let mut resources = lopdf::Dictionary::new();
        resources.set("Font", font_map);
        let mut page_dict = lopdf::Dictionary::new();
        page_dict.set("Type", "Page");
        page_dict.set("Parent", pages_id);
        page_dict.set("MediaBox", vec![0.into(), 0.into(), 595.into(), 842.into()]);
        page_dict.set("Contents", contents_id);
        page_dict.set("Resources", resources);
        doc.objects.insert(page_id, Object::Dictionary(page_dict));

        let mut pages_dict = lopdf::Dictionary::new();
        pages_dict.set("Type", "Pages");
        pages_dict.set("Kids", vec![page_id.into()]);
        pages_dict.set("Count", 1);
        doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

        let catalog_id = doc.new_object_id();
        let mut catalog_dict = lopdf::Dictionary::new();
        catalog_dict.set("Type", "Catalog");
        catalog_dict.set("Pages", pages_id);
        doc.objects
            .insert(catalog_id, Object::Dictionary(catalog_dict));

        doc.trailer.set("Root", catalog_id);
        doc.trailer.set("Info", info_id);

        let tmp_path =
            std::env::temp_dir().join(format!("rust_tools_test_{}.pdf", uuid::Uuid::new_v4()));
        doc.save(&tmp_path).unwrap();

        let parsed = parse_pdf(&tmp_path, PdfParseOptions::default()).unwrap();
        assert_eq!(parsed.page_count, 1);
        assert_eq!(parsed.title.as_deref(), Some("Test PDF"));
        assert!(parsed.text.unwrap_or_default().contains("Hello PDF"));

        let _ = std::fs::remove_file(tmp_path);
    }

    #[test]
    fn tesseract_lang_mapping_is_reasonable() {
        assert_eq!(map_lang_for_tesseract("zh-Hans"), "chi_sim");
        assert_eq!(map_lang_for_tesseract("en-US"), "eng");
        assert_eq!(
            build_tesseract_lang_arg(&["zh-Hans", "en-US"]).as_deref(),
            Some("chi_sim+eng")
        );
    }
}
