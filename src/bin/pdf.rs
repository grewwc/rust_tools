use clap::{ArgAction, Parser};

use rust_tools::pdfw::{
    PdfParseOptions, ocr_pdf_to_markdown, ocr_pdf_to_markdown_pages, parse_pdf,
};

#[derive(Parser)]
#[command(name = "pdf", about = "Parse a PDF file")]
struct Cli {
    #[arg(help = "PDF file path")]
    path: String,

    #[arg(long, help = "Only parse a specific page (1-based)")]
    page: Option<u32>,

    #[arg(long, help = "Do not extract text", action = ArgAction::SetTrue)]
    no_text: bool,

    #[arg(long, help = "Output JSON", action = ArgAction::SetTrue)]
    json: bool,

    #[arg(long, help = "Output Markdown", action = ArgAction::SetTrue)]
    md: bool,

    #[arg(long, help = "Disable OCR when generating Markdown", action = ArgAction::SetTrue)]
    no_ocr: bool,

    #[arg(
        long,
        help = "OCR languages (Vision), comma-separated. Example: zh-Hans,en-US",
        default_value = "zh-Hans,en-US"
    )]
    langs: String,

    #[arg(long, help = "Print content stats", action = ArgAction::SetTrue)]
    stats: bool,

    #[arg(
        long,
        help = "Limit printed text characters (0 = no limit)",
        default_value_t = 0
    )]
    max_chars: usize,
}

fn main() {
    let cli = Cli::parse();

    if cli.stats {
        if let Err(err) = print_stats(&cli.path, cli.page) {
            eprintln!("{err}");
            std::process::exit(1);
        }
        return;
    }

    let opts = PdfParseOptions {
        extract_text: !cli.no_text,
        pages: cli.page.map(|p| vec![p]),
    };

    let mut parsed = match parse_pdf(&cli.path, opts) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    if cli.max_chars > 0
        && let Some(text) = parsed.text.as_mut()
        && text.len() > cli.max_chars
    {
        text.truncate(cli.max_chars);
    }

    if cli.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&parsed).unwrap_or_default()
        );
        return;
    }

    if cli.md {
        let langs = cli
            .langs
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        match render_markdown(&cli.path, &parsed, cli.no_ocr, &langs, cli.page) {
            Ok(md) => {
                print!("{md}");
                return;
            }
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(1);
            }
        }
    }

    println!("path: {}", parsed.path.display());
    println!("pages: {}", parsed.page_count);
    if let Some(v) = parsed.title.as_deref() {
        println!("title: {v}");
    }
    if let Some(v) = parsed.author.as_deref() {
        println!("author: {v}");
    }
    if let Some(v) = parsed.subject.as_deref() {
        println!("subject: {v}");
    }
    if let Some(v) = parsed.keywords.as_deref() {
        println!("keywords: {v}");
    }
    if let Some(v) = parsed.text.as_deref()
        && !v.trim().is_empty()
    {
        println!();
        print!("{v}");
    }
}

fn render_markdown(
    path: &str,
    parsed: &rust_tools::pdfw::ParsedPdf,
    no_ocr: bool,
    langs: &[&str],
    page: Option<u32>,
) -> Result<String, String> {
    let title = parsed
        .title
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .or_else(|| {
            std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "PDF".to_string());

    let mut pieces: Vec<String> = Vec::new();
    pieces.push(format!("# {title}\n"));

    if let Some(v) = parsed.author.as_deref().filter(|s| !s.trim().is_empty()) {
        pieces.push(format!("- author: {}\n", v.trim()));
    }
    if let Some(v) = parsed.subject.as_deref().filter(|s| !s.trim().is_empty()) {
        pieces.push(format!("- subject: {}\n", v.trim()));
    }
    if let Some(v) = parsed.keywords.as_deref().filter(|s| !s.trim().is_empty()) {
        pieces.push(format!("- keywords: {}\n", v.trim()));
    }

    let body = if no_ocr {
        extract_text_markdown_by_page(&parsed.path, page)?
    } else {
        match page {
            Some(p) => ocr_pdf_to_markdown_pages(&parsed.path, langs, Some(&[p]))
                .map_err(|e| e.to_string())?,
            None => ocr_pdf_to_markdown(&parsed.path, langs).map_err(|e| e.to_string())?,
        }
    };
    if !body.trim().is_empty() {
        pieces.push("\n".to_string());
        pieces.push(body);
        return Ok(pieces.concat());
    }

    pieces.push("\n".to_string());
    pieces.push("（未发现可提取的文字）\n".to_string());
    Ok(pieces.concat())
}

fn extract_text_markdown_by_page(
    path: &std::path::Path,
    page: Option<u32>,
) -> Result<String, String> {
    let doc = lopdf::Document::load(path).map_err(|e| e.to_string())?;
    let pages = doc.get_pages();
    let page_count = pages.len();
    if page_count == 0 {
        return Ok(String::new());
    }

    let mut out = String::new();
    let selected_pages: Vec<u32> = match page {
        Some(p) => vec![p],
        None => (1..=page_count as u32).collect(),
    };
    for page_number in selected_pages {
        if !pages.contains_key(&page_number) {
            continue;
        }
        let text = doc
            .extract_text(&[page_number])
            .unwrap_or_default()
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!("## Page {page_number}\n\n"));
        out.push_str(&text);
        out.push_str("\n\n");
    }
    Ok(out)
}

fn print_stats(path: &str, page: Option<u32>) -> Result<(), String> {
    let path = std::path::Path::new(path);
    let doc = lopdf::Document::load(path).map_err(|e| e.to_string())?;
    let pages = doc.get_pages();
    println!("pages: {}", pages.len());

    let selected_pages: Vec<u32> = match page {
        Some(p) => vec![p],
        None => (1..=pages.len() as u32).collect(),
    };
    for page_number in selected_pages {
        let page_id = pages
            .get(&page_number)
            .ok_or_else(|| format!("page {page_number} not found"))?;
        let content = doc
            .get_and_decode_page_content(*page_id)
            .map_err(|e| e.to_string())?;
        let mut text_ops = 0usize;
        let mut image_ops = 0usize;
        for op in &content.operations {
            match op.operator.as_ref() {
                "Tj" | "TJ" | "'" | "\"" => text_ops += 1,
                "Do" => image_ops += 1,
                _ => {}
            }
        }
        let image_infos = page_image_infos(&doc, *page_id);
        if image_infos.is_empty() {
            println!("page {page_number}: text_ops={text_ops} image_ops={image_ops}");
        } else {
            let parts = image_infos
                .into_iter()
                .map(|i| format!("{} {} {}x{}", i.filter, i.color_space, i.width, i.height))
                .collect::<Vec<_>>()
                .join(" | ");
            println!(
                "page {page_number}: text_ops={text_ops} image_ops={image_ops} images={parts}"
            );
        }
    }
    Ok(())
}

struct PageImageInfo {
    filter: String,
    color_space: String,
    width: u32,
    height: u32,
}

fn page_image_infos(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> Vec<PageImageInfo> {
    let mut out = Vec::new();
    let (resource_dict, resource_ids) = match doc.get_page_resources(page_id) {
        Ok(v) => v,
        Err(_) => return out,
    };
    if let Some(resources) = resource_dict {
        collect_image_infos_from_resources(doc, resources, 0, &mut out);
    }
    for id in resource_ids {
        if let Ok(dict) = doc.get_dictionary(id) {
            collect_image_infos_from_resources(doc, dict, 0, &mut out);
        }
    }
    out
}

fn collect_image_infos_from_resources(
    doc: &lopdf::Document,
    resources: &lopdf::Dictionary,
    depth: usize,
    out: &mut Vec<PageImageInfo>,
) {
    if depth >= 8 {
        return;
    }
    let xobj = match resources.get(b"XObject") {
        Ok(v) => v,
        Err(_) => return,
    };
    let dict = match xobj {
        lopdf::Object::Dictionary(d) => Some(d),
        lopdf::Object::Reference(r) => doc.get_dictionary(*r).ok(),
        _ => None,
    };
    let Some(dict) = dict else {
        return;
    };
    for (_, obj) in dict.iter() {
        collect_image_infos_from_xobject(doc, obj, depth + 1, out);
    }
}

fn collect_image_infos_from_xobject(
    doc: &lopdf::Document,
    obj: &lopdf::Object,
    depth: usize,
    out: &mut Vec<PageImageInfo>,
) {
    if depth >= 8 {
        return;
    }
    let stream = match obj {
        lopdf::Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| match o {
            lopdf::Object::Stream(s) => Some(s),
            _ => None,
        }),
        lopdf::Object::Stream(s) => Some(s),
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
            let width = stream
                .dict
                .get(b"Width")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0);
            let height = stream
                .dict
                .get(b"Height")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0);
            let filter = stream
                .dict
                .get(b"Filter")
                .ok()
                .and_then(|o| o.as_name().ok())
                .map(|n| String::from_utf8_lossy(n).to_string())
                .or_else(|| {
                    stream
                        .dict
                        .get(b"Filter")
                        .ok()
                        .and_then(|o| o.as_array().ok())
                        .and_then(|a| a.first())
                        .and_then(|o| o.as_name().ok())
                        .map(|n| String::from_utf8_lossy(n).to_string())
                })
                .unwrap_or_else(|| "None".to_string());
            let color_space = stream
                .dict
                .get(b"ColorSpace")
                .ok()
                .and_then(|o| o.as_name().ok())
                .map(|n| String::from_utf8_lossy(n).to_string())
                .or_else(|| {
                    stream
                        .dict
                        .get(b"ColorSpace")
                        .ok()
                        .and_then(|o| o.as_array().ok())
                        .and_then(|a| a.first())
                        .and_then(|o| o.as_name().ok())
                        .map(|n| String::from_utf8_lossy(n).to_string())
                })
                .unwrap_or_else(|| "Unknown".to_string());
            out.push(PageImageInfo {
                filter,
                color_space,
                width,
                height,
            });
        }
        Some(b"Form") => {
            let resources = stream.dict.get(b"Resources").ok();
            let dict = match resources {
                Some(lopdf::Object::Dictionary(d)) => Some(d),
                Some(lopdf::Object::Reference(r)) => doc.get_dictionary(*r).ok(),
                _ => None,
            };
            if let Some(dict) = dict {
                collect_image_infos_from_resources(doc, dict, depth + 1, out);
            }
        }
        _ => {}
    }
}
