use std::io::{self, IsTerminal, Read};

use clap::{ArgAction, Parser as ClapParser};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal::RoundingStrategy;

const DEFAULT_PREC: usize = 16;

const PI_DIGITS: &str = "3.14159265358979323846264338327950288419716939937510";
const E_DIGITS: &str = "2.71828182845904523536028747135266249775724709369995";

#[derive(ClapParser, Debug)]
#[command(about = "Command-line calculator (c)", long_about = None)]
struct Cli {
    #[arg(short = 'e', long = "expr", default_value = "", help = "explicit expression input")]
    expr: String,

    #[arg(short = 'f', long = "file", default_value = "", help = "read expression from file")]
    file: String,

    #[arg(long = "prec", default_value_t = DEFAULT_PREC, help = "decimal digits after division/float functions (default: 16)")]
    prec: usize,

    #[arg(long = "deg", action = ArgAction::SetTrue, help = "use degrees for sin/cos/tan and inverse trig")]
    degree: bool,

    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn normalize_args(args: impl Iterator<Item = String>) -> Vec<String> {
    args.map(|arg| {
        let bytes = arg.as_bytes();
        if bytes.len() > 2 && bytes[0] == b'-' && bytes[1] != b'-' && bytes[1].is_ascii_alphabetic() {
            format!("-{arg}")
        } else {
            arg
        }
    })
    .collect()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_args(std::env::args()));

    let expr = if !cli.expr.trim().is_empty() {
        normalize_expression(&cli.expr)
    } else if !cli.file.trim().is_empty() {
        let contents = std::fs::read_to_string(&cli.file)?;
        normalize_expression(&contents)
    } else if !cli.args.is_empty() {
        normalize_expression(&cli.args.join(" "))
    } else {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            return Err("missing expression\nusage: c [options] <expr>\n  -prec N   decimal precision (default 16)\n  -deg      use degrees for trig".into());
        }
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        normalize_expression(&input)
    };

    if expr.is_empty() {
        return Err("empty expression".into());
    }

    let result = evaluate_expression(&expr, cli.prec, cli.degree)?;
    println!("{result}");
    Ok(())
}

// ─── Normalise ────────────────────────────────────────────────────────────────

fn normalize_expression(input: &str) -> String {
    let s = input.trim();
    let s = s.replace("**", "^");
    let s = s.replace(['\r', '\n', '\t'], " ");
    s.trim().to_string()
}

// ─── Token ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Number,
    Identifier,
    Operator,
    LParen,
    RParen,
    Comma,
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    value: String,
}

impl Token {
    fn eof() -> Self {
        Token { kind: TokenKind::Eof, value: String::new() }
    }
}

// ─── Tokenize ─────────────────────────────────────────────────────────────────

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut tokens = Vec::new();
    let mut i = 0usize;

    while i < n {
        let ch = bytes[i];
        match ch {
            b' ' => { i += 1; }
            b'0'..=b'9' | b'.' => {
                let start = i;
                let mut dot_seen = ch == b'.';
                if ch == b'.' && (i + 1 >= n || !bytes[i + 1].is_ascii_digit()) {
                    return Err(format!("invalid number near {:?}", &input[start..=start]));
                }
                i += 1;
                while i < n {
                    if bytes[i].is_ascii_digit() {
                        i += 1;
                    } else if bytes[i] == b'.' {
                        if dot_seen {
                            return Err(format!("invalid number {:?}", &input[start..=i]));
                        }
                        dot_seen = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                let mut lit = input[start..i].to_string();
                if lit.starts_with('.') { lit.insert(0, '0'); }
                if lit.ends_with('.') { lit.push('0'); }
                tokens.push(Token { kind: TokenKind::Number, value: clean_number(&lit) });
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = i;
                i += 1;
                while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                tokens.push(Token { kind: TokenKind::Identifier, value: input[start..i].to_lowercase() });
            }
            b'+' | b'-' | b'*' | b'/' | b'%' | b'^' => {
                tokens.push(Token { kind: TokenKind::Operator, value: (ch as char).to_string() });
                i += 1;
            }
            b'(' => { tokens.push(Token { kind: TokenKind::LParen, value: "(".into() }); i += 1; }
            b')' => { tokens.push(Token { kind: TokenKind::RParen, value: ")".into() }); i += 1; }
            b',' => { tokens.push(Token { kind: TokenKind::Comma, value: ",".into() }); i += 1; }
            _ => return Err(format!("unexpected character {:?}", ch as char)),
        }
    }

    tokens.push(Token::eof());
    Ok(insert_implicit_mul(tokens))
}

fn ends_primary(tok: &Token) -> bool {
    matches!(tok.kind, TokenKind::Number | TokenKind::Identifier | TokenKind::RParen)
}

fn starts_primary(tok: &Token) -> bool {
    matches!(tok.kind, TokenKind::Number | TokenKind::Identifier | TokenKind::LParen)
}

fn insert_implicit_mul(tokens: Vec<Token>) -> Vec<Token> {
    if tokens.len() <= 1 {
        return tokens;
    }
    let mut res = Vec::with_capacity(tokens.len() * 2);
    for i in 0..tokens.len() - 1 {
        let cur = &tokens[i];
        let nxt = &tokens[i + 1];
        res.push(cur.clone());
        if ends_primary(cur) && starts_primary(nxt) {
            // don't insert between function-name and its '('
            if cur.kind == TokenKind::Identifier
                && nxt.kind == TokenKind::LParen
                && is_function_name(&cur.value)
            {
                continue;
            }
            res.push(Token { kind: TokenKind::Operator, value: "*".into() });
        }
    }
    res.push(tokens[tokens.len() - 1].clone());
    res
}

// ─── Parser ───────────────────────────────────────────────────────────────────

struct ExprParser {
    tokens: Vec<Token>,
    pos: usize,
    prec: usize,
    degree: bool,
}

impl ExprParser {
    fn new(tokens: Vec<Token>, prec: usize, degree: bool) -> Self {
        ExprParser { tokens, pos: 0, prec, degree }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).map_or_else(|| &self.tokens[self.tokens.len() - 1], |t| t)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn parse(&mut self) -> Result<String, String> {
        let result = self.parse_add_sub()?;
        if self.current().kind != TokenKind::Eof {
            return Err(format!("unexpected token {:?}", self.current().value));
        }
        Ok(clean_number(&result))
    }

    fn parse_add_sub(&mut self) -> Result<String, String> {
        let mut left = self.parse_mul_div()?;
        loop {
            let tok = self.current().clone();
            if tok.kind == TokenKind::Operator && (tok.value == "+" || tok.value == "-") {
                self.advance();
                let right = self.parse_mul_div()?;
                left = apply_binary(&left, &right, tok.value.as_bytes()[0], self.prec)?;
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<String, String> {
        let mut left = self.parse_unary()?;
        loop {
            let tok = self.current().clone();
            if tok.kind == TokenKind::Operator
                && (tok.value == "*" || tok.value == "/" || tok.value == "%")
            {
                self.advance();
                let right = self.parse_unary()?;
                left = apply_binary(&left, &right, tok.value.as_bytes()[0], self.prec)?;
            } else if starts_primary(&tok) {
                let right = self.parse_unary()?;
                left = apply_binary(&left, &right, b'*', self.prec)?;
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<String, String> {
        let left = self.parse_primary()?;
        let tok = self.current().clone();
        if tok.kind == TokenKind::Operator && tok.value == "^" {
            self.advance();
            let right = self.parse_unary()?;
            return apply_binary(&left, &right, b'^', self.prec);
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<String, String> {
        let tok = self.current().clone();
        if tok.kind == TokenKind::Operator {
            match tok.value.as_str() {
                "+" => { self.advance(); return self.parse_unary(); }
                "-" => {
                    self.advance();
                    let v = self.parse_unary()?;
                    return Ok(negate(&v));
                }
                _ => {}
            }
        }
        self.parse_power()
    }

    fn parse_primary(&mut self) -> Result<String, String> {
        let tok = self.current().clone();
        match tok.kind {
            TokenKind::Number => {
                self.advance();
                Ok(tok.value.clone())
            }
            TokenKind::Identifier => {
                self.advance();
                if self.current().kind == TokenKind::LParen && is_function_name(&tok.value) {
                    return self.parse_function_call(&tok.value.clone());
                }
                if let Some(c) = resolve_constant(&tok.value, self.prec) {
                    return Ok(c);
                }
                Err(format!("unknown identifier: {}", tok.value))
            }
            TokenKind::LParen => {
                self.advance();
                let v = self.parse_add_sub()?;
                if self.current().kind != TokenKind::RParen {
                    return Err("missing ')' in expression".into());
                }
                self.advance();
                Ok(v)
            }
            _ => Err(format!("unexpected token {:?}", tok.value)),
        }
    }

    fn parse_function_call(&mut self, name: &str) -> Result<String, String> {
        self.advance(); // skip '('
        let mut args = Vec::new();
        if self.current().kind != TokenKind::RParen {
            loop {
                let arg = self.parse_add_sub()?;
                args.push(arg);
                if self.current().kind == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        if self.current().kind != TokenKind::RParen {
            return Err(format!("missing ')' after {name}"));
        }
        self.advance();
        call_function(name, &args, self.prec, self.degree)
    }
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

fn apply_binary(left: &str, right: &str, op: u8, prec: usize) -> Result<String, String> {
    let l = parse_decimal(left)?;
    let r = parse_decimal(right)?;
    let result = match op {
        b'+' => l + r,
        b'-' => l - r,
        b'*' => l * r,
        b'/' => {
            if r.is_zero() {
                return Err("division by zero".into());
            }
            let scale = prec as u32;
            let result = l.checked_div(r).ok_or("division overflow")?;
            round_to(result, scale)
        }
        b'%' => {
            if r.is_zero() {
                return Err("modulo by zero".into());
            }
            l % r
        }
        b'^' => return pow_decimal(left, right, prec),
        _ => return Err(format!("unsupported operator: {}", op as char)),
    };
    Ok(clean_number(&result.to_string()))
}

fn pow_decimal(base: &str, exp: &str, prec: usize) -> Result<String, String> {
    // Try integer exponent first (exact)
    if let Ok(e) = exp.parse::<i64>() {
        if e == 0 {
            return Ok("1".into());
        }
        if e < 0 {
            let pos = pow_decimal(base, &(-e).to_string(), prec)?;
            return apply_binary("1", &pos, b'/', prec);
        }
        let b = parse_decimal(base)?;
        let mut result = Decimal::ONE;
        let mut factor = b;
        let mut n = e as u64;
        while n > 0 {
            if n % 2 == 1 {
                result = result.checked_mul(factor).ok_or("overflow in pow")?;
            }
            n /= 2;
            if n > 0 {
                factor = factor.checked_mul(factor).ok_or("overflow in pow")?;
            }
        }
        return Ok(clean_number(&result.to_string()));
    }
    // Fall back to f64
    let bf = parse_f64(base)?;
    let ef = parse_f64(exp)?;
    format_float(bf.powf(ef), prec)
}

fn parse_decimal(s: &str) -> Result<Decimal, String> {
    Decimal::from_str(s).map_err(|e| format!("invalid number {:?}: {e}", s))
}

fn parse_f64(s: &str) -> Result<f64, String> {
    s.parse::<f64>().map_err(|_| format!("invalid number {s:?}"))
}

fn round_to(d: Decimal, scale: u32) -> Decimal {
    d.round_dp(scale)
}

fn negate(s: &str) -> String {
    if s == "0" { return "0".into(); }
    if let Some(stripped) = s.strip_prefix('-') {
        stripped.to_string()
    } else {
        format!("-{s}")
    }
}

// ─── Functions ────────────────────────────────────────────────────────────────

fn call_function(name: &str, args: &[String], prec: usize, degree: bool) -> Result<String, String> {
    match name {
        "abs" => {
            require_args(name, args, 1)?;
            Ok(abs_str(&args[0]))
        }
        "sqrt" => {
            require_args(name, args, 1)?;
            let v = parse_f64(&args[0])?;
            if v < 0.0 { return Err("sqrt requires a non-negative argument".into()); }
            format_float(v.sqrt(), prec)
        }
        "sin" | "cos" | "tan" => {
            require_args(name, args, 1)?;
            let mut v = parse_f64(&args[0])?;
            if degree { v = v.to_radians(); }
            let r = match name {
                "sin" => v.sin(), "cos" => v.cos(), _ => v.tan(),
            };
            format_float(r, prec)
        }
        "asin" | "acos" | "atan" => {
            require_args(name, args, 1)?;
            let v = parse_f64(&args[0])?;
            let mut r = match name {
                "asin" => v.asin(), "acos" => v.acos(), _ => v.atan(),
            };
            if degree { r = r.to_degrees(); }
            format_float(r, prec)
        }
        "ln" => {
            require_args(name, args, 1)?;
            let v = parse_f64(&args[0])?;
            if v <= 0.0 { return Err("ln requires a positive argument".into()); }
            format_float(v.ln(), prec)
        }
        "log" => {
            if args.len() != 1 && args.len() != 2 {
                return Err("log expects 1 or 2 arguments".into());
            }
            let v = parse_f64(&args[0])?;
            if v <= 0.0 { return Err("log requires a positive argument".into()); }
            if args.len() == 1 {
                return format_float(v.log10(), prec);
            }
            let base = parse_f64(&args[1])?;
            if base <= 0.0 || (base - 1.0).abs() < 1e-15 {
                return Err("log base must be positive and not equal to 1".into());
            }
            format_float(v.ln() / base.ln(), prec)
        }
        "exp" => {
            require_args(name, args, 1)?;
            let v = parse_f64(&args[0])?;
            format_float(v.exp(), prec)
        }
        "floor" => {
            require_args(name, args, 1)?;
            let v = parse_f64(&args[0])?;
            format_float(v.floor(), 0)
        }
        "ceil" => {
            require_args(name, args, 1)?;
            let v = parse_f64(&args[0])?;
            format_float(v.ceil(), 0)
        }
        "round" => {
            if args.len() != 1 && args.len() != 2 {
                return Err("round expects 1 or 2 arguments".into());
            }
            let d = parse_decimal(&args[0])?;
            if args.len() == 1 {
                return Ok(clean_number(&d.round_dp_with_strategy(0, RoundingStrategy::MidpointAwayFromZero).to_string()));
            }
            let digits: i32 = args[1].parse().map_err(|_| "round precision must be an integer")?;
            if digits >= 0 {
                let rounded = d.round_dp_with_strategy(digits as u32, RoundingStrategy::MidpointAwayFromZero);
                return Ok(clean_number(&rounded.to_string()));
            }
            // Negative digits: round to 10s, 100s, etc.
            let v = parse_f64(&args[0])?;
            let factor = 10f64.powi(-digits);
            format_float((v / factor).round() * factor, 0)
        }
        "min" => {
            if args.len() < 2 {
                return Err("min expects at least 2 arguments".into());
            }
            let mut best = parse_decimal(&args[0])?;
            for a in &args[1..] {
                let v = parse_decimal(a)?;
                if v < best { best = v; }
            }
            Ok(clean_number(&best.to_string()))
        }
        "max" => {
            if args.len() < 2 {
                return Err("max expects at least 2 arguments".into());
            }
            let mut best = parse_decimal(&args[0])?;
            for a in &args[1..] {
                let v = parse_decimal(a)?;
                if v > best { best = v; }
            }
            Ok(clean_number(&best.to_string()))
        }
        "pow" => {
            require_args(name, args, 2)?;
            pow_decimal(&args[0], &args[1], prec)
        }
        _ => Err(format!("unknown function: {name}")),
    }
}

fn require_args(name: &str, args: &[String], want: usize) -> Result<(), String> {
    if args.len() != want {
        Err(format!("{name} expects {want} argument(s)"))
    } else {
        Ok(())
    }
}

// ─── Constants ────────────────────────────────────────────────────────────────

fn resolve_constant(name: &str, prec: usize) -> Option<String> {
    let digits = DEFAULT_PREC.max(prec);
    match name {
        "pi" => Some(trim_constant_digits(PI_DIGITS, digits)),
        "e" => Some(trim_constant_digits(E_DIGITS, digits)),
        "tau" => {
            // 2 * pi
            let pi = trim_constant_digits(PI_DIGITS, digits + 2);
            let result = parse_decimal(&pi).ok()? * Decimal::TWO;
            Some(trim_constant_digits(&result.to_string(), digits))
        }
        _ => None,
    }
}

fn trim_constant_digits(value: &str, digits: usize) -> String {
    let dot = match value.find('.') {
        Some(i) => i,
        None => return value.to_string(),
    };
    let end = (dot + digits + 1).min(value.len());
    clean_number(&value[..end])
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn is_function_name(name: &str) -> bool {
    matches!(
        name,
        "abs" | "acos" | "asin" | "atan" | "ceil" | "cos" | "exp" | "floor"
            | "ln" | "log" | "max" | "min" | "pow" | "round" | "sin" | "sqrt" | "tan"
    )
}

fn abs_str(s: &str) -> String {
    let s = clean_number(s);
    if let Some(stripped) = s.strip_prefix('-') {
        stripped.to_string()
    } else {
        s
    }
}

fn clean_number(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() { return "0".into(); }

    // Remove leading +
    let s = s.strip_prefix('+').unwrap_or(s);

    // Handle sign
    let (sign, s) = if let Some(rest) = s.strip_prefix('-') {
        ("-", rest)
    } else {
        ("", s)
    };

    // Normalise leading dot
    let s = if s.starts_with('.') { format!("0{s}") } else { s.to_string() };

    let s = if s.contains('.') {
        let s = s.trim_end_matches('0');
        s.trim_end_matches('.')
    } else {
        &s
    };

    // Trim leading zeros on integer part
    let (int_part, frac_part) = if let Some(dot) = s.find('.') {
        (&s[..dot], Some(&s[dot + 1..]))
    } else {
        (s, None)
    };
    let int_part = int_part.trim_start_matches('0');
    let int_part = if int_part.is_empty() { "0" } else { int_part };

    let result = if let Some(frac) = frac_part {
        if frac.is_empty() { int_part.to_string() } else { format!("{int_part}.{frac}") }
    } else {
        int_part.to_string()
    };

    if result == "0" {
        return "0".into();
    }
    if sign == "-" { format!("-{result}") } else { result }
}

fn format_float(v: f64, prec: usize) -> Result<String, String> {
    if v.is_nan() || v.is_infinite() {
        return Err("result is not a finite number".into());
    }
    let snapped = snap_tiny(v, prec);
    let s = format!("{:.prec$}", snapped, prec = prec);
    Ok(clean_number(&s))
}

fn snap_tiny(v: f64, prec: usize) -> f64 {
    let eps = 10f64.powi(-(prec.max(4) as i32 + 2));
    if v.abs() < eps { 0.0 } else { v }
}

// ─── Evaluate ─────────────────────────────────────────────────────────────────

fn evaluate_expression(expr: &str, prec: usize, degree: bool) -> Result<String, String> {
    let tokens = tokenize(expr)?;
    let mut parser = ExprParser::new(tokens, prec, degree);
    parser.parse()
}
