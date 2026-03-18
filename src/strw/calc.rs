use std::sync::LazyLock;

use regex::Regex;

static MULTI_MINUS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(--)+").unwrap());
static POSITIVE_NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\+(\d+)").unwrap());

pub trait CalcStrExt: AsRef<str> {
    fn plus<S: AsRef<str>>(&self, rhs: S) -> String {
        plus(self.as_ref(), rhs.as_ref())
    }

    fn plus_all<I, S>(&self, rhs: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut res = self.as_ref().to_string();
        for x in rhs {
            res = plus(&res, x.as_ref());
        }
        res
    }

    fn minus<S: AsRef<str>>(&self, rhs: S) -> String {
        minus(self.as_ref(), rhs.as_ref())
    }

    fn neg(&self) -> String {
        neg(self.as_ref())
    }

    fn mul<S: AsRef<str>>(&self, rhs: S) -> String {
        mul(self.as_ref(), rhs.as_ref())
    }

    fn mul_all<I, S>(&self, rhs: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut res = self.as_ref().to_string();
        for x in rhs {
            res = mul(&res, x.as_ref());
        }
        res
    }

    fn div<S: AsRef<str>>(&self, rhs: S, num_digit_to_keep: i32) -> String {
        div(self.as_ref(), rhs.as_ref(), num_digit_to_keep)
    }

    fn modulo<S: AsRef<str>>(&self, rhs: S) -> String {
        modulo(self.as_ref(), rhs.as_ref())
    }
}

impl<T: AsRef<str>> CalcStrExt for T {}

pub fn plus(a: &str, b: &str) -> String {
    plus2(a, b)
}

pub fn plus_all(xs: &[&str]) -> String {
    if xs.is_empty() {
        return String::new();
    }
    let mut res = xs[0].to_string();
    for x in &xs[1..] {
        res = plus2(&res, x);
    }
    res
}

pub fn minus(a: &str, b: &str) -> String {
    minus2(a, b)
}

pub fn neg(a: &str) -> String {
    if a.is_empty() {
        return String::new();
    }
    if a.as_bytes()[0] == b'-' {
        return a[1..].to_string();
    }
    format!("-{a}")
}

pub fn mul(a: &str, b: &str) -> String {
    mul2(a, b)
}

pub fn mul_all(xs: &[&str]) -> String {
    if xs.is_empty() {
        return String::new();
    }
    let mut res = xs[0].to_string();
    for x in &xs[1..] {
        res = mul2(&res, x);
    }
    res
}

pub fn div(a: &str, b: &str, num_digit_to_keep: i32) -> String {
    div2(a, b, num_digit_to_keep)
}

pub fn modulo(a: &str, b: &str) -> String {
    let quotient = div2(a, b, 0);
    let tmp = mul2(&quotient, b);
    let res = minus2(a, &tmp);
    let mut quotient = quotient;
    if res.as_bytes().first() == Some(&b'-') {
        quotient = minus2(&quotient, "1");
    }
    minus2(a, &mul2(&quotient, b))
}

fn process_input_str(input: &str) -> String {
    let input = MULTI_MINUS.replace_all(input, "+");
    POSITIVE_NUMBER.replace_all(&input, "$1").to_string()
}

fn strip_leading_minus(s: &str) -> &str {
    s.strip_prefix('-').unwrap_or(s)
}

fn remove_leading_zero(s: &str) -> (String, usize) {
    if s.is_empty() {
        return (String::new(), 0);
    }
    let bytes = s.as_bytes();
    let mut idx = 0usize;
    while idx + 1 < bytes.len() && bytes[idx] == b'0' && bytes[idx + 1] != b'.' {
        idx += 1;
    }
    (s[idx..].to_string(), idx)
}

fn remove_suffix_zero(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let mut out = s.trim_end_matches('0').to_string();
    if out.ends_with('.') {
        out.pop();
    }
    out
}

fn count_dot_digit(a: &str, b: &str, add: bool) -> (usize, usize, usize) {
    let a_dot = a.as_bytes().iter().filter(|c| **c == b'.').count();
    let b_dot = b.as_bytes().iter().filter(|c| **c == b'.').count();
    if a_dot > 1 || b_dot > 1 {
        panic!("invalid number: {a}, {b}");
    }

    let ai = a.find('.').unwrap_or_else(|| a.len().saturating_sub(1));
    let bi = b.find('.').unwrap_or_else(|| b.len().saturating_sub(1));
    let ca = a.len().saturating_sub(ai + 1);
    let cb = b.len().saturating_sub(bi + 1);
    if !add {
        return (ca, cb, ca + cb);
    }
    (ca, cb, ca.max(cb))
}

fn prepend_leading_zero(s: &str, decimal_count: usize) -> String {
    if decimal_count == 0 {
        return s.to_string();
    }
    let mut out = s.to_string();
    let mut leading0 = false;
    if out.len() <= decimal_count {
        out = "0".repeat(decimal_count - out.len()) + &out;
        leading0 = true;
    }
    let split = out.len() - decimal_count;
    out.insert(split, '.');
    if leading0 {
        out.insert(0, '0');
    }
    out
}

fn add_integer_strings(a: &str, b: &str) -> String {
    let mut a = a.as_bytes().to_vec();
    let mut b = b.as_bytes().to_vec();
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }

    let mut res = vec![b'0'; a.len() + 1];
    let mut carry = 0u8;
    let mut i: isize = a.len() as isize - 1;
    let mut j: isize = b.len() as isize - 1;
    let mut idx: isize = res.len() as isize - 1;
    while idx >= 0 {
        let mut val = carry as u16;
        if i >= 0 {
            val += (a[i as usize] - b'0') as u16;
            i -= 1;
        }
        if j >= 0 {
            val += (b[j as usize] - b'0') as u16;
            j -= 1;
        }
        if val >= 10 {
            carry = 1;
            val -= 10;
        } else {
            carry = 0;
        }
        res[idx as usize] = (val as u8) + b'0';
        idx -= 1;
    }
    let mut k = 0usize;
    while k + 1 < res.len() && res[k] == b'0' {
        k += 1;
    }
    String::from_utf8_lossy(&res[k..]).to_string()
}

fn sub_integer_strings(a: &str, b: &str) -> String {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut res = vec![b'0'; a_bytes.len()];
    let mut i: isize = a_bytes.len() as isize - 1;
    let mut j: isize = b_bytes.len() as isize - 1;
    let mut idx: isize = res.len() as isize - 1;
    let mut borrow = 0i16;
    while idx >= 0 {
        let mut val_a = 0i16;
        if i >= 0 {
            val_a = (a_bytes[i as usize] - b'0') as i16;
            i -= 1;
        }
        let mut val_b = 0i16;
        if j >= 0 {
            val_b = (b_bytes[j as usize] - b'0') as i16;
            j -= 1;
        }
        let mut val = val_a - val_b - borrow;
        if val < 0 {
            val += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        res[idx as usize] = (val as u8) + b'0';
        idx -= 1;
    }
    let mut k = 0usize;
    while k + 1 < res.len() && res[k] == b'0' {
        k += 1;
    }
    String::from_utf8_lossy(&res[k..]).to_string()
}

fn plus2(a: &str, b: &str) -> String {
    let mut a = process_input_str(a.trim());
    let mut b = process_input_str(b.trim());

    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }

    if !a.starts_with('-') && b.starts_with('-') {
        return minus2(&a, strip_leading_minus(&b));
    }
    if a.starts_with('-') && !b.starts_with('-') {
        return minus2(&b, strip_leading_minus(&a));
    }

    let mut is_minus = false;
    if a.starts_with('-') && b.starts_with('-') {
        a = strip_leading_minus(&a).to_string();
        b = strip_leading_minus(&b).to_string();
        is_minus = true;
    }

    (a, _) = remove_leading_zero(&a);
    (b, _) = remove_leading_zero(&b);

    let (n1, n2, num_dot) = count_dot_digit(&a, &b, true);
    let mut a = a.replace('.', "");
    let mut b = b.replace('.', "");
    a.push_str(&"0".repeat(num_dot.saturating_sub(n1)));
    b.push_str(&"0".repeat(num_dot.saturating_sub(n2)));

    let mut out = add_integer_strings(&a, &b);
    out = prepend_leading_zero(&out, num_dot);
    out = remove_suffix_zero(&out);
    if is_minus {
        out.insert(0, '-');
    }
    out
}

fn minus2(a: &str, b: &str) -> String {
    let a = process_input_str(a.trim());
    let b = process_input_str(b.trim());

    if a.is_empty() {
        if b.starts_with('-') {
            return strip_leading_minus(&b).to_string();
        }
        return format!("-{b}");
    }
    if b.is_empty() {
        return a;
    }

    if !a.starts_with('-') && b.starts_with('-') {
        return plus2(&a, strip_leading_minus(&b));
    }
    if a.starts_with('-') && !b.starts_with('-') {
        return format!("-{}", plus2(strip_leading_minus(&a), &b));
    }
    if a.starts_with('-') && b.starts_with('-') {
        let res = minus2(strip_leading_minus(&a), strip_leading_minus(&b));
        if let Some(stripped) = res.strip_prefix('-') {
            return stripped.to_string();
        }
        if res == "0" {
            return "0".to_string();
        }
        return format!("-{res}");
    }

    let (n1, n2, num_dot) = count_dot_digit(&a, &b, true);
    let mut a = a.replace('.', "");
    let mut b = b.replace('.', "");

    let mut is_minus = false;
    a = strip_leading_minus(&a).to_string();
    b = strip_leading_minus(&b).to_string();
    (a, _) = remove_leading_zero(&a);
    (b, _) = remove_leading_zero(&b);
    a.push_str(&"0".repeat(num_dot.saturating_sub(n1)));
    b.push_str(&"0".repeat(num_dot.saturating_sub(n2)));

    if a.len() < b.len() || (a.len() == b.len() && a < b) {
        is_minus = true;
        std::mem::swap(&mut a, &mut b);
    }

    let mut out = sub_integer_strings(&a, &b);
    out = prepend_leading_zero(&out, num_dot);
    out = remove_suffix_zero(&out);
    if is_minus {
        out.insert(0, '-');
    }
    out
}

fn mul_integer_strings(a: &str, b: &str) -> String {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut res = vec![0u32; a_bytes.len() + b_bytes.len()];

    for (i, &da) in a_bytes.iter().rev().enumerate() {
        let va = (da - b'0') as u32;
        for (j, &db) in b_bytes.iter().rev().enumerate() {
            let vb = (db - b'0') as u32;
            res[i + j] += va * vb;
        }
    }

    let mut carry = 0u32;
    for v in &mut res {
        let total = *v + carry;
        *v = total % 10;
        carry = total / 10;
    }
    while carry > 0 {
        res.push(carry % 10);
        carry /= 10;
    }

    while res.len() > 1 && *res.last().unwrap() == 0 {
        res.pop();
    }

    let out: String = res
        .into_iter()
        .rev()
        .map(|d| (d as u8 + b'0') as char)
        .collect();
    out
}

fn mul2(a: &str, b: &str) -> String {
    let mut a = process_input_str(a.trim());
    let mut b = process_input_str(b.trim());
    if a.is_empty() || b.is_empty() {
        return String::new();
    }

    let is_minus =
        (a.starts_with('-') && !b.starts_with('-')) || (!a.starts_with('-') && b.starts_with('-'));
    a = strip_leading_minus(&a).to_string();
    b = strip_leading_minus(&b).to_string();

    if a.len() > b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    if a == "0" || b == "0" {
        return "0".to_string();
    }

    let (_, _, num_dot) = count_dot_digit(&a, &b, false);
    a = a.replace('.', "");
    b = b.replace('.', "");
    (a, _) = remove_leading_zero(&a);
    (b, _) = remove_leading_zero(&b);

    let mut out = mul_integer_strings(&a, &b);
    out = prepend_leading_zero(&out, num_dot);
    out = remove_suffix_zero(&out);
    (out, _) = remove_leading_zero(&out);
    if is_minus && out != "0" {
        out.insert(0, '-');
    }
    out
}

fn do_div(a: &str, b: &str) -> (String, String, bool) {
    if b == "1" {
        return ("0".to_string(), a.to_string(), true);
    }

    let mut low = 0i32;
    let mut high = 9i32;
    let mut res = 0i32;

    while low <= high {
        let mid = (low + high) / 2;
        let product = mul2(b, &mid.to_string());
        let cmp = minus2(a, &product);
        if cmp == "0" || (!cmp.is_empty() && !cmp.starts_with('-')) {
            res = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    let remainder = minus2(a, &mul2(b, &res.to_string()));
    (remainder.clone(), res.to_string(), remainder == "0")
}

fn round(s: &str, digit_to_keep: i32) -> String {
    let idx = match s.rfind('.') {
        Some(i) => i,
        None => return s.to_string(),
    };

    if digit_to_keep <= 0 {
        let mut add = "0";
        if s.as_bytes().get(idx + 1).copied().unwrap_or(b'0') >= b'5' {
            add = "1";
        }
        return plus2(&s[..idx], add);
    }

    let digit_to_keep = digit_to_keep as usize;
    let num_digit = s.len() - idx - 1;
    if num_digit <= digit_to_keep {
        return s.to_string();
    }
    let val = s.as_bytes()[idx + digit_to_keep + 1];
    if val < b'5' {
        return s[..idx + digit_to_keep + 1].to_string();
    }
    let add = format!("0.{}1", "0".repeat(digit_to_keep.saturating_sub(1)));
    plus2(&s[..idx + digit_to_keep + 1], &add)
}

fn div2(a: &str, b: &str, num_digit_to_keep: i32) -> String {
    let mut a = process_input_str(a.trim());
    let mut b = process_input_str(b.trim());
    if a.is_empty() || b.is_empty() {
        return String::new();
    }
    let keep = num_digit_to_keep.max(0) as usize;

    let is_minus =
        (a.starts_with('-') && !b.starts_with('-')) || (!a.starts_with('-') && b.starts_with('-'));
    a = strip_leading_minus(&a).to_string();
    b = strip_leading_minus(&b).to_string();

    let (d1, d2, d) = count_dot_digit(&a, &b, false);
    let a_no_decimal = a.replace('.', "");
    let b_no_decimal = b.replace('.', "");
    (a, _) = remove_leading_zero(&a_no_decimal);
    (b, _) = remove_leading_zero(&b_no_decimal);

    if b == "0" {
        panic!("b is 0");
    }
    if a == "0" {
        return "0".to_string();
    }

    a.push_str(&"0".repeat(d.saturating_sub(d1)));
    b.push_str(&"0".repeat(d.saturating_sub(d2)));

    let decimal_pos: isize = a.len() as isize - (d.saturating_sub(d1) as isize);
    let total_digits = a.len() + keep;
    let mut res = String::new();
    let mut remainder = "0".to_string();

    for i in 0..total_digits {
        if i < a.len() {
            remainder.push(a.as_bytes()[i] as char);
        } else {
            remainder.push('0');
        }
        (remainder, _) = remove_leading_zero(&remainder);
        if remainder.is_empty() {
            remainder = "0".to_string();
        }

        let (_, digit, _) = do_div(&remainder, &b);
        res.push_str(&digit);
        remainder = minus2(&remainder, &mul2(&b, &digit));

        if i as isize == decimal_pos - 1 && i + 1 < a.len() {
            res.push('.');
        }
    }

    if !res.contains('.') {
        if decimal_pos <= 0 {
            res = format!("0.{}{}", "0".repeat((-decimal_pos) as usize), res);
        } else {
            let p = decimal_pos as usize;
            if p < res.len() {
                res.insert(p, '.');
            } else {
                res.push_str(".0");
            }
        }
    }

    (res, _) = remove_leading_zero(&res);
    if res.is_empty() || res == "." {
        res = "0".to_string();
    }

    if is_minus && res != "0" {
        res.insert(0, '-');
    }
    res = round(&res, num_digit_to_keep);
    remove_suffix_zero(&res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plus_minus_basic() {
        assert_eq!(plus("1", "2"), "3");
        assert_eq!(minus("10", "3"), "7");
        assert_eq!(minus("3", "10"), "-7");
        assert_eq!(plus("-3", "10"), "7");
        assert_eq!(plus("-3", "-10"), "-13");
    }

    #[test]
    fn test_decimal_plus_minus() {
        assert_eq!(plus("0.3", "0.513"), "0.813");
        assert_eq!(minus("0.513", "0.3"), "0.213");
        assert_eq!(minus("1", "0.3"), "0.7");
    }

    #[test]
    fn test_mul_div_mod() {
        assert_eq!(mul("12", "34"), "408");
        assert_eq!(mul("0.3", "0.02"), "0.006");
        assert_eq!(div("10", "4", 3), "2.5");
        assert_eq!(div("1", "8", 4), "0.125");
        assert_eq!(modulo("10", "3"), "1");
    }

    #[test]
    fn test_calc_str_ext_methods() {
        assert_eq!("1".plus("2"), "3");
        assert_eq!("10".minus("3"), "7");
        assert_eq!("0.3".mul("0.02"), "0.006");
        assert_eq!("1".div("8", 4), "0.125");
        assert_eq!("10".modulo("3"), "1");
        assert_eq!("12".plus_all(["3", "4"]), "19");
        assert_eq!("2".mul_all(["3", "4"]), "24");
        assert_eq!("12".neg(), "-12");
        assert_eq!("-12".neg(), "12");
    }
}
