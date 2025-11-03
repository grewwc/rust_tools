pub fn find_first_non_blank<'a>(arr: &[&'a str]) -> Option<&'a str> {
    arr.iter().find(|val| val.len() > 0).map(|val| *val)
}
