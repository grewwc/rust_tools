pub fn find_first_non_blank<'a>(arr: &[&'a str]) -> Option<&'a str> {
    arr.iter().find(|val| !val.is_empty()).copied()
}
