fn main() {
    let s = " |---------|------|---------|---------| ";
    for c in s.chars() {
        println!("char: '{}', unicode: {:x}", c, c as u32);
    }
}
