use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

pub trait IntegerLike: Copy + Ord {
    fn to_i128(self) -> i128;
    fn from_i128(v: i128) -> Self;
}

macro_rules! impl_integer_like_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl IntegerLike for $t {
                fn to_i128(self) -> i128 {
                    self as i128
                }
                fn from_i128(v: i128) -> Self {
                    v as Self
                }
            }
        )*
    };
}

macro_rules! impl_integer_like_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl IntegerLike for $t {
                fn to_i128(self) -> i128 {
                    self as i128
                }
                fn from_i128(v: i128) -> Self {
                    if v < 0 {
                        0
                    } else {
                        v as Self
                    }
                }
            }
        )*
    };
}

impl_integer_like_signed!(i8, i16, i32, i64, i128, isize);
impl_integer_like_unsigned!(u8, u16, u32, u64, usize);

pub fn insertion_sort<T: Ord>(arr: &mut [T]) {
    insertion_sort_by(arr, |a, b| a.cmp(b));
}

pub fn insertion_sort_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let mut cmp = cmp;
    insertion_sort_by_ref(arr, &mut cmp);
}

pub fn quick_sort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    arr.sort_unstable();
}

pub fn quick_sort_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let mut cmp = cmp;
    arr.sort_unstable_by(|a, b| cmp(a, b));
}

pub fn stable_sort<T: Ord>(arr: &mut [T]) {
    arr.sort();
}

pub fn stable_sort_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let mut cmp = cmp;
    arr.sort_by(|a, b| cmp(a, b));
}

pub fn sort<T: Ord>(arr: &mut [T]) {
    quick_sort(arr);
}

pub fn sort_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    quick_sort_by(arr, cmp);
}

pub fn shell_sort<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    let mut gap = 1usize;
    while gap < len / 3 {
        gap = gap * 3 + 1;
    }

    while gap > 0 {
        for i in gap..len {
            let mut j = i;
            while j >= gap && arr[j] < arr[j - gap] {
                arr.swap(j, j - gap);
                j -= gap;
            }
        }
        gap /= 3;
    }
}

pub fn heap_sort<T: Ord>(arr: &mut [T], reverse: bool) {
    arr.sort_unstable();
    if reverse {
        arr.reverse();
    }
}

pub fn count_sort<T: IntegerLike>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }

    let (min_v, max_v) = min_max(arr);
    let min_i = min_v.to_i128();
    let max_i = max_v.to_i128();
    if max_i < min_i {
        return;
    }

    let range = max_i - min_i + 1;
    if range <= 0 {
        return;
    }

    const MAX_BUCKETS: i128 = 5_000_000;
    if range > MAX_BUCKETS {
        arr.sort_unstable();
        return;
    }

    let mut count = vec![0usize; range as usize];
    for &value in arr.iter() {
        let idx = (value.to_i128() - min_i) as usize;
        count[idx] += 1;
    }

    let mut write_idx = 0usize;
    for (offset, &c) in count.iter().enumerate() {
        let value = T::from_i128(min_i + offset as i128);
        for _ in 0..c {
            arr[write_idx] = value;
            write_idx += 1;
        }
    }
}

pub fn radix_sort(nums: &mut [i64]) {
    if nums.len() <= 1 {
        return;
    }

    if nums.iter().any(|&value| value < 0) {
        nums.sort_unstable();
        return;
    }

    let max_val = *nums.iter().max().unwrap_or(&0);
    if max_val <= 0 {
        return;
    }

    let mut exp = 1i64;
    while max_val / exp > 0 {
        counting_sort_by_digit(nums, exp);
        match exp.checked_mul(10) {
            Some(next) => exp = next,
            None => break,
        }
    }
}

pub fn top_k<T: Ord + Clone>(arr: &[T], mut k: usize, min_k: bool) -> Vec<T> {
    if k == 0 || arr.is_empty() {
        return Vec::new();
    }
    if k > arr.len() {
        k = arr.len();
    }

    if min_k {
        let mut heap: BinaryHeap<T> = BinaryHeap::with_capacity(k);
        for value in arr {
            let value = value.clone();
            if heap.len() < k {
                heap.push(value);
                continue;
            }
            if let Some(top) = heap.peek()
                && value < *top
            {
                heap.pop();
                heap.push(value);
            }
        }
        let mut out = heap.into_vec();
        out.sort();
        return out;
    }

    let mut heap: BinaryHeap<Reverse<T>> = BinaryHeap::with_capacity(k);
    for value in arr {
        let value = value.clone();
        if heap.len() < k {
            heap.push(Reverse(value));
            continue;
        }
        if let Some(top) = heap.peek()
            && value > top.0
        {
            heap.pop();
            heap.push(Reverse(value));
        }
    }

    let mut out = heap.into_iter().map(|value| value.0).collect::<Vec<_>>();
    out.sort();
    out.reverse();
    out
}

pub fn are_sorted<T: Ord>(arr: &[T]) -> bool {
    arr.windows(2).all(|pair| pair[0] <= pair[1])
}

pub fn tim_sort<T: Ord + Clone>(arr: &mut [T]) {
    tim_sort_by(arr, |a, b| a.cmp(b));
}

pub fn tim_sort_by<T: Clone, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let mut cmp = cmp;
    let len = arr.len();
    if len <= 1 {
        return;
    }

    let min_run = calc_min_run(len);
    if len <= min_run {
        insertion_sort_by_ref(arr, &mut cmp);
        return;
    }

    let mut start = 0usize;
    while start < len {
        let end = (start + min_run).min(len);
        insertion_sort_by_ref(&mut arr[start..end], &mut cmp);
        start += min_run;
    }

    let mut size = min_run;
    while size < len {
        let mut run_start = 0usize;
        while run_start < len {
            let mid = run_start + size;
            if mid >= len {
                break;
            }
            let end = (run_start + size * 2).min(len);
            merge_by_ref(arr, run_start, mid, end, &mut cmp);
            run_start += size * 2;
        }
        size *= 2;
    }
}

pub fn calc_min_run(mut n: usize) -> usize {
    let mut r = 0usize;
    while n >= 64 {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

pub fn sort_insertion<T: Ord>(arr: &mut [T]) {
    insertion_sort(arr);
}

pub fn sort_insertion_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    insertion_sort_by(arr, cmp);
}

pub fn sort_quick<T: Ord>(arr: &mut [T]) {
    quick_sort(arr);
}

pub fn sort_quick_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    quick_sort_by(arr, cmp);
}

pub fn sort_stable<T: Ord>(arr: &mut [T]) {
    stable_sort(arr);
}

pub fn sort_stable_by<T, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    stable_sort_by(arr, cmp);
}

pub fn sort_shell<T: Ord>(arr: &mut [T]) {
    shell_sort(arr);
}

pub fn sort_heap<T: Ord>(arr: &mut [T], reverse: bool) {
    heap_sort(arr, reverse);
}

pub fn sort_count<T: IntegerLike>(arr: &mut [T]) {
    count_sort(arr);
}

pub fn sort_radix(arr: &mut [i64]) {
    radix_sort(arr);
}

pub fn sort_time<T: Ord + Clone>(arr: &mut [T]) {
    tim_sort(arr);
}

pub fn sort_time_by<T: Clone, F>(arr: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    tim_sort_by(arr, cmp);
}

pub fn sort_top_k<T: Ord + Clone>(arr: &[T], k: usize, min_k: bool) -> Vec<T> {
    top_k(arr, k, min_k)
}

pub fn sort_are_sorted<T: Ord>(arr: &[T]) -> bool {
    are_sorted(arr)
}

fn insertion_sort_by_ref<T, F>(arr: &mut [T], cmp: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if arr.len() <= 1 {
        return;
    }
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && cmp(&arr[j], &arr[j - 1]) == Ordering::Less {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}

fn merge_by_ref<T: Clone, F>(arr: &mut [T], start: usize, mid: usize, end: usize, cmp: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if end - start <= 1 {
        return;
    }

    let left = arr[start..mid].to_vec();
    let right = arr[mid..end].to_vec();
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    let mut write_idx = start;

    while left_idx < left.len() && right_idx < right.len() {
        if cmp(&left[left_idx], &right[right_idx]) != Ordering::Greater {
            arr[write_idx] = left[left_idx].clone();
            left_idx += 1;
        } else {
            arr[write_idx] = right[right_idx].clone();
            right_idx += 1;
        }
        write_idx += 1;
    }

    while left_idx < left.len() {
        arr[write_idx] = left[left_idx].clone();
        left_idx += 1;
        write_idx += 1;
    }

    while right_idx < right.len() {
        arr[write_idx] = right[right_idx].clone();
        right_idx += 1;
        write_idx += 1;
    }
}

fn min_max<T: IntegerLike>(arr: &[T]) -> (T, T) {
    let mut min_v = arr[0];
    let mut max_v = arr[0];
    for &value in arr.iter().skip(1) {
        if value < min_v {
            min_v = value;
        }
        if value > max_v {
            max_v = value;
        }
    }
    (min_v, max_v)
}

fn counting_sort_by_digit(nums: &mut [i64], exp: i64) {
    let mut count = [0usize; 10];
    let mut output = vec![0i64; nums.len()];

    for &num in nums.iter() {
        let digit = ((num / exp) % 10) as usize;
        count[digit] += 1;
    }

    for idx in 1..10 {
        count[idx] += count[idx - 1];
    }

    for idx in (0..nums.len()).rev() {
        let digit = ((nums[idx] / exp) % 10) as usize;
        let out_idx = count[digit] - 1;
        output[out_idx] = nums[idx];
        count[digit] -= 1;
    }

    nums.copy_from_slice(&output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ordered_sorts() {
        let mut a = vec![5, 1, 4, 2, 3];
        insertion_sort(&mut a);
        assert_eq!(a, vec![1, 2, 3, 4, 5]);

        let mut b = vec![5, 1, 4, 2, 3];
        quick_sort(&mut b);
        assert_eq!(b, vec![1, 2, 3, 4, 5]);

        let mut c = vec![5, 1, 4, 2, 3];
        shell_sort(&mut c);
        assert_eq!(c, vec![1, 2, 3, 4, 5]);

        let mut d = vec![5, 1, 4, 2, 3];
        heap_sort(&mut d, false);
        assert_eq!(d, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_stable_and_custom_sort() {
        let mut arr = vec![(2, 'b'), (1, 'a'), (2, 'a')];
        stable_sort_by(&mut arr, |a, b| a.0.cmp(&b.0));
        assert_eq!(arr, vec![(1, 'a'), (2, 'b'), (2, 'a')]);

        sort_by(&mut arr, |a, b| b.0.cmp(&a.0));
        assert_eq!(arr[0].0, 2);
    }

    #[test]
    fn test_count_sort_and_radix_sort() {
        let mut count_arr = vec![3i32, -1, 2, 2, 0];
        count_sort(&mut count_arr);
        assert_eq!(count_arr, vec![-1, 0, 2, 2, 3]);

        let mut radix_arr = vec![170i64, 45, 75, 90, 802, 24, 2, 66];
        radix_sort(&mut radix_arr);
        assert_eq!(radix_arr, vec![2, 24, 45, 66, 75, 90, 170, 802]);
    }

    #[test]
    fn test_top_k_and_are_sorted() {
        let arr = vec![9, 1, 7, 3, 5, 8, 2];
        assert_eq!(top_k(&arr, 3, false), vec![9, 8, 7]);
        assert_eq!(top_k(&arr, 3, true), vec![1, 2, 3]);

        assert!(are_sorted(&[1, 2, 3, 4]));
        assert!(!are_sorted(&[1, 3, 2, 4]));
    }

    #[test]
    fn test_tim_sort() {
        let mut arr = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        tim_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let mut words = vec!["bbb", "a", "cc"];
        tim_sort_by(&mut words, |a, b| a.len().cmp(&b.len()));
        assert_eq!(words, vec!["a", "cc", "bbb"]);
    }
}
