pub fn bisect_left<T: PartialOrd>(arr: &[T], target: &T) -> usize {
    if arr.is_empty() {
        return 0;
    }
    let (mut lo, mut hi) = (0 as usize, arr.len());
    while lo < hi {
        let mid = (hi - lo) / 2 + lo;
        if target > &arr[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub fn bisect_right<T: PartialOrd>(arr: &[T], target: &T) -> usize {
    if arr.is_empty() {
        return 0;
    }

    let (mut lo, mut hi) = (0 as usize, arr.len());
    while lo < hi {
        let mid = (hi - lo) / 2 + lo;
        if target < &arr[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
