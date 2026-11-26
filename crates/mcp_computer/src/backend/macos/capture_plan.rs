//! Pure argument and geometry planning for screencapture's rectangle mode.

pub(super) type Bounds = (f64, f64, f64, f64);

pub(super) fn region_arg((x, y, width, height): Bounds) -> String {
    format!(
        "-R{},{},{},{}",
        x.round() as i64,
        y.round() as i64,
        width.round() as i64,
        height.round() as i64
    )
}

pub(super) fn plan_display(
    display_id: u32,
    bounds_for_id: impl FnOnce(u32) -> Result<Bounds, String>,
) -> Result<(String, Bounds), String> {
    // Screen info exposes CGDirectDisplayID, not the one-based ordinal `-D`
    // expects. Target the identified display's global rectangle instead.
    let (x, y, width, height) = bounds_for_id(display_id)?;
    let [x, y, width, height] = [x, y, width, height].map(f64::round);
    if ![x, y, width, height].iter().all(|value| value.is_finite()) || width <= 0.0 || height <= 0.0
    {
        return Err(format!(
            "display id {display_id} has no usable capture bounds"
        ));
    }
    let bounds = (x, y, width, height);
    Ok((region_arg(bounds), bounds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_uses_the_display_id_bounds_not_a_cli_ordinal() {
        let display_id = 72_871_299;
        let (arg, bounds) = plan_display(display_id, |id| {
            assert_eq!(id, display_id);
            Ok((-1920.0, -300.0, 1920.0, 1080.0))
        })
        .unwrap();
        assert_eq!(arg, "-R-1920,-300,1920,1080");
        assert_eq!(bounds, (-1920.0, -300.0, 1920.0, 1080.0));
    }

    #[test]
    fn display_geometry_uses_the_same_rounded_points_as_the_helper() {
        let (arg, bounds) = plan_display(65, |_| Ok((100.6, -20.6, 1200.7, 800.4))).unwrap();
        assert_eq!(arg, "-R101,-21,1201,800");
        assert_eq!(bounds, (101.0, -21.0, 1201.0, 800.0));
    }

    #[test]
    fn display_capture_requires_resolved_nonempty_bounds() {
        let stale = plan_display(73, |_| Err("not an active display".into()));
        assert_eq!(stale.unwrap_err(), "not an active display");
        let empty = plan_display(65, |_| Ok((0.0, 0.0, 0.0, 0.0)));
        assert!(empty.unwrap_err().contains("no usable capture bounds"));
    }
}
