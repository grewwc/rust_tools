use chrono::{Datelike, Duration, Local, NaiveDate};

pub fn today_local_date() -> NaiveDate {
    Local::now().date_naive()
}

pub fn format_log_tag(date: NaiveDate) -> String {
    format!("log.{:04}-{:02}-{:02}", date.year(), date.month(), date.day())
}

pub fn format_week_tag(date: NaiveDate) -> String {
    format!("week.{:04}-{:02}-{:02}", date.year(), date.month(), date.day())
}

pub fn first_day_of_week(date: NaiveDate) -> NaiveDate {
    let weekday = date.weekday().num_days_from_monday() as i64;
    date - Duration::days(weekday)
}

pub fn add_days(date: NaiveDate, days: i64) -> NaiveDate {
    date + Duration::days(days)
}

