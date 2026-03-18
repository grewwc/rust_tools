#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoRecord {
    pub id: String,
    pub add_date: i64,
    pub modified_date: i64,
    pub finished: bool,
    pub hold: bool,
    pub title: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoTag {
    pub id: String,
    pub name: String,
    pub count: i64,
    pub modified_date: i64,
}
