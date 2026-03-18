pub mod db;
pub mod history;
pub mod model;
pub mod mongo;
pub mod search;
pub mod time;
pub mod ui;
pub mod sync;

pub use db::MemoDb;
pub use model::{MemoRecord, MemoTag};
pub use mongo::MemoMongo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoBackendMode {
    Auto,
    Mongo,
    Sqlite,
}

impl MemoBackendMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "mongo" | "mongodb" => Self::Mongo,
            "sqlite" | "sqlite3" => Self::Sqlite,
            "" | "auto" => Self::Auto,
            _ => Self::Auto,
        }
    }
}

pub enum MemoBackend {
    Sqlite(MemoDb),
    Mongo(MemoMongo),
}

impl MemoBackend {
    pub fn open(mode: MemoBackendMode, sqlite_path_override: Option<&str>) -> Result<Self, String> {
        match mode {
            MemoBackendMode::Sqlite => {
                if let Some(p) = sqlite_path_override.filter(|p| !p.trim().is_empty()) {
                    return MemoDb::open(p.trim())
                        .map(MemoBackend::Sqlite)
                        .map_err(|e| e.to_string());
                }
                MemoDb::open_default()
                    .map(MemoBackend::Sqlite)
                    .map_err(|e| e.to_string())
            }
            MemoBackendMode::Mongo => MemoMongo::connect_default().map(MemoBackend::Mongo),
            MemoBackendMode::Auto => {
                if let Ok(mongo) = MemoMongo::connect_default() {
                    return Ok(MemoBackend::Mongo(mongo));
                }
                if let Some(p) = sqlite_path_override.filter(|p| !p.trim().is_empty()) {
                    return MemoDb::open(p.trim())
                        .map(MemoBackend::Sqlite)
                        .map_err(|e| e.to_string());
                }
                MemoDb::open_default()
                    .map(MemoBackend::Sqlite)
                    .map_err(|e| e.to_string())
            }
        }
    }

    pub fn insert(&self, title: &str, tags: &[String]) -> Result<String, String> {
        match self {
            MemoBackend::Sqlite(db) => db.insert(title, tags).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.insert(title, tags),
        }
    }

    pub fn get_record(&self, id: &str) -> Result<Option<MemoRecord>, String> {
        match self {
            MemoBackend::Sqlite(db) => db.get_record(id).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.get_record(id),
        }
    }

    pub fn upsert_record(&self, record: &MemoRecord) -> Result<(), String> {
        match self {
            MemoBackend::Sqlite(db) => db.upsert_record(record).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.upsert_record(record),
        }
    }

    pub fn update_title(&self, id: &str, title: &str) -> Result<bool, String> {
        match self {
            MemoBackend::Sqlite(db) => db.update_title(id, title).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.update_title(id, title),
        }
    }

    pub fn delete(&self, id: &str) -> Result<bool, String> {
        match self {
            MemoBackend::Sqlite(db) => db.delete(id).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.delete(id),
        }
    }

    pub fn set_finished(&self, id: &str, finished: bool) -> Result<bool, String> {
        match self {
            MemoBackend::Sqlite(db) => db.set_finished(id, finished).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.set_finished(id, finished),
        }
    }

    pub fn add_tags(&self, id: &str, tags: &[String]) -> Result<bool, String> {
        match self {
            MemoBackend::Sqlite(db) => db.add_tags(id, tags).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.add_tags(id, tags),
        }
    }

    pub fn remove_tags(&self, id: &str, tags: &[String]) -> Result<bool, String> {
        match self {
            MemoBackend::Sqlite(db) => db.remove_tags(id, tags).map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.remove_tags(id, tags),
        }
    }

    pub fn list_records(
        &self,
        limit: i64,
        reverse: bool,
        include_finished: bool,
    ) -> Result<Vec<MemoRecord>, String> {
        match self {
            MemoBackend::Sqlite(db) => db
                .list_records(limit, reverse, include_finished)
                .map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.list_records(limit, reverse, include_finished),
        }
    }

    pub fn search_title(
        &self,
        query: &str,
        limit: i64,
        include_finished: bool,
    ) -> Result<Vec<MemoRecord>, String> {
        match self {
            MemoBackend::Sqlite(db) => db
                .search_title(query, limit, include_finished)
                .map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.search_title(query, limit, include_finished),
        }
    }

    pub fn list_tags(
        &self,
        prefix: Option<&str>,
        exclude_prefix: Option<&str>,
        limit: i64,
    ) -> Result<Vec<MemoTag>, String> {
        match self {
            MemoBackend::Sqlite(db) => db
                .list_tags(prefix, exclude_prefix, limit)
                .map_err(|e| e.to_string()),
            MemoBackend::Mongo(db) => db.list_tags(prefix, exclude_prefix, limit),
        }
    }

    pub fn list_records_by_tags(
        &self,
        tags: &[String],
        use_and: bool,
        prefix: bool,
        limit: i64,
        reverse: bool,
        include_finished: bool,
    ) -> Result<Vec<MemoRecord>, String> {
        if tags.is_empty() {
            return self.list_records(limit, reverse, include_finished);
        }
        let mut records = self.list_records(-1, false, include_finished)?;
        records.retain(|r| match_tags(&r.tags, tags, use_and, prefix));
        records.sort_by(|a, b| a.modified_date.cmp(&b.modified_date));
        if !reverse {
            records.reverse();
        }
        if limit > 0 && records.len() > limit as usize {
            records.truncate(limit as usize);
        }
        Ok(records)
    }
}

fn match_tags(record_tags: &[String], filter_tags: &[String], use_and: bool, prefix: bool) -> bool {
    if filter_tags.is_empty() {
        return true;
    }

    let matches_one = |needle: &str| {
        if prefix {
            record_tags.iter().any(|t| t.starts_with(needle))
        } else {
            record_tags.iter().any(|t| t == needle)
        }
    };

    if use_and {
        filter_tags.iter().all(|t| matches_one(t))
    } else {
        filter_tags.iter().any(|t| matches_one(t))
    }
}
