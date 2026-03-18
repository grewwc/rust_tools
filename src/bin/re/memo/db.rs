use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};

use crate::{
    common::configw,
    common::utils::expanduser,
    memo::model::{MemoRecord, MemoTag},
};

const DEFAULT_LOCAL_SQLITE: &str = "~/.go_tools_memo.sqlite3";

pub struct MemoDb {
    path: PathBuf,
}

impl MemoDb {
    pub fn open_default() -> rusqlite::Result<Self> {
        let configured = configw::get_config("memo.sqlite", DEFAULT_LOCAL_SQLITE);
        let path = if configured.trim().is_empty() {
            DEFAULT_LOCAL_SQLITE
        } else {
            configured.as_str()
        };
        Self::open(expanduser(path).as_ref())
    }

    pub fn open<P: AsRef<Path>>(path: P) -> rusqlite::Result<Self> {
        let path = PathBuf::from(expanduser(path.as_ref().to_string_lossy().as_ref()).as_ref());
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let db = Self { path };
        db.ensure_schema()?;
        Ok(db)
    }

    pub fn ensure_schema(&self) -> rusqlite::Result<()> {
        let conn = self.connect()?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                add_date INTEGER NOT NULL,
                modified_date INTEGER NOT NULL,
                my_problem INTEGER NOT NULL,
                finished INTEGER NOT NULL,
                hold INTEGER NOT NULL,
                title TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS record_tags (
                record_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY(record_id, position),
                FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                count INTEGER NOT NULL,
                modified_date INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_modified_date ON records(modified_date);
            CREATE INDEX IF NOT EXISTS idx_records_add_date ON records(add_date);
            CREATE INDEX IF NOT EXISTS idx_records_finished ON records(finished);
            CREATE INDEX IF NOT EXISTS idx_record_tags_record_id ON record_tags(record_id);
            CREATE INDEX IF NOT EXISTS idx_record_tags_tag ON record_tags(tag);
        "#,
        )?;
        Ok(())
    }

    pub fn checkpoint_wal_truncate(&self) -> rusqlite::Result<()> {
        let conn = self.connect()?;
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
        Ok(())
    }

    pub fn insert(&self, title: &str, tags: &[String]) -> rusqlite::Result<String> {
        let mut conn = self.connect()?;
        let now = now_epoch_secs();
        let id = new_object_id_like();
        let tx = conn.transaction()?;
        tx.execute(
            r#"INSERT INTO records (id, add_date, modified_date, my_problem, finished, hold, title)
               VALUES (?1, ?2, ?3, 0, 0, 0, ?4)"#,
            params![id, now, now, title],
        )?;
        upsert_tags_and_record_tags(&tx, &id, tags, now)?;
        tx.commit()?;
        Ok(id)
    }

    pub fn upsert_record(&self, record: &MemoRecord) -> rusqlite::Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        let now = now_epoch_secs();
        let exists: bool = tx
            .query_row(
                "SELECT 1 FROM records WHERE id=?1",
                params![record.id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();

        if exists {
            tx.execute(
                r#"UPDATE records SET modified_date=?2, finished=?3, hold=?4, title=?5 WHERE id=?1"#,
                params![
                    record.id,
                    record.modified_date.max(now),
                    bool_num(record.finished),
                    bool_num(record.hold),
                    record.title
                ],
            )?;
            tx.execute(
                "DELETE FROM record_tags WHERE record_id=?1",
                params![record.id],
            )?;
        } else {
            tx.execute(
                r#"INSERT INTO records (id, add_date, modified_date, my_problem, finished, hold, title)
                   VALUES (?1, ?2, ?3, 0, ?4, ?5, ?6)"#,
                params![
                    record.id,
                    record.add_date.max(now),
                    record.modified_date.max(now),
                    bool_num(record.finished),
                    bool_num(record.hold),
                    record.title
                ],
            )?;
        }
        upsert_tags_and_record_tags(&tx, &record.id, &record.tags, now)?;
        tx.commit()?;
        Ok(())
    }

    pub fn update_title(&self, id: &str, title: &str) -> rusqlite::Result<bool> {
        let conn = self.connect()?;
        let now = now_epoch_secs();
        let changed = conn.execute(
            "UPDATE records SET title=?2, modified_date=?3 WHERE id=?1",
            params![id, title, now],
        )?;
        Ok(changed > 0)
    }

    pub fn delete(&self, id: &str) -> rusqlite::Result<bool> {
        let conn = self.connect()?;
        let changed = conn.execute("DELETE FROM records WHERE id=?1", params![id])?;
        Ok(changed > 0)
    }

    pub fn set_finished(&self, id: &str, finished: bool) -> rusqlite::Result<bool> {
        let conn = self.connect()?;
        let now = now_epoch_secs();
        let changed = conn.execute(
            "UPDATE records SET finished=?2, modified_date=?3 WHERE id=?1",
            params![id, bool_num(finished), now],
        )?;
        Ok(changed > 0)
    }

    pub fn add_tags(&self, id: &str, tags: &[String]) -> rusqlite::Result<bool> {
        let mut conn = self.connect()?;
        let now = now_epoch_secs();
        let tx = conn.transaction()?;
        let exists: bool = tx
            .query_row("SELECT 1 FROM records WHERE id=?1", params![id], |_| Ok(()))
            .optional()?
            .is_some();
        if !exists {
            return Ok(false);
        }
        let start_pos: i64 = tx.query_row(
            "SELECT COALESCE(MAX(position), -1) FROM record_tags WHERE record_id=?1",
            params![id],
            |row| row.get(0),
        )?;
        let mut pos = start_pos + 1;
        for tag in tags.iter().filter(|t| !t.trim().is_empty()) {
            upsert_tag(&tx, tag, now)?;
            tx.execute(
                "INSERT OR IGNORE INTO record_tags (record_id, tag, position) VALUES (?1, ?2, ?3)",
                params![id, tag, pos],
            )?;
            pos += 1;
        }
        tx.execute(
            "UPDATE records SET modified_date=?2 WHERE id=?1",
            params![id, now],
        )?;
        tx.commit()?;
        Ok(true)
    }

    pub fn remove_tags(&self, id: &str, tags: &[String]) -> rusqlite::Result<bool> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        let exists: bool = tx
            .query_row("SELECT 1 FROM records WHERE id=?1", params![id], |_| Ok(()))
            .optional()?
            .is_some();
        if !exists {
            return Ok(false);
        }
        for tag in tags.iter().filter(|t| !t.trim().is_empty()) {
            tx.execute(
                "DELETE FROM record_tags WHERE record_id=?1 AND tag=?2",
                params![id, tag],
            )?;
        }
        tx.commit()?;
        Ok(true)
    }

    pub fn get_record(&self, id: &str) -> rusqlite::Result<Option<MemoRecord>> {
        let conn = self.connect()?;
        let record: Option<(String, i64, i64, i64, i64, String)> = conn
            .query_row(
                "SELECT id, add_date, modified_date, finished, hold, title FROM records WHERE id=?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
            )
            .optional()?;
        let Some((id, add_date, modified_date, finished, hold, title)) = record else {
            return Ok(None);
        };
        let tags = query_record_tags(&conn, &id)?;
        Ok(Some(MemoRecord {
            id,
            add_date,
            modified_date,
            finished: finished != 0,
            hold: hold != 0,
            title,
            tags,
        }))
    }

    pub fn list_records(
        &self,
        limit: i64,
        reverse: bool,
        include_finished: bool,
    ) -> rusqlite::Result<Vec<MemoRecord>> {
        let conn = self.connect()?;
        let mut sql =
            String::from("SELECT id, add_date, modified_date, finished, hold, title FROM records");
        if !include_finished {
            sql.push_str(" WHERE finished=0");
        }
        sql.push_str(" ORDER BY modified_date ");
        sql.push_str(if reverse { "ASC" } else { "DESC" });
        if limit > 0 {
            sql.push_str(" LIMIT ");
            sql.push_str(&limit.to_string());
        }
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let tags = query_record_tags(&conn, &id)?;
            out.push(MemoRecord {
                id,
                add_date: row.get(1)?,
                modified_date: row.get(2)?,
                finished: row.get::<_, i64>(3)? != 0,
                hold: row.get::<_, i64>(4)? != 0,
                title: row.get(5)?,
                tags,
            });
        }
        Ok(out)
    }

    pub fn search_title(
        &self,
        query: &str,
        limit: i64,
        include_finished: bool,
    ) -> rusqlite::Result<Vec<MemoRecord>> {
        let conn = self.connect()?;
        let mut sql = String::from(
            "SELECT id, add_date, modified_date, finished, hold, title FROM records WHERE title LIKE ?1",
        );
        if !include_finished {
            sql.push_str(" AND finished=0");
        }
        sql.push_str(" ORDER BY modified_date DESC");
        if limit > 0 {
            sql.push_str(" LIMIT ");
            sql.push_str(&limit.to_string());
        }
        let like = format!("%{}%", query);
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params![like])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let tags = query_record_tags(&conn, &id)?;
            out.push(MemoRecord {
                id,
                add_date: row.get(1)?,
                modified_date: row.get(2)?,
                finished: row.get::<_, i64>(3)? != 0,
                hold: row.get::<_, i64>(4)? != 0,
                title: row.get(5)?,
                tags,
            });
        }
        Ok(out)
    }

    pub fn list_tags(
        &self,
        prefix: Option<&str>,
        exclude_prefix: Option<&str>,
        limit: i64,
    ) -> rusqlite::Result<Vec<MemoTag>> {
        let conn = self.connect()?;
        let mut sql = String::from(
            "SELECT rt.tag, COUNT(*) as cnt, MAX(r.modified_date) as mt \
             FROM record_tags rt JOIN records r ON r.id=rt.record_id \
             WHERE r.finished=0",
        );
        let mut parts = Vec::new();
        let mut values: Vec<String> = Vec::new();

        if let Some(p) = prefix.filter(|s| !s.is_empty()) {
            parts.push(format!("rt.tag LIKE ?{}", values.len() + 1));
            values.push(format!("{p}%"));
        }
        if let Some(p) = exclude_prefix.filter(|s| !s.is_empty()) {
            parts.push(format!("rt.tag NOT LIKE ?{}", values.len() + 1));
            values.push(format!("{p}%"));
        }
        if !parts.is_empty() {
            sql.push_str(" AND ");
            sql.push_str(&parts.join(" AND "));
        }

        sql.push_str(" GROUP BY rt.tag ORDER BY cnt DESC, mt DESC");
        if limit > 0 {
            sql.push_str(" LIMIT ");
            sql.push_str(&limit.to_string());
        }

        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(values.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(MemoTag {
                id: String::new(),
                name: row.get(0)?,
                count: row.get(1)?,
                modified_date: row.get(2)?,
            });
        }
        Ok(out)
    }

    #[allow(dead_code)]
    pub fn search_by_tags(
        &self,
        tags: &[String],
        use_and: bool,
        limit: i64,
        include_finished: bool,
    ) -> rusqlite::Result<Vec<MemoRecord>> {
        if tags.is_empty() {
            return self.list_records(limit, false, include_finished);
        }

        let mut records = self.list_records(-1, false, include_finished)?;
        records.retain(|record| match_tags_exact(&record.tags, tags, use_and));
        if limit > 0 && records.len() > limit as usize {
            records.truncate(limit as usize);
        }
        Ok(records)
    }

    fn connect(&self) -> rusqlite::Result<Connection> {
        let conn = Connection::open(&self.path)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(conn)
    }
}

#[allow(dead_code)]
fn match_tags_exact(record_tags: &[String], filter_tags: &[String], use_and: bool) -> bool {
    if filter_tags.is_empty() {
        return true;
    }

    let matches_one = |needle: &str| record_tags.iter().any(|tag| tag == needle);
    if use_and {
        filter_tags.iter().all(|tag| matches_one(tag))
    } else {
        filter_tags.iter().any(|tag| matches_one(tag))
    }
}

fn upsert_tags_and_record_tags(
    tx: &rusqlite::Transaction<'_>,
    record_id: &str,
    tags: &[String],
    now: i64,
) -> rusqlite::Result<()> {
    for (pos, tag) in tags.iter().filter(|t| !t.trim().is_empty()).enumerate() {
        upsert_tag(tx, tag, now)?;
        tx.execute(
            "INSERT OR REPLACE INTO record_tags (record_id, tag, position) VALUES (?1, ?2, ?3)",
            params![record_id, tag, pos as i64],
        )?;
    }
    Ok(())
}

fn upsert_tag(tx: &rusqlite::Transaction<'_>, name: &str, now: i64) -> rusqlite::Result<()> {
    let existing: Option<(String, i64)> = tx
        .query_row(
            "SELECT id, count FROM tags WHERE name=?1",
            params![name],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((id, count)) = existing {
        tx.execute(
            "UPDATE tags SET count=?2, modified_date=?3 WHERE id=?1",
            params![id, count, now],
        )?;
        return Ok(());
    }
    let id = new_object_id_like();
    tx.execute(
        "INSERT INTO tags (id, name, count, modified_date) VALUES (?1, ?2, 0, ?3)",
        params![id, name, now],
    )?;
    Ok(())
}

fn query_record_tags(conn: &Connection, record_id: &str) -> rusqlite::Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT tag FROM record_tags WHERE record_id=?1 ORDER BY position ASC")?;
    let mut rows = stmt.query(params![record_id])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(row.get(0)?);
    }
    Ok(out)
}

fn bool_num(v: bool) -> i64 {
    if v { 1 } else { 0 }
}

fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn new_object_id_like() -> String {
    let u = uuid::Uuid::new_v4();
    let bytes = u.as_bytes();
    let mut out = String::with_capacity(24);
    for b in &bytes[..12] {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> MemoDb {
        let dir =
            std::env::temp_dir().join(format!("rust_tools_memo_test_{}", uuid::Uuid::new_v4()));
        let path = dir.join("memo.sqlite3");
        MemoDb::open(path).unwrap()
    }

    #[test]
    fn test_insert_and_get() {
        let db = temp_db();
        let id = db
            .insert("hello", &vec!["a".to_string(), "b".to_string()])
            .unwrap();
        let r = db.get_record(&id).unwrap().unwrap();
        assert_eq!(r.title, "hello");
        assert_eq!(r.tags, vec!["a", "b"]);
    }

    #[test]
    fn test_update_title_and_finish() {
        let db = temp_db();
        let id = db.insert("hello", &vec![]).unwrap();
        assert!(db.update_title(&id, "world").unwrap());
        assert!(db.set_finished(&id, true).unwrap());
        let r = db.get_record(&id).unwrap().unwrap();
        assert_eq!(r.title, "world");
        assert!(r.finished);
    }

    #[test]
    fn test_add_remove_tags_and_search() {
        let db = temp_db();
        let id = db.insert("alpha", &vec![]).unwrap();
        db.add_tags(&id, &vec!["x".to_string(), "y".to_string()])
            .unwrap();
        let found = db
            .search_by_tags(&vec!["x".to_string()], false, 10, true)
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, id);

        db.remove_tags(&id, &vec!["x".to_string()]).unwrap();
        let found = db
            .search_by_tags(&vec!["x".to_string()], false, 10, true)
            .unwrap();
        assert!(found.is_empty());
    }
}
