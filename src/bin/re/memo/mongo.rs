use std::time::Duration;

use futures_util::TryStreamExt;
use mongodb::{
    Client, Collection,
    bson::{self, DateTime, doc, oid::ObjectId},
    options::{ClientOptions, FindOptions, UpdateOptions},
};

use crate::{
    common::configw,
    memo::model::{MemoRecord, MemoTag},
};

const DB_NAME: &str = "daily";
const RECORDS_COLLECTION: &str = "memo";
const TAGS_COLLECTION: &str = "tag";
const DEFAULT_LOCAL_MONGO_URI: &str = "mongodb://localhost:27017";
const DEFAULT_MONGO_PORT: u16 = 27017;

pub struct MemoMongo {
    rt: tokio::runtime::Runtime,
    records: Collection<bson::Document>,
    tags: Collection<bson::Document>,
}

impl MemoMongo {
    pub fn connect_default() -> Result<Self, String> {
        let uri = configw::get_config("mongo.local", DEFAULT_LOCAL_MONGO_URI);
        Self::connect(&uri)
    }

    pub fn connect_remote_host(host: &str) -> Result<Self, String> {
        let uri = normalize_mongo_uri(host);
        Self::connect(&uri)
    }

    pub fn connect(uri: &str) -> Result<Self, String> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|e| e.to_string())?;

        let (records, tags) = rt.block_on(async {
            let mut opts = ClientOptions::parse(uri).await.map_err(|e| e.to_string())?;
            opts.connect_timeout = Some(Duration::from_secs(2));
            opts.server_selection_timeout = Some(Duration::from_secs(2));
            let client = Client::with_options(opts).map_err(|e| e.to_string())?;
            client
                .database(DB_NAME)
                .run_command(doc! {"ping": 1})
                .await
                .map_err(|e| e.to_string())?;
            Ok::<_, String>((
                client.database(DB_NAME).collection(RECORDS_COLLECTION),
                client.database(DB_NAME).collection(TAGS_COLLECTION),
            ))
        })?;

        Ok(Self { rt, records, tags })
    }

    pub fn insert(&self, title: &str, tags: &[String]) -> Result<String, String> {
        let now = DateTime::now();
        let tags = normalize_tags(tags);
        let id = ObjectId::new();
        let doc = doc! {
            "_id": id,
            "tags": tags.clone(),
            "add_date": now,
            "modified_date": now,
            "my_problem": true,
            "finished": false,
            "hold": false,
            "title": title,
        };

        self.rt.block_on(async {
            self.records
                .insert_one(doc)
                .await
                .map_err(|e| e.to_string())
        })?;
        self.increment_tag_counts(&tags, 1)?;
        Ok(id.to_hex())
    }

    pub fn get_record(&self, id: &str) -> Result<Option<MemoRecord>, String> {
        let Some(oid) = parse_object_id(id) else {
            return Ok(None);
        };
        let doc_opt = self.rt.block_on(async {
            self.records
                .find_one(doc! {"_id": oid})
                .await
                .map_err(|e| e.to_string())
        })?;
        Ok(doc_opt.and_then(doc_to_record))
    }

    pub fn upsert_record(&self, record: &MemoRecord) -> Result<(), String> {
        let Some(oid) = parse_object_id(&record.id) else {
            return Err("invalid record id".to_string());
        };
        let doc = record_to_doc(oid, record);
        self.rt.block_on(async {
            self.records
                .replace_one(doc! {"_id": oid}, doc)
                .upsert(true)
                .await
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(())
        })?;
        Ok(())
    }

    pub fn update_title(&self, id: &str, title: &str) -> Result<bool, String> {
        let Some(oid) = parse_object_id(id) else {
            return Ok(false);
        };
        let now = DateTime::now();
        let matched = self.rt.block_on(async {
            let res = self
                .records
                .update_one(
                    doc! {"_id": oid},
                    doc! {"$set": {"title": title, "modified_date": now}},
                )
                .await
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(res.matched_count > 0)
        })?;
        Ok(matched)
    }

    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let Some(oid) = parse_object_id(id) else {
            return Ok(false);
        };
        let record = self.get_record(id)?;
        let tags = record.as_ref().map(|r| r.tags.clone()).unwrap_or_default();
        let deleted = self.rt.block_on(async {
            let res = self
                .records
                .delete_one(doc! {"_id": oid})
                .await
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(res.deleted_count > 0)
        })?;
        if deleted && !tags.is_empty() {
            self.increment_tag_counts(&tags, -1)?;
        }
        Ok(deleted)
    }

    pub fn set_finished(&self, id: &str, finished: bool) -> Result<bool, String> {
        let Some(oid) = parse_object_id(id) else {
            return Ok(false);
        };
        let before = self.get_record(id)?;
        let old_finished = before.as_ref().map(|r| r.finished).unwrap_or(false);
        let tags = before.as_ref().map(|r| r.tags.clone()).unwrap_or_default();
        let now = DateTime::now();
        let matched = self.rt.block_on(async {
            let res = self
                .records
                .update_one(
                    doc! {"_id": oid},
                    doc! {"$set": {"finished": finished, "modified_date": now}},
                )
                .await
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(res.matched_count > 0)
        })?;
        if matched && old_finished != finished && !tags.is_empty() {
            let inc = if finished { -1 } else { 1 };
            self.increment_tag_counts(&tags, inc)?;
        }
        Ok(matched)
    }

    pub fn add_tags(&self, id: &str, tags: &[String]) -> Result<bool, String> {
        let Some(mut record) = self.get_record(id)? else {
            return Ok(false);
        };
        let add = normalize_tags(tags);
        if add.is_empty() {
            return Ok(true);
        }
        let mut new_tags = Vec::new();
        for t in add {
            if !record.tags.contains(&t) {
                record.tags.push(t.clone());
                new_tags.push(t);
            }
        }
        if new_tags.is_empty() {
            return Ok(true);
        }
        let ok = self.upsert_record(&record).is_ok();
        if ok {
            self.increment_tag_counts(&new_tags, 1)?;
        }
        Ok(ok)
    }

    pub fn remove_tags(&self, id: &str, tags: &[String]) -> Result<bool, String> {
        let Some(mut record) = self.get_record(id)? else {
            return Ok(false);
        };
        let remove = normalize_tags(tags);
        if remove.is_empty() {
            return Ok(true);
        }
        let mut removed = Vec::new();
        record.tags.retain(|t| {
            if remove.contains(t) {
                removed.push(t.clone());
                false
            } else {
                true
            }
        });
        if removed.is_empty() {
            return Ok(true);
        }
        let ok = self.upsert_record(&record).is_ok();
        if ok {
            self.increment_tag_counts(&removed, -1)?;
        }
        Ok(ok)
    }

    pub fn list_records(
        &self,
        limit: i64,
        reverse: bool,
        include_finished: bool,
    ) -> Result<Vec<MemoRecord>, String> {
        let mut filter = doc! {};
        if !include_finished {
            filter.insert("finished", false);
        }
        let sort = doc! {"modified_date": if reverse { 1 } else { -1 }};
        let opts = FindOptions::builder()
            .sort(sort)
            .limit(if limit > 0 { Some(limit) } else { None })
            .build();

        self.rt.block_on(async {
            let mut cursor = self
                .records
                .find(filter)
                .with_options(opts)
                .await
                .map_err(|e| e.to_string())?;
            let mut out = Vec::new();
            while let Some(doc) = cursor.try_next().await.map_err(|e| e.to_string())? {
                if let Some(r) = doc_to_record(doc) {
                    out.push(r);
                }
            }
            Ok(out)
        })
    }

    pub fn search_title(
        &self,
        query: &str,
        limit: i64,
        include_finished: bool,
    ) -> Result<Vec<MemoRecord>, String> {
        let mut filter = doc! {"title": {"$regex": bson::Regex{ pattern: regex::escape(query), options: String::new() }}};
        if !include_finished {
            filter.insert("finished", false);
        }
        let opts = FindOptions::builder()
            .sort(doc! {"modified_date": -1})
            .limit(if limit > 0 { Some(limit) } else { None })
            .build();
        self.rt.block_on(async {
            let mut cursor = self
                .records
                .find(filter)
                .with_options(opts)
                .await
                .map_err(|e| e.to_string())?;
            let mut out = Vec::new();
            while let Some(doc) = cursor.try_next().await.map_err(|e| e.to_string())? {
                if let Some(r) = doc_to_record(doc) {
                    out.push(r);
                }
            }
            Ok(out)
        })
    }

    pub fn list_tags(
        &self,
        prefix: Option<&str>,
        exclude_prefix: Option<&str>,
        limit: i64,
    ) -> Result<Vec<MemoTag>, String> {
        let mut filter = doc! {};
        if let Some(p) = prefix.filter(|s| !s.trim().is_empty()) {
            filter.insert("name", doc! {"$regex": bson::Regex{ pattern: format!("^{}", regex::escape(p)), options: String::new() }});
        }
        if let Some(p) = exclude_prefix.filter(|s| !s.trim().is_empty()) {
            filter.insert("name", doc! {"$not": bson::Regex{ pattern: format!("^{}", regex::escape(p)), options: String::new() }});
        }
        let opts = FindOptions::builder()
            .sort(doc! {"count": -1, "modified_date": -1})
            .limit(if limit > 0 { Some(limit) } else { None })
            .build();
        self.rt.block_on(async {
            let mut cursor = self
                .tags
                .find(filter)
                .with_options(opts)
                .await
                .map_err(|e| e.to_string())?;
            let mut out = Vec::new();
            while let Some(doc) = cursor.try_next().await.map_err(|e| e.to_string())? {
                if let Some(t) = doc_to_tag(doc) {
                    out.push(t);
                }
            }
            Ok(out)
        })
    }

    fn increment_tag_counts(&self, tags: &[String], delta: i64) -> Result<(), String> {
        let now = DateTime::now();
        self.rt.block_on(async {
            for tag in tags.iter().filter(|t| !t.trim().is_empty()) {
                let update = doc! {
                    "$setOnInsert": { "_id": ObjectId::new(), "name": tag, "count": 0_i64, "modified_date": now },
                    "$inc": { "count": delta },
                    "$set": { "modified_date": now }
                };
                self.tags
                    .update_one(
                        doc! {"name": tag},
                        update,
                    )
                    .with_options(UpdateOptions::builder().upsert(true).build())
                    .await
                    .map_err(|e| e.to_string())?;
            }
            Ok::<_, String>(())
        })?;
        Ok(())
    }
}

pub fn normalize_mongo_uri(host: &str) -> String {
    let h = host.trim();
    if h.starts_with("mongodb://") || h.starts_with("mongodb+srv://") {
        return h.to_string();
    }
    let mut host = h.to_string();
    if let Some(idx) = host.rfind('@') {
        host = host[idx + 1..].to_string();
    }
    if host.is_empty() {
        return DEFAULT_LOCAL_MONGO_URI.to_string();
    }
    if host.contains("://") {
        return host;
    }
    if host.contains(':') {
        return format!("mongodb://{host}");
    }
    format!("mongodb://{host}:{DEFAULT_MONGO_PORT}")
}

fn parse_object_id(s: &str) -> Option<ObjectId> {
    ObjectId::parse_str(s).ok()
}

fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut out = tags
        .iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>();
    if out.is_empty() {
        out.push("auto".to_string());
    }
    out.sort();
    out.dedup();
    out
}

fn doc_to_record(doc: bson::Document) -> Option<MemoRecord> {
    let id = doc.get_object_id("_id").ok()?.to_hex();
    let title = doc.get_str("title").ok().unwrap_or_default().to_string();
    let tags = doc
        .get_array("tags")
        .ok()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let add_date = doc.get_datetime("add_date").ok().map(to_epoch).unwrap_or(0);
    let modified_date = doc
        .get_datetime("modified_date")
        .ok()
        .map(to_epoch)
        .unwrap_or(0);
    let finished = doc.get_bool("finished").ok().unwrap_or(false);
    let hold = doc.get_bool("hold").ok().unwrap_or(false);
    Some(MemoRecord {
        id,
        add_date,
        modified_date,
        finished,
        hold,
        title,
        tags,
    })
}

fn record_to_doc(oid: ObjectId, record: &MemoRecord) -> bson::Document {
    doc! {
        "_id": oid,
        "tags": record.tags.clone(),
        "add_date": from_epoch(record.add_date),
        "modified_date": from_epoch(record.modified_date),
        "my_problem": true,
        "finished": record.finished,
        "hold": record.hold,
        "title": record.title.clone(),
    }
}

fn to_epoch(dt: &DateTime) -> i64 {
    dt.timestamp_millis() / 1000
}

fn from_epoch(secs: i64) -> DateTime {
    DateTime::from_millis(secs.saturating_mul(1000))
}

fn doc_to_tag(doc: bson::Document) -> Option<MemoTag> {
    let id = doc.get_object_id("_id").ok()?.to_hex();
    let name = doc.get_str("name").ok()?.to_string();
    let count = doc.get_i64("count").ok().unwrap_or(0);
    let modified_date = doc
        .get_datetime("modified_date")
        .ok()
        .map(to_epoch)
        .unwrap_or(0);
    Some(MemoTag {
        id,
        name,
        count,
        modified_date,
    })
}
