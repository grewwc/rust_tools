//! Persist sidebar groups separately from session history and runtime state.

use std::path::{Path as FsPath, PathBuf};

use crate::commonw::FastMap;

use super::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct SessionGroups {
    pub(super) groups: Vec<SessionGroup>,
    pub(super) assignments: FastMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct SessionGroup {
    id: String,
    name: String,
    order: f64,
    pinned: bool,
    #[serde(default)]
    hidden: bool,
}

impl SessionGroups {
    fn contains_group(&self, id: &str) -> bool {
        self.groups.iter().any(|group| group.id == id)
    }

    fn remove_group(&mut self, id: &str) -> bool {
        let previous_len = self.groups.len();
        self.groups.retain(|group| group.id != id);
        self.assignments.retain(|_, group_id| group_id != id);
        previous_len != self.groups.len()
    }

    pub(super) fn group_id(&self, id: &str) -> Option<String> {
        self.assignments
            .get(id)
            .filter(|group_id| self.contains_group(group_id))
            .cloned()
    }

    pub(super) fn matches_group(&self, id: &str, filter: Option<&str>) -> bool {
        match filter {
            None => true,
            Some("unassigned") => self.group_id(id).is_none(),
            Some(group) => {
                self.contains_group(group)
                    && self
                        .assignments
                        .get(id)
                        .is_some_and(|assigned| assigned == group)
            }
        }
    }
}

fn groups_file(history_file: &FsPath) -> PathBuf {
    history_file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(FsPath::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("serve-groups.json")
}

/// A missing or damaged sidecar never prevents history from loading.
pub(super) fn load_groups(history_file: &FsPath) -> SessionGroups {
    std::fs::read(groups_file(history_file))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_groups(history_file: &FsPath, groups: &SessionGroups) -> std::io::Result<()> {
    let path = groups_file(history_file);
    let bytes = serde_json::to_vec(groups).map_err(std::io::Error::other)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4()));
    let result = std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, &path));
    if result.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    result
}

pub(super) fn groups_snapshot(state: &ServeState) -> SessionGroups {
    state
        .groups
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Keep only the group lock across publication. No session/history lock is
/// acquired here, and failed writes leave the in-memory state unchanged.
fn update_groups<T>(
    state: &ServeState,
    update: impl FnOnce(&mut SessionGroups) -> Result<T, Response>,
) -> Result<T, Response> {
    let mut guard = state.groups.lock().unwrap_or_else(|e| e.into_inner());
    let mut next = guard.clone();
    let result = update(&mut next)?;
    save_groups(&state.history_file, &next).map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response()
    })?;
    *guard = next;
    Ok(result)
}

fn group_not_found(id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": format!("group not found: {id}")})),
    )
        .into_response()
}

fn validated_name(name: &str) -> Result<String, Response> {
    let name = name.trim();
    if !(1..=40).contains(&name.chars().count()) {
        return Err(
            bad_request("group name must contain 1 to 40 characters".into()).into_response(),
        );
    }
    Ok(name.to_string())
}

#[derive(Deserialize)]
pub(super) struct CreateGroupReq {
    name: String,
}

#[derive(Deserialize)]
pub(super) struct UpdateGroupReq {
    name: Option<String>,
    pinned: Option<bool>,
    hidden: Option<bool>,
    order: Option<f64>,
}

#[derive(Default, Deserialize)]
pub(super) struct SetSessionGroupReq {
    pub(super) group_id: Option<String>,
}

pub(super) async fn list_session_groups(
    State(state): State<ServeState>,
    headers: HeaderMap,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    Json(groups_snapshot(&state)).into_response()
}

/// Sample all hidden assignments independently of session paging and metadata.
pub(super) async fn list_session_group_activity(
    State(state): State<ServeState>,
    headers: HeaderMap,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    // Release the organization lock before any live-state or file-lease probe.
    let snapshot = groups_snapshot(&state);
    let result = tokio::task::spawn_blocking(move || {
        let mut groups = Vec::new();
        let mut indices = FastMap::default();
        for group in snapshot.groups.into_iter().filter(|group| group.hidden) {
            if indices.contains_key(&group.id) {
                continue;
            }
            indices.insert(group.id.clone(), groups.len());
            groups.push(SessionGroupActivity {
                group_id: group.id,
                running: 0,
                waiting: 0,
                unknown: 0,
                active: Vec::new(),
            });
        }
        let store = SessionStore::new(&state.history_file);
        // Assignments are keyed by session ID, so no candidate can count twice.
        for (id, group_id) in snapshot.assignments {
            let Some(&index) = indices.get(&group_id) else {
                continue;
            };
            if SessionStore::validate_session_id(&id).is_err() {
                continue;
            }
            let activity = session_activity(&state, &store, &id);
            let waiting = session_waiting(&state, &id);
            let group = &mut groups[index];
            if waiting.is_some() {
                group.waiting += 1;
            } else {
                match activity {
                    SessionActivity::Running => group.running += 1,
                    SessionActivity::Unknown => group.unknown += 1,
                    SessionActivity::Idle => continue,
                }
            }
            if waiting.is_some() || activity == SessionActivity::Running {
                group.active.push(SessionGroupActiveItem {
                    id,
                    activity,
                    waiting: waiting.as_ref().map(|(kind, _)| *kind),
                    waiting_prompt: waiting.map(|(_, prompt)| prompt),
                });
            }
        }
        for group in &mut groups {
            group.active.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        }
        SessionGroupActivityResponse { groups }
    })
    .await;
    let mut response = match result {
        Ok(activity) => Json(activity).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    };
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-store"));
    response
}

pub(super) async fn create_session_group(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Json(req): Json<CreateGroupReq>,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    let name = match validated_name(&req.name) {
        Ok(name) => name,
        Err(err) => return err,
    };
    match update_groups(&state, |data| {
        let group = SessionGroup {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            order: data
                .groups
                .iter()
                .map(|group| group.order)
                .reduce(f64::max)
                .unwrap_or(0.0)
                + 1.0,
            pinned: false,
            hidden: false,
        };
        data.groups.push(group.clone());
        Ok(group)
    }) {
        Ok(group) => Json(group).into_response(),
        Err(err) => err,
    }
}

pub(super) async fn update_session_group(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<UpdateGroupReq>,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    match update_groups(&state, |data| {
        let group = data
            .groups
            .iter_mut()
            .find(|group| group.id == id)
            .ok_or_else(|| group_not_found(&id))?;
        if let Some(name) = req.name {
            group.name = validated_name(&name)?;
        }
        if let Some(pinned) = req.pinned {
            group.pinned = pinned;
        }
        if let Some(hidden) = req.hidden {
            group.hidden = hidden;
        }
        if let Some(order) = req.order {
            group.order = order;
        }
        Ok(group.clone())
    }) {
        Ok(group) => Json(group).into_response(),
        Err(err) => err,
    }
}

pub(super) async fn delete_session_group(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    match update_groups(&state, |data| {
        if !data.remove_group(&id) {
            return Err(group_not_found(&id));
        }
        Ok(())
    }) {
        Ok(()) => Json(serde_json::json!({"ok": true})).into_response(),
        Err(err) => err,
    }
}

pub(super) fn assign_session_group(
    state: &ServeState,
    id: &str,
    group_id: Option<&str>,
) -> Result<(), Response> {
    update_groups(state, |data| {
        if let Some(group_id) = group_id {
            if !data.contains_group(group_id) {
                return Err(group_not_found(group_id));
            }
            data.assignments
                .insert(id.to_string(), group_id.to_string());
        } else {
            data.assignments.remove(id);
        }
        Ok(())
    })
}

pub(super) async fn set_session_group(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<SetSessionGroupReq>,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    match assign_session_group(&state, &id, req.group_id.as_deref()) {
        Ok(()) => Json(serde_json::json!({"id": id, "group_id": req.group_id})).into_response(),
        Err(err) => err,
    }
}

/// Lifecycle changes already committed to history must still succeed if the
/// organizational sidecar cannot be saved. Log the persistence failure.
pub(super) fn drop_session_group(state: &ServeState, id: &str) {
    let mut data = state.groups.lock().unwrap_or_else(|e| e.into_inner());
    if data.assignments.remove(id).is_some()
        && let Err(err) = save_groups(&state.history_file, &data)
    {
        eprintln!("[serve] warning: failed to clear session group: {err}");
    }
}

pub(super) fn copy_session_group(state: &ServeState, src: &str, dst: &str) -> Option<String> {
    let mut data = state.groups.lock().unwrap_or_else(|e| e.into_inner());
    let group_id = data.group_id(src);
    let changed = if let Some(group_id) = &group_id {
        data.assignments.insert(dst.to_string(), group_id.clone());
        true
    } else {
        data.assignments.remove(dst).is_some()
    };
    if changed && let Err(err) = save_groups(&state.history_file, &data) {
        eprintln!("[serve] warning: failed to inherit session group: {err}");
    }
    group_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_groups_sidecar_roundtrip_and_corruption() {
        let root = std::env::temp_dir().join(format!("a-serve-groups-{}", uuid::Uuid::new_v4()));
        let history_file = root.join("history.sqlite");
        assert_eq!(load_groups(&history_file), SessionGroups::default());
        let mut data = SessionGroups::default();
        data.groups.push(SessionGroup {
            id: "topic".into(),
            name: "Topic".into(),
            order: 2.5,
            pinned: true,
            hidden: true,
        });
        data.assignments
            .insert("lazy-session".into(), "topic".into());
        save_groups(&history_file, &data).unwrap();
        assert_eq!(load_groups(&history_file), data);
        std::fs::write(groups_file(&history_file), b"{broken").unwrap();
        assert_eq!(load_groups(&history_file), SessionGroups::default());
        save_groups(&history_file, &data).unwrap();
        assert_eq!(load_groups(&history_file), data);
        assert_eq!(
            std::fs::read_dir(&root).unwrap().count(),
            1,
            "temporary files must not survive publication"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_groups_hidden_legacy_sidecar_defaults_to_visible() {
        let root = std::env::temp_dir().join(format!("a-serve-groups-{}", uuid::Uuid::new_v4()));
        let history_file = root.join("history.sqlite");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            groups_file(&history_file),
            br#"{"groups":[{"id":"topic","name":"Topic","order":1.0,"pinned":true}],"assignments":{"session":"topic"}}"#,
        )
        .unwrap();
        let data = load_groups(&history_file);
        assert_eq!(data.groups.len(), 1, "legacy groups must survive loading");
        assert!(!data.groups[0].hidden);
        assert!(data.groups[0].pinned);
        assert_eq!(data.group_id("session").as_deref(), Some("topic"));
        assert_eq!(
            serde_json::to_value(&data).unwrap()["groups"][0]["hidden"],
            false
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_groups_delete_cleans_assignments() {
        let mut data = SessionGroups::default();
        for id in ["keep", "delete"] {
            data.groups.push(SessionGroup {
                id: id.into(),
                name: id.into(),
                order: 1.0,
                pinned: false,
                hidden: false,
            });
        }
        for (id, group) in [("a", "delete"), ("b", "delete"), ("c", "keep")] {
            data.assignments.insert(id.into(), group.into());
        }
        assert!(data.remove_group("delete"));
        assert!(!data.remove_group("missing"));
        assert_eq!(data.assignments.len(), 1);
        assert_eq!(data.group_id("c").as_deref(), Some("keep"));
        assert!(data.matches_group("a", Some("unassigned")));
        assert!(!data.matches_group("a", Some("delete")));
        assert!(!data.matches_group("c", Some("missing")));
    }
}
