"use strict";
const $ = (id) => document.getElementById(id);
const form = $("filters");
let view = "events",
  cursor = null,
  history = [],
  next = null,
  requestSerial = 0,
  detailSerial = 0,
  bodyChunk = null,
  bodyKind = "",
  selectedEvent = null;
const labels = {
  TASK_STATUS_CHANGED: "状态变更",
  EXTERNAL_REQUEST: "外部请求",
  EXTERNAL_RESPONSE: "外部响应",
  EXTERNAL_ERROR: "调用异常",
  EXTERNAL_STREAM_EVENT: "流事件",
  EXTERNAL_STREAM_END: "流结束",
  BUSINESS_EVENT: "业务动作",
  LOG: "应用日志",
  TASK_CONTEXT: "关联任务",
};
const time = (value) =>
  value
    ? new Intl.DateTimeFormat("zh-CN", {
        timeZone: "Asia/Shanghai",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        fractionalSecondDigits: 3,
        hour12: false,
      }).format(new Date(value))
    : "—";
function node(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function link(text, url) {
  try {
    const u = new URL(url);
    if (
      u.protocol !== "https:" ||
      u.hostname !== "code.byted.org" ||
      u.username ||
      u.password
    )
      return node("span", "—");
    const a = node("a", text);
    a.href = u.href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  } catch {
    return node("span", "—");
  }
}
function sessionLink(id, label = "Session") {
  return id && /^\d+$/.test(id)
    ? link(
        label,
        "https://code.byted.org/managed-agents/conciliator-service/session/" +
          id,
      )
    : node("span", "—");
}
function badge(value) {
  let cls = "";
  if (/APPROVED|SUCCESS/.test(value || "")) cls = "success";
  if (/FAILED|ERROR|REJECTED/.test(value || "")) cls = "error";
  if (/RUNNING|PREPAR|PENDING/.test(value || "")) cls = "running";
  return node("span", value || "—", "badge " + cls);
}
async function api(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    let message = "HTTP " + res.status;
    try {
      const data = await res.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail);
    } catch {}
    throw new Error(message);
  }
  return res.json();
}
function params() {
  const p = new URLSearchParams();
  for (const [k, v] of new FormData(form)) {
    if (!v) continue;
    p.set(k, k === "start" || k === "end" ? new Date(v).toISOString() : v);
  }
  return p;
}
function saveURL() {
  const p = params();
  if (view === "tasks") p.set("view", "tasks");
  window.history.replaceState(null, "", "/trace" + (p.size ? "?" + p : ""));
}
function readURL() {
  const p = new URLSearchParams(location.search);
  view = p.get("view") === "tasks" ? "tasks" : "events";
  if (p.has("task_id") && !p.has("order")) form.elements.order.value = "asc";
  for (const [k, v] of p) {
    const input = form.elements.namedItem(k);
    if (!input) continue;
    if (k === "start" || k === "end") {
      const d = new Date(v);
      if (!Number.isNaN(d.valueOf()))
        input.value = new Date(d.getTime() - d.getTimezoneOffset() * 60000)
          .toISOString()
          .slice(0, 19);
    } else input.value = v;
  }
}
function taskButton(task) {
  const b = node("button", "#" + task.id, "task-id");
  b.onclick = () => {
    form.elements.task_id.value = task.id;
    form.elements.order.value = "asc";
    view = "events";
    resetPage();
    load();
  };
  return b;
}
function taskLinks(task) {
  const d = node("div", undefined, "links");
  d.append(link("原 MR", task.original_mr_url));
  if (task.new_mr_url) d.append(link("新 MR", task.new_mr_url));
  return d;
}
function cell(row, child, cls) {
  const c = node("td", undefined, cls);
  c.append(child);
  row.append(c);
  return c;
}
function render(data) {
  $("rows").replaceChildren();
  $("thead").replaceChildren();
  const head = node("tr");
  const titles =
    view === "tasks"
      ? [
          "任务 ID",
          "创建时间",
          "当前状态",
          "仓库 / 分支",
          "MR 链接（当前）",
          "Coco Session",
        ]
      : [
          "时间",
          "任务 ID",
          "当前状态",
          "记录时状态",
          "类型 / 操作",
          "摘要",
          "MR 链接（当前）",
          "Coco Session",
          "明细",
        ];
  for (const t of titles) head.append(node("th", t));
  $("thead").append(head);
  for (const item of data.items) {
    const row = node("tr");
    if (view === "tasks") {
      cell(row, taskButton(item));
      cell(row, node("span", time(item.created_at)));
      cell(row, badge(item.status));
      const repo = node("div", item.repo_id);
      repo.append(
        node(
          "span",
          item.source_branch + " → " + item.target_branch,
          "secondary",
        ),
      );
      cell(row, repo);
      cell(row, taskLinks(item));
      cell(row, sessionLink(item.coco_session_id));
    } else {
      const task = item.tasks[0];
      cell(row, node("span", time(item.occurred_at)));
      const ids = node("div");
      for (const t of item.tasks) ids.append(taskButton(t));
      cell(row, ids);
      cell(row, badge(task?.status));
      cell(
        row,
        item.from_status || item.to_status
          ? node("span", (item.from_status || "创建") + " → " + item.to_status)
          : badge(item.task_status),
      );
      const op = node("div");
      op.append(
        node("span", labels[item.event_type] || item.event_type, "event-type"),
        node(
          "span",
          (item.provider ? item.provider + " · " : "") +
            (item.operation || "—"),
          "operation",
        ),
      );
      cell(row, op);
      const summary = node("div", item.summary);
      if (item.duration_ms !== null)
        summary.append(
          node(
            "span",
            item.duration_ms + " ms · " + (item.outcome || ""),
            "secondary",
          ),
        );
      cell(row, summary, "summary");
      cell(row, task ? taskLinks(task) : node("span", "—"));
      const identity = item.details || {};
      const sid =
        identity.SessionId || identity.session_id || identity.coco_session_id;
      cell(
        row,
        sessionLink(
          sid || task?.coco_session_id,
          sid ? "Session" : "当前 Session",
        ),
      );
      const b = node(
        "button",
        item.body_state === "EMPTY" ? "详情" : "展开",
        "expand",
      );
      b.onclick = () => openDetail(item);
      cell(row, b);
    }
    $("rows").append(row);
  }
  $("empty").hidden = data.items.length > 0;
  $("table-wrap").hidden = !data.items.length;
  $("result-count").textContent = "本页 " + data.items.length + " 条";
  next = data.next_cursor;
  $("prev").disabled = history.length === 0;
  $("next").disabled = !next;
  $("page-number").textContent = "第 " + (history.length + 1) + " 页";
  $("updated").textContent =
    "更新于 " + new Date().toLocaleTimeString("zh-CN", { hour12: false });
}
function resetPage() {
  cursor = null;
  history = [];
  next = null;
}
async function load() {
  const serial = ++requestSerial;
  $("error").hidden = true;
  $("refresh").disabled = true;
  for (const name of [
    "provider",
    "event_type",
    "task_status",
    "operation",
    "outcome",
    "call_id",
  ]) {
    form.elements[name].disabled = view === "tasks";
  }
  const p = params();
  saveURL();
  if (cursor) p.set("cursor", cursor);
  $("events-tab").classList.toggle("active", view === "events");
  $("tasks-tab").classList.toggle("active", view === "tasks");
  const task = form.elements.task_id.value;
  $("result-title").textContent =
    view === "tasks"
      ? "任务列表"
      : task
        ? "任务 #" + task + " · 时间线"
        : "全部明细";
  $("clear-task").hidden = !task;
  try {
    const data = await api("/api/trace/" + view + "?" + p);
    if (serial === requestSerial) render(data);
  } catch (e) {
    if (serial === requestSerial) {
      $("error").textContent = "读取失败：" + e.message;
      $("error").hidden = false;
    }
  } finally {
    if (serial === requestSerial) $("refresh").disabled = false;
  }
}
async function loadHealth() {
  try {
    const data = await api("/api/trace/health");
    const active = data.instances.filter((i) => !i.stale && !i.stopping);
    const gaps = data.instances.some(
      (i) => i.dropped > 0 || i.errors > 0 || (i.stale && !i.stopping),
    );
    $("health").textContent =
      (active.length
        ? "采集进程 " + active.length + " 个在线"
        : "尚未收到采集心跳") +
      (gaps
        ? " · 存在采集异常或失联进程，部分明细可能缺失"
        : " · 状态与外部调用持续入库") +
      "。" +
      data.note;
    $("health").classList.toggle("warning", gaps || !active.length);
  } catch (e) {
    $("health").textContent = "采集状态暂不可用：" + e.message;
    $("health").classList.add("warning");
  }
}
async function openDetail(item) {
  const serial = ++detailSerial;
  selectedEvent = item.id;
  bodyChunk = null;
  $("detail-title").textContent =
    (labels[item.event_type] || item.event_type) + " · #" + item.id;
  $("body-content").value = "";
  $("body-tabs").replaceChildren();
  $("detail-meta").replaceChildren();
  $("body-status").textContent = "正在加载…";
  $("more-body").hidden = true;
  $("detail").showModal();
  try {
    const data = await api("/api/trace/events/" + item.id);
    if (serial !== detailSerial) return;
    const meta = node("div", undefined, "meta-grid");
    for (const [k, v] of Object.entries({
      发生时间: time(data.occurred_at),
      记录时状态: data.task_status || "未观察到",
      操作: data.operation || "—",
      调用结果: data.outcome || "—",
      "调用 ID": data.call_id || "—",
      "逻辑调用 ID": data.operation_id || "—",
      请求次数: data.attempt || "—",
      "Request ID": data.external_request_id || "—",
      "执行 ID": data.execution_id || "—",
    })) {
      meta.append(node("span", k), node("span", v));
    }
    $("detail-meta").append(meta);
    $("body-status").textContent =
      data.body_state === "TRUNCATED"
        ? "正文超出采集上限，已截断。原始大小 " + data.body_bytes + " 字节"
        : data.bodies.length
          ? "已脱敏 · 分段加载"
          : "此条记录不保存正文；Coco 内容请在 Session 查看。";
    for (const body of data.bodies) {
      const b = node("button", body.kind + " · " + body.chunks + " 段");
      b.onclick = () => {
        bodyKind = body.kind;
        bodyChunk = 0;
        $("body-content").value = "";
        loadBody(serial);
      };
      $("body-tabs").append(b);
    }
    if (data.bodies.length) {
      bodyKind = data.bodies[0].kind;
      bodyChunk = 0;
      await loadBody(serial);
    }
  } catch (e) {
    if (serial === detailSerial) $("body-status").textContent = e.message;
  }
}
async function loadBody(serial = detailSerial) {
  if (bodyChunk === null) return;
  $("more-body").disabled = true;
  try {
    const data = await api(
      "/api/trace/events/" +
        selectedEvent +
        "/body?kind=" +
        encodeURIComponent(bodyKind) +
        "&chunk=" +
        bodyChunk,
    );
    if (serial !== detailSerial) return;
    let content = data.content;
    if (bodyChunk === 0 && data.next_chunk === null) {
      try {
        content = JSON.stringify(JSON.parse(content), null, 2);
      } catch {}
    }
    $("body-content").value += content;
    bodyChunk = data.next_chunk;
    $("more-body").hidden = bodyChunk === null;
  } catch (e) {
    $("body-status").textContent = "读取正文失败：" + e.message;
  } finally {
    $("more-body").disabled = false;
  }
}
form.onsubmit = (e) => {
  e.preventDefault();
  resetPage();
  load();
};
$("reset").onclick = () => {
  HTMLFormElement.prototype.reset.call(form);
  resetPage();
  load();
};
$("refresh").onclick = () => {
  resetPage();
  load();
  loadHealth();
};
$("clear-task").onclick = () => {
  form.elements.task_id.value = "";
  form.elements.order.value = "desc";
  resetPage();
  load();
};
$("next").onclick = () => {
  if (!next) return;
  history.push(cursor);
  cursor = next;
  load();
};
$("prev").onclick = () => {
  cursor = history.pop() || null;
  load();
};
for (const name of ["events", "tasks"])
  $(name + "-tab").onclick = () => {
    view = name;
    resetPage();
    load();
  };
$("close-detail").onclick = () => {
  $("detail").close();
  detailSerial++;
};
$("detail").addEventListener("close", () => {
  detailSerial++;
});
$("more-body").onclick = () => loadBody();
$("body-search").onkeydown = (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  const area = $("body-content"),
    term = e.target.value;
  if (!term) return;
  let index = area.value.indexOf(term, area.selectionEnd);
  if (index < 0) index = area.value.indexOf(term);
  if (index >= 0) {
    area.focus();
    area.setSelectionRange(index, index + term.length);
  }
};
window.addEventListener("popstate", () => {
  HTMLFormElement.prototype.reset.call(form);
  readURL();
  resetPage();
  load();
});
setInterval(() => {
  if ($("auto").checked && cursor === null && !$("detail").open) {
    load();
    loadHealth();
  }
}, 15000);
(async () => {
  try {
    const data = await api("/api/trace/options");
    for (const name of ["current_status", "task_status"])
      for (const status of data.statuses) {
        const option = node("option", status);
        option.value = status;
        form.elements[name].append(option);
      }
    for (const type of data.event_types) {
      const option = node("option", labels[type] || type);
      option.value = type;
      form.elements.event_type.append(option);
    }
  } catch (e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
  }
  readURL();
  await load();
  loadHealth();
})();
