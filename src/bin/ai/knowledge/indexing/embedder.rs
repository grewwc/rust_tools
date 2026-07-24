//! 嵌入向量提供方（远程 OpenAI 兼容接口）
//!
//! 历史上这里挂的是 `fastembed::TextEmbedding` (MultilingualE5Small)，需要静态
//! 链接 ONNX Runtime + XNNPACK，占整个 binary ~25MB / 23.6%——属于体积膨胀
//! 的最大单一原因。
//!
//! 当前由 `warm_up` 根据配置安装远程 provider，默认使用火山方舟 Coding Plan 的
//! `doubao-embedding-vision`。未配置或请求失败时，所有调用方都会自动回退到
//! BM25 + lexical similarity，检索功能不会中断。

use std::path::PathBuf;
use std::sync::OnceLock;

use dirs::cache_dir;

const DEFAULT_EMBEDDING_ENDPOINT: &str =
    "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings";
const DEFAULT_EMBEDDING_MODEL: &str = "doubao-embedding-vision";

pub trait EmbeddingProvider: Sync + Send {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>>;
    fn is_ready(&self) -> bool;
    fn try_load(&self);
}

/// 默认 provider：永远返回 None，让调用方自动走 BM25 / lexical 降级路径。
pub struct NullEmbeddingProvider;

impl EmbeddingProvider for NullEmbeddingProvider {
    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
    fn embed_batch(&self, _texts: &[String]) -> Option<Vec<Vec<f32>>> {
        None
    }
    fn is_ready(&self) -> bool {
        false
    }
    fn try_load(&self) {
        // no-op
    }
}

static GLOBAL_PROVIDER: OnceLock<Box<dyn EmbeddingProvider>> = OnceLock::new();

/// 旧 API 兼容：返回默认 cache 路径。fastembed 已移除，但保留路径以便外部
/// embedding provider（如挂 HTTP 服务时本地落盘缓存）继续复用。
#[allow(dead_code)]
fn default_cache_dir() -> PathBuf {
    cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("rust_tools")
        .join("embedding_cache")
}

fn default_provider() -> Box<dyn EmbeddingProvider> {
    Box::new(NullEmbeddingProvider)
}

fn global_provider() -> &'static dyn EmbeddingProvider {
    GLOBAL_PROVIDER.get_or_init(default_provider).as_ref()
}

pub fn set_provider(provider: Box<dyn EmbeddingProvider>) {
    let _ = GLOBAL_PROVIDER.set(provider);
}

pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    global_provider().embed(text)
}

pub fn embed_texts(texts: &[String]) -> Option<Vec<Vec<f32>>> {
    global_provider().embed_batch(texts)
}

pub fn is_ready() -> bool {
    global_provider().is_ready()
}

/// 启动时从配置安装远程 embedding provider（默认火山方舟 / 任意 OpenAI 兼容端点）。
/// 任意一步缺失都保持 NullEmbeddingProvider（即当前 BM25/lexical 行为），不报错。
pub fn warm_up() {
    use crate::ai::config_schema::AiConfig;
    use crate::commonw::configw;

    let cfg = configw::get_all_config();
    let get_nonempty = |key: &str| {
        cfg.get_opt(key)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };

    // 默认开启：只要配了 api_key（专用 key 或当前端点对应的通用 key）就启用语义检索；
    // 显式设为 false 可关闭。没配 key 时下面会自然回退，不安装 provider。
    let enabled = cfg
        .get_opt(AiConfig::EMBEDDING_ENABLE)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true);
    if !enabled {
        return;
    }

    // endpoint：默认使用 Coding Plan 专用的 /api/coding/v3；/api/v3 会产生额外费用。
    let endpoint = get_nonempty(AiConfig::EMBEDDING_ENDPOINT)
        .unwrap_or_else(|| DEFAULT_EMBEDDING_ENDPOINT.to_string());

    // api_key：专用 key 优先；无专用 key 时按端点回退到对应平台的通用 key。
    // 其他 OpenAI 兼容端点必须显式配置 ai.embedding.api_key，避免误用无关平台的 key。
    let endpoint_lc = endpoint.to_ascii_lowercase();
    let api_key = get_nonempty(AiConfig::EMBEDDING_API_KEY).or_else(|| {
        if endpoint_lc.contains("volces.com") {
            get_nonempty(AiConfig::MODEL_VOLCANO_API_KEY)
        } else if endpoint_lc.contains("aliyuncs.com") {
            get_nonempty(AiConfig::MODEL_ALIYUN_API_KEY)
        } else {
            None
        }
    });
    let Some(api_key) = api_key else {
        // 没配 key：保持降级，不安装。
        return;
    };

    let model = get_nonempty(AiConfig::EMBEDDING_MODEL)
        .unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string());

    let timeout_ms = cfg
        .get_opt(AiConfig::EMBEDDING_TIMEOUT_MS)
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(8000);

    set_provider(Box::new(RemoteEmbeddingProvider {
        endpoint,
        api_key,
        model,
        timeout: std::time::Duration::from_millis(timeout_ms),
    }));
}

/// 远程 embedding provider：走 OpenAI 兼容 `/embeddings` 接口。
///
/// 关键约定：**任何失败（网络 / 鉴权 / 解析）都返回 `None`**，让所有调用方
/// （memory_store::search 的向量重排、rag_tools、dedup 等）自动回退到
/// BM25 + lexical 的当前行为，绝不因为 embedding 出问题而中断检索或报错。
struct RemoteEmbeddingProvider {
    endpoint: String,
    api_key: String,
    model: String,
    timeout: std::time::Duration,
}

impl RemoteEmbeddingProvider {
    /// 在独立 std 线程里跑 reqwest::blocking，避免在 tokio 运行时线程上
    /// 直接使用 blocking client 触发 "blocking inside runtime" panic。
    fn request(&self, inputs: Vec<String>) -> Option<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Some(Vec::new());
        }
        let endpoint = self.endpoint.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let timeout = self.timeout;

        let handle = std::thread::spawn(move || -> Option<Vec<Vec<f32>>> {
            let client = reqwest::blocking::Client::builder()
                .timeout(timeout)
                .connect_timeout(std::time::Duration::from_secs(5))
                .build()
                .ok()?;

            let body = serde_json::json!({
                "model": model,
                "input": inputs,
                "encoding_format": "float",
            });

            let resp = client
                .post(&endpoint)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let json: serde_json::Value = resp.json().ok()?;
            let data = json.get("data")?.as_array()?;
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(data.len());
            for item in data {
                let arr = item.get("embedding")?.as_array()?;
                let vec: Vec<f32> = arr
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                if vec.is_empty() {
                    return None;
                }
                out.push(vec);
            }
            if out.len() != data.len() || out.is_empty() {
                return None;
            }
            Some(out)
        });

        handle.join().ok().flatten()
    }
}

impl EmbeddingProvider for RemoteEmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut v = self.request(vec![text.to_string()])?;
        if v.is_empty() {
            None
        } else {
            Some(v.remove(0))
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        let out = self.request(texts.to_vec())?;
        if out.len() == texts.len() {
            Some(out)
        } else {
            None
        }
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn try_load(&self) {}
}
