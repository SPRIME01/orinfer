//! Bounded JSONL input and an append-only, identity-checked result journal.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, Write},
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::Path,
};

const INPUT_LIMIT: u64 = 32 << 20;
const RESULT_LIMIT: u64 = 256 << 20;

fn line(reader: &mut impl BufRead, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err(format!("JSONL line exceeds {limit} bytes"));
    }
    Ok(bytes)
}

pub(super) struct Record {
    pub id: String,
    pub line: usize,
    pub bytes: usize,
    pub body: Result<Value>,
}
fn record(bytes: &[u8], line: usize) -> Record {
    let mut id = format!("line-{line}");
    let body = (|| {
        let mut value: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let object = value.as_object_mut().ok_or("Expected a JSON object")?;
        if let Some(custom) = object.remove("custom_id") {
            let custom = custom
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 256)
                .ok_or("custom_id must be a nonempty string of at most 256 bytes")?;
            id = custom.to_owned();
        }
        if object.contains_key("body")
            || object.contains_key("method")
            || object.contains_key("url")
        {
            if object.get("method").and_then(Value::as_str) != Some("POST")
                || object.get("url").and_then(Value::as_str) != Some("/v1/chat/completions")
            {
                return Err("Batch requests require POST /v1/chat/completions".into());
            }
            let body = object.remove("body").ok_or("Missing request body")?;
            object.remove("method");
            object.remove("url");
            if !object.is_empty() {
                return Err("Unknown batch envelope field".into());
            }
            Ok(body)
        } else {
            Ok(value)
        }
    })();
    Record {
        id,
        line,
        bytes: bytes.len(),
        body,
    }
}

pub(super) struct Input {
    reader: BufReader<File>,
    line: usize,
    pub ids: BTreeMap<String, usize>,
    pub sha256: String,
    metadata: std::fs::Metadata,
}
impl Input {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| format!("Open input {}: {e}", path.display()))?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() {
            return Err("Input must be a regular JSONL file".into());
        }
        let mut reader = BufReader::new(file);
        let mut digest = Sha256::new();
        let mut ids = BTreeMap::new();
        let mut n = 0;
        loop {
            let bytes = line(&mut reader, INPUT_LIMIT)?;
            if bytes.is_empty() {
                break;
            }
            digest.update(&bytes);
            n += 1;
            if bytes.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let r = record(&bytes, n);
            if let Some(previous) = ids.insert(r.id.clone(), n) {
                return Err(format!(
                    "Duplicate custom_id {:?} at lines {previous} and {n}",
                    r.id
                ));
            }
        }
        reader.rewind().map_err(|e| e.to_string())?;
        Ok(Self {
            reader,
            line: 0,
            ids,
            sha256: hex::encode(digest.finalize()),
            metadata,
        })
    }
    pub fn next(&mut self) -> Result<Option<Record>> {
        let current = self
            .reader
            .get_ref()
            .metadata()
            .map_err(|e| e.to_string())?;
        if current.len() != self.metadata.len()
            || current.modified().ok() != self.metadata.modified().ok()
        {
            return Err("Input changed during batch execution".into());
        }
        loop {
            let bytes = line(&mut self.reader, INPUT_LIMIT)?;
            if bytes.is_empty() {
                return Ok(None);
            }
            self.line += 1;
            if !bytes.iter().all(u8::is_ascii_whitespace) {
                return Ok(Some(record(&bytes, self.line)));
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Outcome {
    Success,
    Truncated,
    Failed,
}
fn outcome(value: &Value) -> Result<Outcome> {
    if !value["error"].is_null() && value["response"].is_null() {
        if !value["error"]["message"].is_string() {
            return Err("Malformed result error".into());
        }
        return Ok(Outcome::Failed);
    }
    if !value["error"].is_null() || value["response"]["status_code"] != 200 {
        return Err("Malformed result response".into());
    }
    match value["response"]["body"]["choices"][0]["finish_reason"].as_str() {
        Some("stop" | "tool_calls") => Ok(Outcome::Success),
        Some("length") => Ok(Outcome::Truncated),
        _ => Err("Missing or invalid result finish_reason".into()),
    }
}

pub(super) struct Journal {
    file: File,
    pub states: BTreeMap<String, Outcome>,
    synced: Instant,
}
impl Journal {
    pub fn open(path: &Path, resume: bool, identity: &Value, input: &Input) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        if !resume {
            options.create_new(true);
        }
        let mut file = options
            .open(path)
            .map_err(|e| format!("Open output {}: {e}", path.display()))?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file()
            || (metadata.dev() == input.metadata.dev() && metadata.ino() == input.metadata.ino())
        {
            return Err("Output must be a regular file distinct from input".into());
        }
        // SAFETY: A live descriptor owned for the whole run; concurrent writers
        // cannot inspect/repair/append this journal while the lock is held.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err("Batch output is locked by another process".into());
        }
        let mut meta = path.as_os_str().to_os_string();
        meta.push(".meta.json");
        let mut states = BTreeMap::new();
        if resume {
            let stored: Value = serde_json::from_reader(
                File::open(&meta).map_err(|e| format!("Open resume identity: {e}"))?,
            )
            .map_err(|e| e.to_string())?;
            if &stored != identity {
                return Err(
                    "Resume identity mismatch: input, model or request defaults changed".into(),
                );
            }
            let mut reader = BufReader::new(&mut file);
            let mut end = 0;
            loop {
                let bytes = line(&mut reader, RESULT_LIMIT)?;
                if bytes.is_empty() {
                    break;
                }
                if !bytes.ends_with(b"\n") {
                    break;
                }
                let value: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("Corrupt completed result: {e}"))?;
                let id = value["custom_id"]
                    .as_str()
                    .ok_or("Result custom_id missing")?;
                if !input.ids.contains_key(id) {
                    return Err("Result ID does not belong to input".into());
                }
                states.insert(id.to_owned(), outcome(&value)?);
                end = reader.stream_position().map_err(|e| e.to_string())?;
            }
            // An interrupted final write is uncommitted. Interior corruption
            // above is an error and never causes truncation of valid results.
            file.set_len(end).map_err(|e| e.to_string())?;
        } else {
            let mut meta = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&meta)
                .map_err(|e| format!("Create batch identity: {e}"))?;
            serde_json::to_writer(&mut meta, identity).map_err(|e| e.to_string())?;
            meta.write_all(b"\n").map_err(|e| e.to_string())?;
            meta.sync_all().map_err(|e| e.to_string())?;
        }
        file.seek(std::io::SeekFrom::End(0))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file,
            states,
            synced: Instant::now(),
        })
    }
    pub fn done(&self, id: &str) -> bool {
        self.states.get(id).is_some_and(|s| *s != Outcome::Failed)
    }
    pub fn append(&mut self, value: &Value) -> Result<()> {
        let id = value["custom_id"].as_str().ok_or("Result ID missing")?;
        let status = outcome(value)?;
        let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
        if bytes.len() as u64 >= RESULT_LIMIT {
            return Err("Result exceeds JSONL line limit".into());
        }
        self.file
            .write_all(&bytes)
            .and_then(|_| self.file.write_all(b"\n"))
            .map_err(|e| e.to_string())?;
        self.states.insert(id.to_owned(), status);
        if self.synced.elapsed().as_secs() >= 1 {
            self.sync()?;
        }
        Ok(())
    }
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_data().map_err(|e| e.to_string())?;
        self.synced = Instant::now();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn line_limit_is_checked_before_growing_to_the_rest_of_the_input() {
        let mut input = std::io::Cursor::new(b"123456789\nrest\n");
        assert!(line(&mut input, 8).is_err());
        assert_eq!(input.position(), 9);
        let mut input = std::io::Cursor::new(b"{}\r\n\n");
        assert_eq!(line(&mut input, 8).unwrap(), b"{}\r\n");
    }
}
