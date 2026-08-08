//! A small, strict JSON reader — enough for a manifest and a safetensors header.
//!
//! M3 is the first milestone that has to read a *file format* rather than an
//! integer stream, and two of them: `artifact/manifest.json` carries the
//! guardrails the bot refuses to start without, and a safetensors file begins
//! with a JSON header naming every tensor's dtype, shape, and byte range.
//!
//! Why not `serde_json`. The crate stays dependency-free to protect the
//! 10,000-file unpacked budget, and the measured price of that budget is now
//! known rather than assumed (`tools/dependency_budget.py`) — this is a
//! judgement call, not an article of faith. What tips it is that both inputs
//! are files *we* generate, from `tools/convert_artifact.py`, with a fixed
//! writer; the parser needs to be correct on that grammar and to refuse
//! everything else loudly, not to be fast or forgiving. Two hundred lines buy
//! that, and the round-trip test in the converter checks our writer against
//! the reference implementation, so neither side is grading its own homework.
//!
//! Strictness is deliberate: no trailing commas, no comments, no NaN, no
//! duplicate keys. A malformed artifact must fail at load, where there is a
//! human to read the message, rather than at the first move of a rated game.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(map) => map.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// A JSON number that is exactly an integer. `21.5` is an error, not a 21.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(n) if n.fract() == 0.0 && n.is_finite() => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(v) => Some(v),
            _ => None,
        }
    }

    /// `obj.field("architecture").field("n_blocks")`, with the path in errors.
    pub fn field(&self, key: &str) -> Result<&Json, String> {
        self.get(key).ok_or_else(|| format!("missing key {key:?}"))
    }

    pub fn str_field(&self, key: &str) -> Result<&str, String> {
        self.field(key)?
            .as_str()
            .ok_or_else(|| format!("{key:?} is not a string"))
    }

    pub fn int_field(&self, key: &str) -> Result<i64, String> {
        self.field(key)?
            .as_i64()
            .ok_or_else(|| format!("{key:?} is not an integer"))
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    parser.skip_ws();
    let value = parser.value()?;
    parser.skip_ws();
    if parser.at != parser.bytes.len() {
        return Err(format!("trailing input at byte {}", parser.at));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.at) {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Result<u8, String> {
        self.bytes
            .get(self.at)
            .copied()
            .ok_or_else(|| "unexpected end of JSON".to_string())
    }

    fn eat(&mut self, expected: u8) -> Result<(), String> {
        if self.peek()? != expected {
            return Err(format!(
                "expected {:?} at byte {}, found {:?}",
                expected as char,
                self.at,
                self.peek()? as char
            ));
        }
        self.at += 1;
        Ok(())
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(format!("bad literal at byte {}", self.at))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::String(self.string()?)),
            b't' => self.literal("true", Json::Bool(true)),
            b'f' => self.literal("false", Json::Bool(false)),
            b'n' => self.literal("null", Json::Null),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek()? == b'}' {
            self.at += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            let value = self.value()?;
            if map.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate key {key:?}"));
            }
            self.skip_ws();
            match self.peek()? {
                b',' => self.at += 1,
                b'}' => {
                    self.at += 1;
                    return Ok(Json::Object(map));
                }
                other => return Err(format!("expected ',' or '}}', found {:?}", other as char)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek()? == b']' {
            self.at += 1;
            return Ok(Json::Array(out));
        }
        loop {
            self.skip_ws();
            out.push(self.value()?);
            self.skip_ws();
            match self.peek()? {
                b',' => self.at += 1,
                b']' => {
                    self.at += 1;
                    return Ok(Json::Array(out));
                }
                other => return Err(format!("expected ',' or ']', found {:?}", other as char)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        // A byte buffer, not a `String`: unescaped multi-byte UTF-8 arrives one
        // byte at a time, and a continuation byte is not a `char`. The input
        // came from a `&str`, so the assembled bytes are valid by construction
        // and the final conversion cannot fail — but it is checked anyway,
        // because "cannot fail" is how a parser acquires an `unwrap`.
        let mut out: Vec<u8> = Vec::new();
        loop {
            let b = self.peek()?;
            self.at += 1;
            match b {
                b'"' => {
                    return String::from_utf8(out).map_err(|_| "string is not UTF-8".to_string())
                }
                b'\\' => {
                    let esc = self.peek()?;
                    self.at += 1;
                    let ch = match esc {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => self.unicode_escape()?,
                        other => return Err(format!("bad escape \\{:?}", other as char)),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => out.push(b),
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char, String> {
        let hex = self
            .bytes
            .get(self.at..self.at + 4)
            .ok_or_else(|| "truncated \\u escape".to_string())?;
        let text = std::str::from_utf8(hex).map_err(|_| "bad \\u escape".to_string())?;
        let code = u32::from_str_radix(text, 16).map_err(|_| "bad \\u escape".to_string())?;
        self.at += 4;
        // Surrogate pairs are not produced by our writer; refusing them beats
        // pretending to handle them.
        char::from_u32(code).ok_or_else(|| format!("\\u{code:04x} is not a scalar value"))
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        if self.peek()? == b'-' {
            self.at += 1;
        }
        while let Some(&b) = self.bytes.get(self.at) {
            if b.is_ascii_digit() || b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
                self.at += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| "bad number".to_string())?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| format!("bad number {text:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_nested_document() {
        let value = parse(r#"{"a":[1,2.5,-3e2],"b":{"c":"x\ny"},"d":true,"e":null}"#).unwrap();
        assert_eq!(value.field("a").unwrap().as_array().unwrap().len(), 3);
        assert_eq!(value.get("a").unwrap().as_array().unwrap()[2].as_f64(), Some(-300.0));
        assert_eq!(value.field("b").unwrap().str_field("c").unwrap(), "x\ny");
        assert_eq!(value.get("d"), Some(&Json::Bool(true)));
        assert_eq!(value.get("e"), Some(&Json::Null));
    }

    #[test]
    fn integers_must_be_integral() {
        let value = parse(r#"{"n":21,"f":21.5}"#).unwrap();
        assert_eq!(value.int_field("n").unwrap(), 21);
        assert!(value.int_field("f").is_err());
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "{",
            "{\"a\":1,}",
            "[1,2",
            "{\"a\":1} trailing",
            "{\"a\":1,\"a\":2}",
            "{'a':1}",
        ] {
            assert!(parse(bad).is_err(), "should have rejected {bad:?}");
        }
    }

    #[test]
    fn handles_unicode_and_escapes() {
        let value = parse(r#"{"k":"é\t\"q\"", "u":"café"}"#).unwrap();
        assert_eq!(value.str_field("k").unwrap(), "é\t\"q\"");
        assert_eq!(value.str_field("u").unwrap(), "café");
    }
}
