```markdown
# video-pipeline Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the `video-pipeline` TypeScript codebase. It covers file naming, import/export styles, commit message conventions, and testing patterns. This guide helps contributors maintain consistency and efficiency when working on the project.

## Coding Conventions

### File Naming
- Use **camelCase** for all file names.
  - Example: `videoProcessor.ts`, `frameExtractor.test.ts`

### Import Style
- Use **relative imports** for internal modules.
  - Example:
    ```typescript
    import { extractFrames } from './frameExtractor';
    ```

### Export Style
- Use **named exports** for all modules.
  - Example:
    ```typescript
    // In videoProcessor.ts
    export function processVideo(input: string): void { ... }
    ```

### Commit Messages
- Follow the **Conventional Commits** standard.
- Use the `build` prefix for build-related changes.
  - Example:
    ```
    build: update TypeScript to version 4.9.5
    ```

## Workflows

### Build Workflow
**Trigger:** When you need to build the project (e.g., after making changes)
**Command:** `/build`

1. Ensure all dependencies are installed.
2. Run the TypeScript compiler to build the project.
   - Example command: `tsc`
3. Verify that the output files are generated as expected.

### Test Workflow
**Trigger:** When you want to run the test suite
**Command:** `/test`

1. Locate all test files matching the `*.test.*` pattern.
2. Run the test runner (framework is unknown; use the project's standard).
   - Example: `npm test` or `ts-node frameExtractor.test.ts`
3. Review the output to ensure all tests pass.

## Testing Patterns

- Test files follow the `*.test.*` naming convention.
  - Example: `videoProcessor.test.ts`
- The testing framework is not specified; check the project documentation or scripts for details.
- Tests are likely written in TypeScript and may use assertions similar to:
  ```typescript
  import { processVideo } from './videoProcessor';

  test('processVideo handles valid input', () => {
    expect(processVideo('input.mp4')).toBe(/* expected value */);
  });
  ```

## Commands

| Command   | Purpose                              |
|-----------|--------------------------------------|
| /build    | Build the TypeScript project         |
| /test     | Run all tests in the codebase        |
```
