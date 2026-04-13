---
name: readme-tech
description: "Generate README (usage guide) and TECHNICAL.md (implementation details) for a module or directory. Invoke when the user asks to create documentation for a component."
argument-hint: "[directory-or-module-path]"
allowed-tools: Read Glob Grep Bash(ls *) Bash(git log *) Bash(git diff *) Write
---

# readme-tech: Generate README + Technical Documentation

For the given module or directory, generate two documents:

1. **README.md** - User-facing usage guide
2. **TECHNICAL.md** - Developer-facing technical documentation

## Target

If `$ARGUMENTS` is provided, use it as the target directory/module path.
If empty, ask the user which module to document.

## Step 1: Explore the target

- Use Glob to find all source files in the target directory
- Read the key files: `__init__.py`, main modules, config files
- Identify: purpose, public API, dependencies, usage patterns

## Step 2: Generate README.md

Write `README.md` in the target directory with these sections:

- **Title + one-line description** (Chinese)
- **Quick Start** - minimal steps to get it running
- **Config** - how to configure via config files (if applicable)
- **Code Usage** - Python code examples for programmatic use
- **Parameter Reference** - table of all configurable parameters
- **Notes** - gotchas, requirements, common issues

Style:
- Use Chinese for all prose
- Keep code examples runnable and realistic
- Include tables for parameter references
- Be concise; no filler text

## Step 3: Generate TECHNICAL.md

Write `TECHNICAL.md` in the target directory with these sections:

- **Architecture Overview** - ASCII diagram showing component relationships
- **Module Structure** - file tree with one-line descriptions
- **Protocol / API** - data formats, message structures, function signatures
- **Core Classes** - detailed explanation of each class with method signatures
- **Data Flow** - step-by-step trace of a typical request/operation
- **Performance** - profiling info, typical latency, optimization notes (if applicable)

Style:
- Use Chinese for prose, English for code/identifiers
- Include ASCII diagrams for architecture
- Show actual code signatures and type annotations
- Include tables for data specifications (shapes, dtypes, etc.)

## Step 4: Report

Tell the user which files were created and summarize the contents.
