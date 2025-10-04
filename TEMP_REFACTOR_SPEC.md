# SQLTest CLI Refactor – Remaining Work Specification

## Context Recap
- **Current state**: `sqltest/cli/main.py` now acts primarily as an orchestrator. Global command groups (`profile`, `validate`, `test`, `report`, `init`, `db`, `config`) reside in `sqltest/cli/commands/`, each owning its helper logic.
- **Outstanding inline code**: Business rule execution (`business_rules` command) and display/export helpers (`show_validation_summary`, `show_field_validation_summary`, etc.) still live inside `main.py`.
- **Next objective**: Complete the CLI modularisation (Step 2) and then streamline the entry point and workflows (Step 3).

## Step 2 – Remaining Tasks
1. **Business Rule Command Extraction**
   - Create `sqltest/cli/commands/business_rules.py`.
   - Move the `business_rules` command and its helper functions/results exporters from `main.py` to the new module.
   - Ensure imports (`BusinessRuleValidator`, `ValidationSummary`, etc.) are localised and that verbose/error handling matches other command modules.
   - Update `main.py` to register the new command via `cli.add_command(business_rules_command)`.

2. **Shared Helper Consolidation**
   - Review leftover helper functions in `main.py` (`show_validation_summary`, `show_field_validation_result`, etc.).
   - Split into:
     - Helpers that are specific to a command module → relocate beside the owning command (`business_rules`, `validate`, etc.).
     - Truly shared helpers → move to `sqltest/cli/utils.py` (but keep surface area minimal).
   - Confirm no residual command-specific logic remains in `main.py` after the move.

3. **Code Hygiene**
   - Remove now-unused imports from `main.py`.
   - Ensure each command module exposes only what `main.py` needs (typically a single command/group object).
   - Re-run `python -m compileall sqltest/cli` to validate import integrity.

## Step 3 – Entry Point Streamlining (to start after Step 2)
1. **Main CLI Cleanup**
   - Reduce `sqltest/cli/main.py` to:
     - global context setup
     - dashboard display function (optional)
     - command registration calls
     - app execution guard
   - Document the registration order and add comments for new contributors.

2. **`sqltest init` Workflow Refresh**
   - Audit `commands/init.py` for user experience improvements (e.g., confirm prompts, optional features, consistent output).
   - Consider extracting repeated YAML templates/fixtures into data files or simplified factory functions if warranted.
   - Update README/help strings if the workflow changes.

3. **Documentation & Tests**
   - Ensure CLI usage examples (docs/README) reflect the modular commands and new init experience.
   - Add/adjust smoke tests (if available) covering `--help` output and key commands.

## Acceptance Criteria
- `sqltest/cli/main.py` contains no command implementations—only registration utilities and dashboard logic.
- Each command module is importable and exposes a single command/group.
- `python -m sqltest.cli.main --help` lists the new command names without regressions.
- Temporary specification file (`TEMP_REFACTOR_SPEC.md`) can be deleted once tasks are complete.
