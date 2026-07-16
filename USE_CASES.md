# SWC Viewer Use Cases

This document describes what the SWC Viewer plugin can do through concrete,
repeatable user workflows. Each use case also serves as a manual test case for
confirming that the documented capability works in napari.

Use cases are added as workflows are described and refined. They document
observable behavior rather than implementation details.

## Running the Manual Tests

Unless a use case says otherwise:

1. From the repository root, launch napari with the development build:

   ```bash
   pixi run napari
   ```

2. Open the **SWC Viewer** plugin from napari's **Plugins** menu.
3. Follow the selected use case from a clean napari session.
4. Record the result in that use case's **Manual verification** section. A use
   case remains `Not run` until someone actually performs it in napari.

## Use-Case Index

No use cases have been recorded yet.

## Use-Case Template

Copy this section when adding a use case. Remove guidance in parentheses and
replace `XXX` with the next sequential identifier.

### UC-XXX: Descriptive Workflow Name

**Capability**

(Explain what the plugin lets the user accomplish and why the workflow is
useful.)

**Prerequisites**

- (List required input files, atlas data, existing layers, configuration, or
  starting state.)

**Steps and expected results**

1. **Action:** (Describe one user action using the visible UI labels.)
   **Expected:** (Describe the observable result of that action.)
2. **Action:** (Continue until the workflow and its final outcome are
   complete.)
   **Expected:** (Describe the observable result.)

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None
